"""
Global Encoding Manager for Stream Engine.

Wraps the model's scheduler-side or local image encoder to:
  1. Provide a unified ``encode`` interface for frame-level encoding.
  2. Track the total number of multimodal tokens produced across all
     encoding calls (for monitoring / budget control).

The manager supports both:
  * Qwen-style ``model.visual(...)``
  * LLaVA / LLaVA-OneVision-style ``model.get_image_feature(...)`` or
    ``model.get_image_features(...)``

The wrapper is architecture-agnostic in its interface but aware of the
FLAT vs DEEPSTACK distinction internally so that downstream consumers
can inspect ``feature_layout``.

Usage::

    from sglang.srt.stream_engine.encoding_mgr import GlobalEncodingManager

    mgr = GlobalEncodingManager(model=model)
    image_embeds = mgr.encode(pixel_values, grid_thw=image_grid_thw)
    print(mgr.total_mm_tokens)
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.stream_engine.embedding_cache import EmbeddingCacheManager
from sglang.srt.stream_engine.types import Cluster
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vision feature layout enum  (mirrors types.py but kept self-contained)
# ---------------------------------------------------------------------------

class VisionFeatureLayout(str, Enum):
    """Indicates how the vision encoder returns features."""
    FLAT = "flat"           # Qwen2-VL: single merged tensor  (seq, D)
    DEEPSTACK = "deepstack" # Qwen3-VL: merged + deepstack    (seq, D*(1+N))


# ---------------------------------------------------------------------------
# Architecture identification helpers
# ---------------------------------------------------------------------------

# Known architecture strings → layout mapping
_ARCH_TO_LAYOUT = {
    "Qwen2VLForConditionalGeneration":     VisionFeatureLayout.FLAT,
    "Qwen2_5VLForConditionalGeneration":   VisionFeatureLayout.FLAT,
    "Qwen3VLForConditionalGeneration":     VisionFeatureLayout.DEEPSTACK,
    "Qwen3VLMoeForConditionalGeneration":  VisionFeatureLayout.DEEPSTACK,
    "LlavaOnevisionForConditionalGeneration": VisionFeatureLayout.FLAT,
    "LlavaForConditionalGeneration": VisionFeatureLayout.FLAT,
    "LlavaQwenForCausalLM": VisionFeatureLayout.FLAT,
    "LlavaLlamaForCausalLM": VisionFeatureLayout.FLAT,
    "LlavaMistralForCausalLM": VisionFeatureLayout.FLAT,
}

# Substring heuristics for model path / model_type
_PATH_HINTS = [
    ("qwen3-vl", VisionFeatureLayout.DEEPSTACK),
    ("qwen3_vl", VisionFeatureLayout.DEEPSTACK),
    ("qwen2.5-vl", VisionFeatureLayout.FLAT),
    ("qwen2_5-vl", VisionFeatureLayout.FLAT),
    ("qwen2-vl", VisionFeatureLayout.FLAT),
    ("qwen2_vl", VisionFeatureLayout.FLAT),
    ("llava-onevision", VisionFeatureLayout.FLAT),
    ("llava_onevision", VisionFeatureLayout.FLAT),
]


def _count_mm_tokens(output: Any) -> int:
    """Count encoded multimodal tokens from heterogeneous encoder outputs."""
    if isinstance(output, torch.Tensor):
        if output.dim() <= 1:
            return int(output.shape[0]) if output.dim() == 1 else 1
        hidden_dim = max(int(output.shape[-1]), 1)
        return int(output.numel() // hidden_dim)

    if (
        isinstance(output, tuple)
        and len(output) == 2
        and isinstance(output[0], torch.Tensor)
    ):
        # Qwen3-VL-style tuple: (main_embedding, [deepstack_layer_embeddings...]).
        return _count_mm_tokens(output[0])

    if isinstance(output, (list, tuple)):
        return sum(_count_mm_tokens(item) for item in output)

    raise TypeError(
        "Unsupported encoder output type for token counting: "
        f"{type(output)!r}"
    )


def detect_feature_layout(
    model: Optional[nn.Module] = None,
    hf_config=None,
    model_path: Optional[str] = None,
) -> VisionFeatureLayout:
    """Detect the vision feature layout for a given model.

    Detection priority:
      1. *hf_config.architectures* – most reliable.
      2. *hf_config.vision_config.deepstack_visual_indexes* attribute.
      3. *model_path* substring heuristics.

    Falls back to ``FLAT`` if nothing matches.
    """
    # --- from hf_config.architectures ---
    if hf_config is not None:
        archs = getattr(hf_config, "architectures", None) or []
        for arch in archs:
            if arch in _ARCH_TO_LAYOUT:
                return _ARCH_TO_LAYOUT[arch]

        # --- from deepstack_visual_indexes ---
        vision_cfg = getattr(hf_config, "vision_config", None)
        if vision_cfg is not None:
            ds_idx = getattr(vision_cfg, "deepstack_visual_indexes", None)
            if ds_idx is not None and len(ds_idx) > 0:
                return VisionFeatureLayout.DEEPSTACK

    # --- from model object ---
    if model is not None:
        cls_name = type(model).__name__
        if cls_name in _ARCH_TO_LAYOUT:
            return _ARCH_TO_LAYOUT[cls_name]
        # check for deepstack attribute
        if hasattr(model, "deepstack_visual_indexes"):
            ds_idx = model.deepstack_visual_indexes
            if ds_idx is not None and len(ds_idx) > 0:
                return VisionFeatureLayout.DEEPSTACK

    # --- from model_path substring ---
    if model_path is not None:
        path_lower = model_path.lower()
        for hint, layout in _PATH_HINTS:
            if hint in path_lower:
                return layout

    logger.warning(
        "Could not determine feature layout; defaulting to FLAT. "
        "model_path=%s", model_path,
    )
    return VisionFeatureLayout.FLAT


# ---------------------------------------------------------------------------
# Vision encoder wrapper
# ---------------------------------------------------------------------------

class _VisualEncoderWrapper(nn.Module):
    """Thin wrapper around ``model.visual`` that counts output tokens.

    This wrapper is designed to be *drop-in compatible* with the original
    ``model.visual`` so that the rest of the model (e.g. ``get_image_feature``)
    can call it transparently.

    The wrapper keeps a running tally of multimodal tokens produced in
    ``total_mm_tokens`` (thread-safe counter).
    """

    def __init__(self, original_visual: nn.Module):
        super().__init__()
        self._original = original_visual
        self._total_mm_tokens: int = 0
        self._lock = threading.Lock()

    # ------ forwarding properties expected by caller code ------

    @property
    def dtype(self):
        return self._original.dtype

    @property
    def device(self):
        return self._original.device

    # ------ core forward ------

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Call the underlying vision encoder and track token count.

        The number of *multimodal tokens* equals the first dimension (sequence
        length) of the output tensor — for both FLAT and DEEPSTACK layouts the
        first dim is ``seq_len``.
        """
        if grid_thw is None:
            raise ValueError("Qwen visual encoding requires `grid_thw`.")

        output = self._original(x, grid_thw=grid_thw, **kwargs)
        num_tokens = _count_mm_tokens(output)

        with self._lock:
            self._total_mm_tokens += num_tokens
        return output

    # ------ statistics ------

    @property
    def total_mm_tokens(self) -> int:
        """Total multimodal tokens produced since creation / last reset."""
        return self._total_mm_tokens

    def reset_stats(self) -> None:
        """Reset the token counter to zero."""
        with self._lock:
            self._total_mm_tokens = 0

    # ------ transparent attribute access ------

    def __getattr__(self, name: str):
        """Delegate attribute access to the original visual encoder.

        This ensures that any code accessing sub-modules, parameters, or
        methods on ``model.visual`` continues to work (e.g.
        ``model.visual.patch_embed``).
        """
        # First check nn.Module's own attributes
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._original, name)


class _LlavaImageFeatureWrapper(nn.Module):
    """Wrapper for LLaVA-style image feature APIs.

    Prefers the scheduler-side SGLang ``get_image_feature`` interface and falls
    back to the HuggingFace ``get_image_features`` helper for local encoding.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self._model = model
        inner_model = getattr(model, "model", None)
        self._hf_inner_model = inner_model if isinstance(inner_model, nn.Module) else None
        self._total_mm_tokens: int = 0
        self._lock = threading.Lock()

    def _iter_candidate_modules(self):
        for root in (self._model, self._hf_inner_model):
            if root is None:
                continue
            for module_name in ("vision_tower", "multi_modal_projector"):
                module = getattr(root, module_name, None)
                if module is not None:
                    yield module

    @property
    def dtype(self):
        for module in self._iter_candidate_modules():
            try:
                return next(module.parameters()).dtype
            except StopIteration:
                continue
        return getattr(self._model, "dtype", torch.float16)

    @property
    def device(self):
        for module in self._iter_candidate_modules():
            try:
                return next(module.parameters()).device
            except StopIteration:
                continue
        return torch.device("cpu")

    @property
    def total_mm_tokens(self) -> int:
        return self._total_mm_tokens

    def reset_stats(self) -> None:
        with self._lock:
            self._total_mm_tokens = 0

    @staticmethod
    def _normalize_image_sizes(
        image_sizes: Optional[Any],
    ) -> Optional[Any]:
        if image_sizes is None:
            return None
        if isinstance(image_sizes, torch.Tensor):
            if image_sizes.dim() == 1:
                return image_sizes.reshape(1, -1)
            return image_sizes
        if isinstance(image_sizes, tuple):
            return [image_sizes]
        return image_sizes

    def _split_inputs(
        self,
        pixel_values: torch.Tensor,
        image_sizes: Optional[Any],
    ) -> List[Tuple[torch.Tensor, Optional[Any]]]:
        normalized_sizes = self._normalize_image_sizes(image_sizes)
        if normalized_sizes is None:
            return [(pixel_values, None)]

        if isinstance(normalized_sizes, torch.Tensor):
            num_images = int(normalized_sizes.shape[0])
        elif isinstance(normalized_sizes, list):
            num_images = len(normalized_sizes)
        else:
            return [(pixel_values, normalized_sizes)]

        if num_images <= 1:
            return [(pixel_values, normalized_sizes)]

        if pixel_values.dim() >= 4 and int(pixel_values.shape[0]) == num_images:
            split_batches: List[Tuple[torch.Tensor, Optional[Any]]] = []
            for idx in range(num_images):
                if isinstance(normalized_sizes, torch.Tensor):
                    cur_sizes = normalized_sizes[idx : idx + 1]
                else:
                    cur_sizes = normalized_sizes[idx : idx + 1]
                split_batches.append((pixel_values[idx : idx + 1], cur_sizes))
            return split_batches

        return [(pixel_values, normalized_sizes)]

    def _encode_with_hf_get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: Optional[Any],
        **kwargs,
    ) -> Any:
        hf_model = self._hf_inner_model if self._hf_inner_model is not None else self._model
        get_image_features = getattr(hf_model, "get_image_features")
        outputs: List[Any] = []
        for cur_pixels, cur_sizes in self._split_inputs(pixel_values, image_sizes):
            call_kwargs: Dict[str, Any] = {
                "pixel_values": cur_pixels,
                "image_sizes": cur_sizes,
            }
            for attr_name in (
                "vision_feature_layer",
                "vision_feature_select_strategy",
                "vision_aspect_ratio",
            ):
                attr_value = getattr(hf_model, attr_name, None)
                if attr_value is None:
                    config = getattr(hf_model, "config", None)
                    attr_value = (
                        getattr(config, attr_name, None) if config is not None else None
                    )
                if attr_value is not None:
                    call_kwargs[attr_name] = attr_value

            if self._hf_inner_model is not None:
                call_kwargs.setdefault(
                    "batch_num_images",
                    torch.tensor(
                        [int(cur_sizes.shape[0])] if isinstance(cur_sizes, torch.Tensor) else [1],
                        dtype=torch.long,
                        device=cur_pixels.device,
                    ),
                )
            call_kwargs.update(kwargs)

            try:
                outputs.append(get_image_features(**call_kwargs))
            except TypeError:
                call_kwargs.pop("vision_feature_layer", None)
                call_kwargs.pop("vision_feature_select_strategy", None)
                call_kwargs.pop("vision_aspect_ratio", None)
                call_kwargs.pop("batch_num_images", None)
                outputs.append(get_image_features(**call_kwargs))

        return outputs[0] if len(outputs) == 1 else outputs

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        image_sizes = kwargs.pop("image_sizes", None)
        del grid_thw  # unused for LLaVA-style encoders

        output = self._encode_with_hf_get_image_features(
            x, image_sizes, **kwargs
        )

        with self._lock:
            self._total_mm_tokens += _count_mm_tokens(output)
        return output

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)


# ---------------------------------------------------------------------------
# Global Encoding Manager
# ---------------------------------------------------------------------------

class GlobalEncodingManager:
    """Manages a wrapped vision encoder and tracks encoding statistics.

    Parameters
    ----------
    model : nn.Module
        The full VLM. Supported encoder interfaces are:
          * Qwen-style ``.visual``
          * LLaVA-style ``.get_image_feature(...)``
          * HF LLaVA-style ``.get_image_features(...)``

        For Qwen models, the manager will **replace** ``model.visual`` with a
        tracking wrapper. For LLaVA models, the original model is left intact
        and the manager owns a separate wrapper object.
    model_path : str, optional
        Model path, used only for architecture detection when *hf_config*
        is not available.
    hf_config : object, optional
        HuggingFace model config, used for architecture detection.

    After construction:
      * Qwen: ``model.visual`` is replaced by a ``_VisualEncoderWrapper``.
      * LLaVA: ``mgr.visual`` exposes the wrapped image-feature encoder.
      * ``mgr.feature_layout`` indicates FLAT or DEEPSTACK.
      * ``mgr.total_mm_tokens`` returns accumulated token count.
    """

    def __init__(
        self,
        model: nn.Module,
        model_path: Optional[str] = None,
        hf_config=None,
    ):
        has_qwen_visual = hasattr(model, "visual")
        has_llava_encoder = any(
            callable(getattr(model, attr, None))
            for attr in ("get_image_feature", "get_image_features")
        )
        if not has_qwen_visual and not has_llava_encoder:
            raise AttributeError(
                f"Model {type(model).__name__} does not expose a supported vision "
                "encoding interface (`visual`, `get_image_feature`, or "
                "`get_image_features`)."
            )

        # Detect architecture
        self.feature_layout = detect_feature_layout(
            model=model, hf_config=hf_config, model_path=model_path,
        )
        logger.info(
            "GlobalEncodingManager: feature_layout=%s (model=%s)",
            self.feature_layout.value, type(model).__name__,
        )

        # Wrap the vision encoder.
        if has_qwen_visual:
            original_visual = model.visual
            self._wrapper: nn.Module = _VisualEncoderWrapper(original_visual)
            model.visual = self._wrapper
            logger.info("GlobalEncodingManager: using Qwen visual wrapper.")
        else:
            self._wrapper = _LlavaImageFeatureWrapper(model)
            logger.info("GlobalEncodingManager: using LLaVA image-feature wrapper.")
        self._model = model
        self._single_frame_encoder: Optional[
            Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None

    # ------ convenience properties ------

    @property
    def total_mm_tokens(self) -> int:
        """Total multimodal tokens encoded since creation / last reset."""
        return self._wrapper.total_mm_tokens

    def reset_stats(self) -> None:
        """Reset the multimodal token counter."""
        self._wrapper.reset_stats()

    @property
    def visual(self) -> nn.Module:
        """Direct access to the wrapped encoding interface."""
        return self._wrapper

    def encode(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Any:
        """Encode pixel values through the wrapped vision encoder.

        This is a convenience method equivalent to calling
        the model-family-specific wrapped encoder entrypoint.
        """
        return self._wrapper(pixel_values, grid_thw=grid_thw, **kwargs)

    def register_single_frame_encoder(
        self,
        encoder_fn: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        """Register the runtime single-frame encoder used by stream engine.

        The callback should follow the same convention as ``StreamEngine._embed_frame``:
        ``encoder_fn(frame_image, hot_token_mask=None, mask_mode="off", cache_grid=False)``.
        """
        self._single_frame_encoder = encoder_fn

    def encode_single_frame(
        self,
        frame_image: Any,
        *,
        hot_token_mask: Optional[Any] = None,
        mask_mode: str = "off",
        cache_grid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a single frame using the registered frame encoder callback."""
        if self._single_frame_encoder is None:
            raise RuntimeError(
                "single_frame_encoder is not registered. "
                "Call register_single_frame_encoder(...) first."
            )

        try:
            result = self._single_frame_encoder(
                frame_image,
                hot_token_mask=hot_token_mask,
                mask_mode=mask_mode,
                cache_grid=cache_grid,
            )
        except TypeError:
            try:
                result = self._single_frame_encoder(
                    frame_image,
                    hot_token_mask=hot_token_mask,
                    mask_mode=mask_mode,
                )
            except TypeError:
                # Backward compatibility with callbacks that only accept image input.
                result = self._single_frame_encoder(frame_image)
        return result

    def encode_cluster(
        self,
        *,
        stream_id: str,
        cluster: Cluster,
        stage_mode: str,
        cold_token_threshold: float,
        cold_token_ratio: float = -1.0,
    ) -> Dict[str, Any]:
        """Encode a cluster and return unified stats for stream engine usage."""
        llm_embedding, grid_thw, retrieval_embedding, cache_stats = (
            EmbeddingCacheManager().encode_cluster(
                stream_id=stream_id,
                cluster=cluster,
                stage_mode=stage_mode,
                cold_token_threshold=cold_token_threshold,
                cold_token_ratio=cold_token_ratio,
            )
        )
        token_stats = {
            "cache_hit_ratio": float(cache_stats.get("cache_hit_ratio", 0.0)),
            "encoded_token_count": float(cache_stats.get("encoded_token_count", 0.0)),
            "reused_token_count": float(cache_stats.get("reused_token_count", 0.0)),
            "total_token_count": float(cache_stats.get("total_token_count", 0.0)),
            "sparse_vit_encoded_token_count": float(
                cache_stats.get("sparse_vit_encoded_token_count", 0.0)
            ),
            "sparse_vit_total_token_count": float(
                cache_stats.get("sparse_vit_total_token_count", 0.0)
            ),
            "sparse_vit_fallback_count": float(
                cache_stats.get("sparse_vit_fallback_count", 0.0)
            ),
        }
        selected_frames = cluster.get_selected_frames()
        selected_frame_count = int(len(selected_frames)) if selected_frames else 0
        cache_ratio = token_stats["cache_hit_ratio"]
        fwd_encoded_frames = float(selected_frame_count) * (1.0 - cache_ratio)

        return {
            "llm_embedding": llm_embedding,
            "grid_thw": grid_thw,
            "retrieval_embedding": retrieval_embedding,
            "cache_ratio": cache_ratio,
            "selected_frame_count": selected_frame_count,
            "fwd_encoded_frames": fwd_encoded_frames,
            "token_stats": token_stats,
        }

    def __repr__(self) -> str:
        return (
            f"GlobalEncodingManager("
            f"layout={self.feature_layout.value}, "
            f"total_mm_tokens={self.total_mm_tokens})"
        )
