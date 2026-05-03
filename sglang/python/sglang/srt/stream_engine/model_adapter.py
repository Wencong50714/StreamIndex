"""
Model-family adapter layer for the Stream Engine.

Isolates all model-family-specific behavior behind a clean interface so that
``StreamEngine`` itself remains model-agnostic.

Classes
-------
ModelFamily              – enum identifying the VLM family
detect_model_family      – resolve ModelFamily from a model path string
BaseStreamModelAdapter   – abstract base; defines the adapter contract
QwenVLAdapter            – Qwen2-VL / Qwen2.5-VL / Qwen3-VL implementation
LlavaOneVisionAdapter    – LLaVA-OneVision (7B and 0.5B) implementation
create_adapter           – factory that picks the right adapter for a model path
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.stream_engine.encoding_mgr import (
    GlobalEncodingManager,
    VisionFeatureLayout,
    detect_feature_layout,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image-resize helpers (shared across adapter implementations)
# ---------------------------------------------------------------------------

IMAGE_FACTOR = 28
MAX_RATIO = 200


def round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    """Resize (height, width) so both dims are divisible by *factor* and
    total pixels stay within [min_pixels, max_pixels]."""
    from sglang.srt.stream_engine.config import Config
    if min_pixels is None:
        min_pixels = Config.min_pixels
    if max_pixels is None:
        max_pixels = Config.max_pixels

    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def resize_frame(frame: np.ndarray, factor: int = IMAGE_FACTOR) -> np.ndarray:
    from sglang.srt.stream_engine.config import Config

    if not Config.enable_frame_resize:
        return frame

    h, w = frame.shape[:2]
    new_h, new_w = smart_resize(h, w, factor=factor)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# Model family enum and detection
# ---------------------------------------------------------------------------

class ModelFamily(str, Enum):
    QWEN_VL = "qwen_vl"
    LLAVA_ONEVISION = "llava_onevision"
    UNKNOWN = "unknown"


def detect_model_family(model_path: str) -> ModelFamily:
    """Determine the model family from a model path string."""
    path_lower = model_path.lower()
    if any(k in path_lower for k in ("llava-onevision", "llava_onevision")):
        return ModelFamily.LLAVA_ONEVISION
    if any(k in path_lower for k in (
        "qwen3-vl", "qwen3_vl",
        "qwen2.5-vl", "qwen2.5_vl",
        "qwen2-vl", "qwen2_vl",
    )):
        return ModelFamily.QWEN_VL
    return ModelFamily.UNKNOWN


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------

class BaseStreamModelAdapter(ABC):
    """Abstract adapter that owns all model-family-specific streaming behavior.

    ``StreamEngine`` delegates the following responsibilities to the adapter:

    * Chat-template resolution
    * Local vision model / processor loading (encoding fallback)
    * Frame-level visual embedding
    * Query-text embedding for retrieval
    * Retrieval hidden-size lookup
    * Generation-payload construction (prompt + ``engine.generate`` kwargs)

    The adapter holds the :class:`GlobalEncodingManager` after initialization
    so that ``StreamEngine`` can call ``adapter.encoding_manager`` for cluster
    encoding and token statistics.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.encoding_manager: Optional[GlobalEncodingManager] = None
        self.feature_layout: VisionFeatureLayout = detect_feature_layout(
            model_path=model_path
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def family(self) -> ModelFamily:
        """Return the ModelFamily this adapter handles."""
        ...

    # ------------------------------------------------------------------
    # Chat-template resolution
    # ------------------------------------------------------------------

    @abstractmethod
    def resolve_chat_template(self) -> str:
        """Return the sglang chat-template name for this model family."""
        ...

    # ------------------------------------------------------------------
    # Encoding-manager initialization
    # ------------------------------------------------------------------

    @abstractmethod
    def try_init_from_scheduler_model(
        self,
        model: nn.Module,
        hf_config: Any,
    ) -> bool:
        """Attempt to wrap the scheduler-side model for in-process encoding.

        Called when the sglang ``Engine`` exposes its internal model object.
        Returns ``True`` on success, ``False`` if the model object is not
        compatible with this adapter (e.g. missing expected attributes).
        """
        ...

    @abstractmethod
    def init_local_encoding(self) -> None:
        """Load a local HuggingFace model as the encoding fallback.

        Populates ``self.encoding_manager``, ``self.feature_layout``, and
        any internal processor / text-embedding references needed by
        ``embed_frame`` and ``get_query_embedding``.
        """
        ...

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def get_retrieval_hidden_size(self) -> int:
        """Return the hidden dimension *D* used for retrieval embeddings."""
        ...

    @abstractmethod
    def embed_frame(
        self,
        frames: np.ndarray,
        hot_token_mask: Optional[Any] = None,
        mask_mode: str = "off",
        cache_grid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one or more frames into visual embeddings.

        Parameters
        ----------
        frames:
            BGR frame array, shape ``(H, W, 3)`` or ``(N, H, W, 3)``.
        hot_token_mask:
            Optional boolean mask (for approximate sparse ViT encoding).
        mask_mode:
            ``"off"`` or ``"approx"``.
        cache_grid:
            When supported by the model adapter, return embeddings on the
            fixed spatial token grid used by cluster-token cache.

        Returns
        -------
        llm_embedding : Tensor, shape ``(num_tokens, D_llm)``
        grid_thw     : Tensor, shape ``(1, 3)`` or ``(N, 3)``
        retrieval_embedding : Tensor, shape ``(num_tokens, D_ret)``
        """
        ...

    @abstractmethod
    def get_query_embedding(self, query_text: str) -> torch.Tensor:
        """Return token-level text embeddings for retrieval, shape ``(S, D)``."""
        ...

    # ------------------------------------------------------------------
    # Generation-payload construction
    # ------------------------------------------------------------------

    @abstractmethod
    def build_generation_payload(
        self,
        all_embeddings: List[torch.Tensor],
        all_grid_thws: List[torch.Tensor],
        query: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Build the prompt string and ``engine.generate`` keyword arguments.

        Parameters
        ----------
        all_embeddings:
            List of per-cluster / per-frame visual embedding tensors.
        all_grid_thws:
            Matching list of ``(3,)`` grid tensors ``[T, H, W]``.
        query:
            The raw query string.

        Returns
        -------
        prompt : str
            Fully formatted prompt ready for ``engine.generate``.
        generate_kwargs : dict
            Additional kwargs merged into the ``engine.generate`` call
            (e.g. ``{"video_data": [...]}`` for Qwen-style generation).
        """
        ...


# ---------------------------------------------------------------------------
# QwenVLAdapter – Qwen2-VL / Qwen2.5-VL / Qwen3-VL
# ---------------------------------------------------------------------------

class QwenVLAdapter(BaseStreamModelAdapter):
    """Adapter for the Qwen-VL model family.

    Encapsulates all Qwen-specific assumptions:
    * ``model.visual`` for vision encoding
    * ``image_grid_thw`` / ``video_grid_thw`` with mRoPE temporal merging
    * ``<|vision_start|><|video_pad|>…<|vision_end|>`` prompt placeholders
    * ``video_data`` with ``format: "precomputed_embedding"`` for generation
    * ``qwen2-vl`` chat template
    """

    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)
        self._local_processor = None
        self._local_vision_model = None
        self._local_text_embeddings: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def family(self) -> ModelFamily:
        return ModelFamily.QWEN_VL

    # ------------------------------------------------------------------
    # Chat-template resolution
    # ------------------------------------------------------------------

    def resolve_chat_template(self) -> str:
        # Qwen2-VL, Qwen2.5-VL, and Qwen3-VL all share the qwen2-vl template.
        return "qwen2-vl"

    # ------------------------------------------------------------------
    # Encoding-manager initialization
    # ------------------------------------------------------------------

    def try_init_from_scheduler_model(
        self,
        model: nn.Module,
        hf_config: Any,
    ) -> bool:
        """Wrap the scheduler-side ``model.visual`` with GlobalEncodingManager."""
        if not hasattr(model, "visual"):
            logger.info(
                "QwenVLAdapter.try_init_from_scheduler_model: model has no "
                "'visual' attribute; skipping."
            )
            return False

        self.encoding_manager = GlobalEncodingManager(
            model=model,
            model_path=self.model_path,
            hf_config=hf_config,
        )
        self.encoding_manager.register_single_frame_encoder(self.embed_frame)
        self.feature_layout = self.encoding_manager.feature_layout
        logger.info(
            "QwenVLAdapter: scheduler-side GlobalEncodingManager active. "
            "layout=%s", self.feature_layout.value,
        )
        return True

    def init_local_encoding(self) -> None:
        """Load a local Qwen model and keep only vision tower + input embeddings."""
        from transformers import AutoModelForVision2Seq, AutoProcessor

        logger.info(
            "QwenVLAdapter: loading local vision model fallback. model_path=%s",
            self.model_path,
        )
        self._local_processor = AutoProcessor.from_pretrained(self.model_path)
        self._local_vision_model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep only the modules needed on-device for stream retrieval.
        self._local_vision_model.visual = self._local_vision_model.visual.to(
            device=target_device, dtype=torch.float16,
        )
        self._local_text_embeddings = self._local_vision_model.get_input_embeddings()
        self._local_text_embeddings = self._local_text_embeddings.to(device=target_device)

        for param in self._local_vision_model.visual.parameters():
            param.requires_grad_(False)
        for param in self._local_text_embeddings.parameters():
            param.requires_grad_(False)

        self._local_vision_model.eval()
        self._local_text_embeddings.eval()

        hf_config = getattr(self._local_vision_model, "config", None)
        self.encoding_manager = GlobalEncodingManager(
            model=self._local_vision_model,
            model_path=self.model_path,
            hf_config=hf_config,
        )
        self.encoding_manager.register_single_frame_encoder(self.embed_frame)
        self.feature_layout = self.encoding_manager.feature_layout
        logger.info(
            "QwenVLAdapter: local encoding ready. layout=%s device=%s",
            self.feature_layout.value, target_device,
        )

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_retrieval_hidden_size(self) -> int:
        if self._local_text_embeddings is not None:
            emb_weight = getattr(self._local_text_embeddings, "weight", None)
            if emb_weight is not None and emb_weight.dim() == 2:
                return int(emb_weight.shape[1])
        raise RuntimeError(
            "QwenVLAdapter: local text embedding layer is unavailable for "
            "retrieval projection."
        )

    def embed_frame(
        self,
        frames: np.ndarray,
        hot_token_mask: Optional[Any] = None,
        mask_mode: str = "off",
        cache_grid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode frames using the Qwen visual tower (via GlobalEncodingManager)."""
        if self.encoding_manager is None or self._local_processor is None:
            raise NotImplementedError(
                "QwenVLAdapter: encoding manager is not initialized; "
                "cannot encode frames."
            )

        if frames.ndim == 4:
            image_list = [
                cv2.cvtColor(resize_frame(f), cv2.COLOR_BGR2RGB) for f in frames
            ]
        else:
            image_list = [cv2.cvtColor(resize_frame(frames), cv2.COLOR_BGR2RGB)]

        inputs = self._local_processor.image_processor(
            images=image_list, return_tensors="pt"
        )

        with torch.no_grad():
            visual = self.encoding_manager.visual
            pixel_values = inputs["pixel_values"].to(device=visual.device, dtype=visual.dtype)
            image_grid_thw = inputs["image_grid_thw"].to(device=visual.device)

            if (
                hot_token_mask is not None
                and str(mask_mode).lower() == "approx"
            ):
                pixel_values, image_grid_thw = self._build_mask_approx_inputs(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    hot_token_mask=hot_token_mask,
                )

            visual_output = self.encoding_manager.encode(pixel_values, grid_thw=image_grid_thw)

        main_feature, deepstack_features = self._normalize_visual_output(visual_output)
        llm_embedding = (
            torch.cat([main_feature] + deepstack_features, dim=-1)
            if deepstack_features
            else main_feature
        )

        retrieval_source = deepstack_features[-1] if deepstack_features else main_feature
        retrieval_dim = self.get_retrieval_hidden_size()
        retrieval_embedding = self._project_feature_to_dim(retrieval_source, retrieval_dim)

        return llm_embedding, image_grid_thw, retrieval_embedding

    def get_query_embedding(self, query_text: str) -> torch.Tensor:
        """Tokenize *query_text* and return token-level embeddings, shape (S, D)."""
        if self._local_processor is None or self._local_text_embeddings is None:
            raise RuntimeError(
                "QwenVLAdapter: local vision model is not initialized; "
                "cannot compute query embeddings."
            )
        inputs = self._local_processor(text=query_text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        emb_weight = getattr(self._local_text_embeddings, "weight", None)
        embedding_device = emb_weight.device if emb_weight is not None else torch.device("cpu")
        input_ids = input_ids.to(embedding_device)

        with torch.no_grad():
            query_embeddings = self._local_text_embeddings(input_ids)  # (1, S, D)

        return query_embeddings.squeeze(0)  # (S, D)

    # ------------------------------------------------------------------
    # Generation-payload construction
    # ------------------------------------------------------------------

    def build_generation_payload(
        self,
        all_embeddings: List[torch.Tensor],
        all_grid_thws: List[torch.Tensor],
        query: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Concatenate embeddings and build a Qwen-style video generation payload.

        Merges per-frame ``[1, H, W]`` grid entries into video segments with
        matching spatial resolution so mRoPE receives correct temporal indices.
        Wraps the result in ``video_data`` with ``format: "precomputed_embedding"``.
        """
        # Align embedding dims (FLAT vs DEEPSTACK abstract embeddings may differ).
        target_embed_dim = max(int(emb.shape[-1]) for emb in all_embeddings)
        aligned = [self._align_embedding_dim(emb, target_embed_dim) for emb in all_embeddings]
        concat_embeddings = torch.cat(aligned, dim=0)

        # Merge consecutive same-resolution frames into temporal segments.
        video_grid_thw: Optional[torch.Tensor] = None
        if all_grid_thws:
            merged_segments: List[List[int]] = []
            for grid in all_grid_thws:
                h_val = int(grid[-2])
                w_val = int(grid[-1])
                if (
                    merged_segments
                    and merged_segments[-1][1] == h_val
                    and merged_segments[-1][2] == w_val
                ):
                    merged_segments[-1][0] += 1
                else:
                    merged_segments.append([1, h_val, w_val])
            video_grid_thw = torch.tensor(
                merged_segments, dtype=torch.long, device=concat_embeddings.device,
            )

        # Build prompt: single contiguous video-pad block + query.
        num_video_tokens = int(concat_embeddings.shape[0])
        video_placeholder = (
            "<|vision_start|>"
            + "<|video_pad|>" * num_video_tokens
            + "<|vision_end|>"
        )
        prompt_body = video_placeholder + query

        tokenizer = getattr(self._local_processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_body}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = prompt_body

        # Assemble video_data payload.
        video_data_payload: Dict[str, Any] = {
            "format": "precomputed_embedding",
            "feature": concat_embeddings,
        }
        if video_grid_thw is not None:
            video_data_payload["video_grid_thw"] = video_grid_thw
            num_segments = int(video_grid_thw.shape[0])
            video_data_payload["second_per_grid_ts"] = torch.ones(
                num_segments, dtype=torch.float32,
            )

        generate_kwargs: Dict[str, Any] = {"video_data": [video_data_payload]}
        return prompt, generate_kwargs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_visual_output(
        visual_output: Any,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return ``(main_feature, deepstack_features)`` from a visual encoder output."""
        if isinstance(visual_output, torch.Tensor):
            return visual_output, []
        if (
            isinstance(visual_output, tuple)
            and len(visual_output) == 2
            and isinstance(visual_output[0], torch.Tensor)
            and isinstance(visual_output[1], (list, tuple))
        ):
            deepstack_features = [
                feat for feat in visual_output[1] if isinstance(feat, torch.Tensor)
            ]
            return visual_output[0], deepstack_features
        raise TypeError(
            "QwenVLAdapter: unsupported visual encoder output type. "
            f"Expected Tensor or Tuple[Tensor, List[Tensor]], got: {type(visual_output)}"
        )

    @staticmethod
    def _project_feature_to_dim(feature: torch.Tensor, target_dim: int) -> torch.Tensor:
        current_dim = int(feature.shape[-1])
        if current_dim == target_dim:
            return feature
        if current_dim > target_dim:
            if current_dim % target_dim == 0:
                group = current_dim // target_dim
                return feature.reshape(*feature.shape[:-1], target_dim, group).mean(dim=-1)
            if feature.dim() == 2:
                return (
                    F.adaptive_avg_pool1d(
                        feature.unsqueeze(0).transpose(1, 2), target_dim
                    ).transpose(1, 2).squeeze(0)
                )
            return F.adaptive_avg_pool1d(feature.transpose(-2, -1), target_dim).transpose(-2, -1)
        return F.pad(feature, (0, target_dim - current_dim))

    @staticmethod
    def _align_embedding_dim(embedding: torch.Tensor, target_dim: int) -> torch.Tensor:
        cur_dim = int(embedding.shape[-1])
        if cur_dim == target_dim:
            return embedding
        if cur_dim < target_dim:
            return F.pad(embedding, (0, target_dim - cur_dim))
        return embedding[..., :target_dim]

    def _get_spatial_merge_size(self) -> int:
        if self.encoding_manager is not None:
            visual = self.encoding_manager.visual
            merge_size = getattr(visual, "spatial_merge_size", None)
            if isinstance(merge_size, int) and merge_size > 0:
                return merge_size
        vision_cfg = getattr(
            getattr(self._local_vision_model, "config", None), "vision_config", None
        )
        merge_size = getattr(vision_cfg, "spatial_merge_size", None)
        if isinstance(merge_size, int) and merge_size > 0:
            return merge_size
        return 2

    def _build_mask_approx_inputs(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        hot_token_mask: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack hot-token patches into a compact pseudo-grid for approximate sparse ViT."""
        if image_grid_thw.dim() != 2 or image_grid_thw.shape[0] <= 0:
            return pixel_values, image_grid_thw

        spatial_merge_size = self._get_spatial_merge_size()
        group_size = int(spatial_merge_size * spatial_merge_size)
        if group_size <= 0:
            return pixel_values, image_grid_thw

        if isinstance(hot_token_mask, torch.Tensor):
            hot_mask = hot_token_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
        else:
            hot_mask = torch.from_numpy(np.asarray(hot_token_mask).astype(np.bool_)).reshape(-1)

        hot_token_indices = torch.nonzero(hot_mask, as_tuple=False).reshape(-1)
        if int(hot_token_indices.numel()) <= 0:
            return pixel_values, image_grid_thw
        hot_token_indices = hot_token_indices.to(device=pixel_values.device, dtype=torch.long)

        hot_token_count = int(hot_token_indices.numel())
        patch_dim = int(pixel_values.shape[-1])
        total_patch_count = int(pixel_values.shape[0])
        patch_cursor = 0
        packed_pixel_chunks: List[torch.Tensor] = []
        packed_grid_rows: List[List[int]] = []

        for row in image_grid_thw:
            row_patch_count = int(torch.prod(row.to(dtype=torch.long)).item())
            if row_patch_count <= 0 or row_patch_count % group_size != 0:
                return pixel_values, image_grid_thw

            full_token_count = row_patch_count // group_size
            if int(hot_mask.numel()) != full_token_count:
                return pixel_values, image_grid_thw

            row_end = patch_cursor + row_patch_count
            if row_end > total_patch_count:
                return pixel_values, image_grid_thw

            grouped_pixels = pixel_values[patch_cursor:row_end].reshape(
                full_token_count,
                group_size,
                patch_dim,
            )
            packed_pixel_chunks.append(
                grouped_pixels[hot_token_indices].reshape(-1, patch_dim).contiguous()
            )
            packed_grid_rows.append([1, hot_token_count * spatial_merge_size, spatial_merge_size])
            patch_cursor = row_end

        if patch_cursor != total_patch_count or not packed_pixel_chunks:
            return pixel_values, image_grid_thw

        packed_hot_pixels = torch.cat(packed_pixel_chunks, dim=0)
        packed_grid = torch.tensor(
            packed_grid_rows,
            dtype=image_grid_thw.dtype,
            device=image_grid_thw.device,
        )
        return packed_hot_pixels, packed_grid


# LlavaOneVisionAdapter – llava-hf/llava-onevision-qwen2-*
# ---------------------------------------------------------------------------

class LlavaOneVisionAdapter(BaseStreamModelAdapter):
    """Adapter for LLaVA-OneVision models.

    Supports:

    * ``llava-hf/llava-onevision-qwen2-7b-ov-hf``
    * ``llava-hf/llava-onevision-qwen2-0.5b-ov-hf``

    Both checkpoints share the same code path; they differ only in the
    number of parameters loaded from *model_path*.

    Design notes
    ------------
    * LLaVA-OneVision does **not** expose ``model.visual``; instead it uses
      ``get_image_features()`` (vision tower + multi-modal projector).
    * Frame-level encoding is performed per image rather than via a
      contiguous-video path, matching the stream engine's cluster-based model.
    * Retrieved visual context is packaged as an ordered multi-image sequence
      using ``image_data`` with ``format: "precomputed_embedding"``.
    * The chat template is ``chatml-llava`` (used by sglang for the
      ``llava-onevision-qwen2`` model family).
    """

    MAX_TOKENS_PER_IMAGE = 196

    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)
        self._local_processor = None
        self._local_model = None
        self._scheduler_model: Optional[nn.Module] = None
        self._local_text_embeddings: Optional[nn.Module] = None
        self._hidden_size: Optional[int] = None
        self._target_device: Optional[torch.device] = None
        # Override default feature layout; LLaVA has no deepstack.
        from sglang.srt.stream_engine.encoding_mgr import VisionFeatureLayout
        self.feature_layout = VisionFeatureLayout.FLAT

    def _ensure_processor(self):
        if self._local_processor is None:
            from transformers import AutoProcessor
            local_hf_path = self._resolve_local_hf_path()

            logger.info(
                "LlavaOneVisionAdapter: loading processor for tokenizer/image preprocessing. "
                "model_path=%s local_hf_path=%s",
                self.model_path,
                local_hf_path,
            )
            self._local_processor = AutoProcessor.from_pretrained(
                local_hf_path,
                use_fast=False,
            )
        return self._local_processor

    def _resolve_local_hf_path(self) -> str:
        path_lower = self.model_path.lower()
        legacy_aliases = {
            "lmms-lab/llava-onevision-qwen2-7b-ov":
                "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            "lmms-lab/llava-onevision-qwen2-0.5b-ov":
                "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        }
        resolved_path = legacy_aliases.get(path_lower, self.model_path)
        if resolved_path != self.model_path:
            logger.info(
                "LlavaOneVisionAdapter: using HF-converted checkpoint for local "
                "encoding fallback. model_path=%s local_hf_path=%s",
                self.model_path,
                resolved_path,
            )
        return resolved_path

    def _get_active_model(self) -> Optional[nn.Module]:
        if self._scheduler_model is not None:
            return self._scheduler_model
        return self._local_model

    def _resolve_target_device(self, model: Optional[nn.Module]) -> torch.device:
        if self._target_device is not None:
            return self._target_device
        if model is not None:
            candidate_roots = [model]
            inner_model = getattr(model, "model", None)
            if isinstance(inner_model, nn.Module):
                candidate_roots.append(inner_model)
            for root in candidate_roots:
                for module_name in ("vision_tower", "multi_modal_projector", "language_model"):
                    module = getattr(root, module_name, None)
                    if module is None:
                        continue
                    try:
                        self._target_device = next(iter(module.parameters())).device
                        return self._target_device
                    except StopIteration:
                        continue
            try:
                self._target_device = next(iter(model.parameters())).device
                return self._target_device
            except StopIteration:
                pass
        self._target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._target_device

    def _ensure_query_encoder_ready(self):
        processor = self._ensure_processor()
        if self._local_text_embeddings is None and self._local_model is not None:
            target_device = self._resolve_target_device(self._local_model)
            self._local_text_embeddings = self._local_model.get_input_embeddings().to(
                device=target_device
            )
        if self._local_text_embeddings is None:
            raise RuntimeError(
                "LlavaOneVisionAdapter: text embedding layer is unavailable; "
                "cannot compute query embeddings."
            )
        return processor

    def _prepare_resized_rgb_frames(self, frames: np.ndarray) -> List[np.ndarray]:
        """Apply the shared stream-engine resize policy before LLaVA preprocessing."""
        if frames.ndim == 3:
            frame_list = [frames]
        else:
            frame_list = list(frames)
        return [
            cv2.cvtColor(resize_frame(frame), cv2.COLOR_BGR2RGB)
            for frame in frame_list
        ]

    def _build_generation_compatible_image_inputs(
        self,
        pil_images: List[Any],
    ) -> Dict[str, torch.Tensor]:
        """Preprocess images with the same multi-image semantics used at decode time.

        Stream-engine generation always serializes retrieved LLaVA context as an
        ordered multi-image prompt. If we preprocess each frame as a standalone
        single-image request, HF falls back to anyres patching and produces a
        different embedding layout from the decoder's expected multi-image path.
        """
        processor = self._ensure_processor()
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError(
                "LlavaOneVisionAdapter: processor.image_processor is unavailable; "
                "cannot encode frames."
            )

        if len(pil_images) > 1:
            inputs = image_processor(
                images=[pil_images],
                return_tensors="pt",
            )
            # Match HF LLaVA-OneVision's multi-image path: when a sample has
            # multiple images, get_image_features disables anyres patching and
            # keeps one spatial grid per image.
            inputs["batch_num_images"] = torch.tensor(
                [len(pil_images)], dtype=torch.long
            )
            return inputs

        # HF's public image processor only disables anyres patching when the
        # request is represented as a multi-image sample. The stream engine
        # later feeds even a single retrieved frame through the same
        # precomputed multi-image transport, so we force that layout here too.
        duplicated_inputs = image_processor(
            images=[[pil_images[0], pil_images[0]]],
            return_tensors="pt",
        )
        return {
            "pixel_values": duplicated_inputs["pixel_values"][:1],
            "image_sizes": duplicated_inputs["image_sizes"][:1],
            "batch_num_images": torch.tensor([2], dtype=torch.long),
        }

    def _normalize_image_feature_batches(self, image_features: Any) -> List[torch.Tensor]:
        """Preserve per-image boundaries for downstream precomputed-image packing."""
        if isinstance(image_features, torch.Tensor):
            if image_features.dim() == 2:
                return [image_features]
            if image_features.dim() == 3:
                return [feature.reshape(-1, feature.shape[-1]) for feature in image_features]
            raise RuntimeError(
                "LlavaOneVisionAdapter: unsupported tensor image feature shape "
                f"{tuple(image_features.shape)}."
            )

        if not isinstance(image_features, (list, tuple)):
            raise RuntimeError(
                "LlavaOneVisionAdapter: unsupported image feature output type "
                f"{type(image_features)!r}."
            )

        normalized_features: List[torch.Tensor] = []
        for feature in image_features:
            if not isinstance(feature, torch.Tensor):
                feature = torch.as_tensor(feature)
            if feature.dim() == 1:
                feature = feature.unsqueeze(0)
            elif feature.dim() > 2:
                feature = feature.reshape(-1, feature.shape[-1])
            normalized_features.append(feature)

        if not normalized_features:
            raise RuntimeError(
                "LlavaOneVisionAdapter: image feature extraction returned no features."
            )
        return normalized_features

    def _cap_image_feature_tokens(
        self,
        image_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Bound each LLaVA frame embedding to a fixed token budget."""
        if image_feature.dim() != 2:
            image_feature = image_feature.reshape(-1, image_feature.shape[-1])

        max_tokens = int(self.MAX_TOKENS_PER_IMAGE)
        current_tokens = int(image_feature.shape[0])
        if current_tokens <= max_tokens:
            return image_feature

        pooled_feature = F.adaptive_avg_pool1d(
            image_feature.transpose(0, 1).unsqueeze(0),
            output_size=max_tokens,
        ).squeeze(0).transpose(0, 1).contiguous()
        logger.info(
            "LlavaOneVisionAdapter: capped per-frame visual tokens from %d to %d.",
            current_tokens,
            max_tokens,
        )
        return pooled_feature

    @staticmethod
    def _infer_square_grid(token_count: int) -> Optional[Tuple[int, int]]:
        side = int(math.isqrt(max(int(token_count), 0)))
        if side > 0 and side * side == int(token_count):
            return side, side
        return None

    @staticmethod
    def _build_grid_groups(
        raw_hw: Tuple[int, int],
        target_hw: Tuple[int, int],
    ) -> List[torch.Tensor]:
        raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        groups: List[torch.Tensor] = []
        for out_r in range(target_h):
            row_start = int(math.floor(out_r * raw_h / target_h))
            row_end = int(math.floor((out_r + 1) * raw_h / target_h))
            row_end = max(row_end, row_start + 1)
            for out_c in range(target_w):
                col_start = int(math.floor(out_c * raw_w / target_w))
                col_end = int(math.floor((out_c + 1) * raw_w / target_w))
                col_end = max(col_end, col_start + 1)
                indices = [
                    row * raw_w + col
                    for row in range(row_start, min(row_end, raw_h))
                    for col in range(col_start, min(col_end, raw_w))
                ]
                groups.append(torch.tensor(indices, dtype=torch.long))
        return groups

    def _resolve_llava_inner_model(self) -> Optional[nn.Module]:
        active_model = self._get_active_model()
        if active_model is None:
            return None
        inner_model = getattr(active_model, "model", None)
        return inner_model if isinstance(inner_model, nn.Module) else active_model

    def _should_use_spatial_cache_grid(self) -> bool:
        from sglang.srt.stream_engine.config import Config

        return bool(
            getattr(Config, "llava_ov_spatial_cache_grid", True)
            and getattr(Config, "enable_embedding_cache", True)
        )

    def _resolve_llava_vision_settings(self, inner_model: nn.Module) -> Tuple[Any, str]:
        config = getattr(inner_model, "config", None)
        vision_feature_layer = getattr(inner_model, "vision_feature_layer", None)
        if vision_feature_layer is None and config is not None:
            vision_feature_layer = getattr(config, "vision_feature_layer", None)
        if vision_feature_layer is None:
            vision_feature_layer = -2

        select_strategy = getattr(inner_model, "vision_feature_select_strategy", None)
        if select_strategy is None and config is not None:
            select_strategy = getattr(config, "vision_feature_select_strategy", None)
        if select_strategy is None:
            select_strategy = "full"
        return vision_feature_layer, str(select_strategy)

    @staticmethod
    def _normalize_hidden_state_index(index: int, total_states: int) -> Optional[int]:
        normalized = int(index)
        if normalized < 0:
            normalized = total_states + normalized
        if normalized < 0 or normalized >= total_states:
            return None
        return normalized

    def _run_siglip_encoder_for_layers(
        self,
        vision_model: nn.Module,
        hidden_states: torch.Tensor,
        vision_feature_layer: Any,
    ) -> Optional[torch.Tensor]:
        encoder = getattr(vision_model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is None:
            return None

        layer_list = list(layers)
        total_states = len(layer_list) + 1
        requested_layers = (
            list(vision_feature_layer)
            if isinstance(vision_feature_layer, (list, tuple))
            else [vision_feature_layer]
        )
        target_indices: List[int] = []
        for layer_idx in requested_layers:
            normalized = self._normalize_hidden_state_index(int(layer_idx), total_states)
            if normalized is None:
                return None
            target_indices.append(normalized)

        max_target = max(target_indices)
        captured: Dict[int, torch.Tensor] = {}
        if 0 in target_indices:
            captured[0] = hidden_states

        for state_idx, encoder_layer in enumerate(layer_list, start=1):
            if state_idx > max_target:
                break
            hidden_states = encoder_layer(hidden_states, None)
            if state_idx in target_indices:
                captured[state_idx] = hidden_states

        if any(idx not in captured for idx in target_indices):
            return None
        selected = [captured[idx] for idx in target_indices]
        return selected[0] if len(selected) == 1 else torch.cat(selected, dim=-1)

    @staticmethod
    def _select_llava_vision_tokens(
        selected_feature: torch.Tensor,
        select_strategy: str,
        expected_patch_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        if select_strategy == "default":
            if expected_patch_tokens is not None:
                token_count = int(selected_feature.shape[1])
                if token_count == int(expected_patch_tokens) + 1:
                    return selected_feature[:, 1:]
                if token_count == int(expected_patch_tokens):
                    return selected_feature
            return selected_feature[:, 1:]
        return selected_feature

    def _pool_projected_spatial_tokens(
        self,
        projected_features: torch.Tensor,
        raw_hw: Tuple[int, int],
        target_hw: Tuple[int, int],
        hot_token_mask: Optional[Any] = None,
    ) -> Optional[List[torch.Tensor]]:
        groups = self._build_grid_groups(raw_hw=raw_hw, target_hw=target_hw)
        if not groups:
            return None

        if hot_token_mask is not None:
            hot_mask = self._normalize_hot_token_mask(hot_token_mask)
            if hot_mask is None or int(hot_mask.numel()) != len(groups):
                return None
            selected_group_indices = torch.nonzero(hot_mask, as_tuple=False).reshape(-1)
            if int(selected_group_indices.numel()) <= 0:
                return None
        else:
            selected_group_indices = torch.arange(len(groups), dtype=torch.long)

        pooled_per_image: List[torch.Tensor] = []
        for image_idx in range(int(projected_features.shape[0])):
            pooled_tokens: List[torch.Tensor] = []
            for group_idx in selected_group_indices.tolist():
                group = groups[int(group_idx)].to(device=projected_features.device)
                if int(group.numel()) <= 0:
                    return None
                pooled_tokens.append(projected_features[image_idx, group].mean(dim=0))
            if not pooled_tokens:
                return None
            pooled_per_image.append(torch.stack(pooled_tokens, dim=0).contiguous())
        return pooled_per_image

    @staticmethod
    def _pair(value: Any) -> Tuple[int, int]:
        if isinstance(value, str):
            normalized = value.lower()
            if normalized == "valid":
                return 0, 0
            # Unsupported symbolic padding should force the sparse helper to
            # fall back instead of crashing or computing incorrect patch coords.
            return -1, -1
        if isinstance(value, (tuple, list)):
            return int(value[0]), int(value[1])
        return int(value), int(value)

    def _infer_siglip_patch_grid(
        self,
        embeddings_module: nn.Module,
        pixel_values: torch.Tensor,
    ) -> Optional[Tuple[int, int]]:
        patch_embedding = getattr(embeddings_module, "patch_embedding", None)
        if not isinstance(patch_embedding, nn.Conv2d):
            return None
        _, _, image_h, image_w = pixel_values.shape
        kernel_h, kernel_w = self._pair(patch_embedding.kernel_size)
        stride_h, stride_w = self._pair(patch_embedding.stride)
        pad_h, pad_w = self._pair(patch_embedding.padding)
        dil_h, dil_w = self._pair(patch_embedding.dilation)
        if pad_h < 0 or pad_w < 0:
            return None
        grid_h = ((int(image_h) + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h) + 1
        grid_w = ((int(image_w) + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w) + 1
        if grid_h <= 0 or grid_w <= 0:
            return None
        return int(grid_h), int(grid_w)

    def _build_sparse_siglip_patch_embeddings(
        self,
        embeddings_module: nn.Module,
        pixel_values: torch.Tensor,
        raw_indices: torch.Tensor,
        raw_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        patch_embedding = getattr(embeddings_module, "patch_embedding", None)
        position_embedding = getattr(embeddings_module, "position_embedding", None)
        position_ids = getattr(embeddings_module, "position_ids", None)
        if (
            not isinstance(patch_embedding, nn.Conv2d)
            or position_embedding is None
            or position_ids is None
        ):
            return None

        kernel_h, kernel_w = self._pair(patch_embedding.kernel_size)
        stride_h, stride_w = self._pair(patch_embedding.stride)
        pad_h, pad_w = self._pair(patch_embedding.padding)
        dil_h, dil_w = self._pair(patch_embedding.dilation)
        if pad_h < 0 or pad_w < 0:
            return None
        if pad_h != 0 or pad_w != 0 or dil_h != 1 or dil_w != 1:
            return None

        raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
        flat_indices = raw_indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if int(flat_indices.numel()) <= 0:
            return None

        patches: List[torch.Tensor] = []
        source_pixels = pixel_values.to(dtype=patch_embedding.weight.dtype)
        for flat_index in flat_indices.tolist():
            row = int(flat_index) // raw_w
            col = int(flat_index) % raw_w
            if row < 0 or row >= raw_h or col < 0 or col >= raw_w:
                return None
            top = row * stride_h
            left = col * stride_w
            patch = source_pixels[
                :,
                :,
                top : top + kernel_h,
                left : left + kernel_w,
            ]
            if int(patch.shape[-2]) != kernel_h or int(patch.shape[-1]) != kernel_w:
                return None
            patches.append(patch.reshape(int(source_pixels.shape[0]), -1))

        patch_matrix = torch.stack(patches, dim=1)
        weight = patch_embedding.weight.reshape(int(patch_embedding.out_channels), -1)
        sparse_embeddings = torch.matmul(patch_matrix, weight.t())
        if patch_embedding.bias is not None:
            sparse_embeddings = sparse_embeddings + patch_embedding.bias

        flat_position_ids = position_ids.reshape(-1)
        if int(flat_position_ids.numel()) < int(raw_h * raw_w):
            return None
        selected_position_ids = flat_position_ids[flat_indices].to(
            device=source_pixels.device,
            dtype=torch.long,
        )
        sparse_embeddings = sparse_embeddings + position_embedding(
            selected_position_ids
        ).unsqueeze(0).to(dtype=sparse_embeddings.dtype)
        return sparse_embeddings.contiguous()

    def _encode_spatial_image_features(
        self,
        pixel_values: torch.Tensor,
        *,
        hot_token_mask: Optional[Any] = None,
        sparse: bool = False,
    ) -> Optional[Tuple[List[torch.Tensor], torch.Tensor]]:
        """Encode LLaVA-OV features as a spatial 2D token grid.

        Dense mode returns a fixed 2D cache grid, e.g. 14x14. Sparse mode uses
        one shared hot-token mask for the whole image batch, extracts only the
        raw SigLIP patches behind those hot cells, adds their original absolute
        position embeddings, and runs the compact batch through the requested
        vision-tower layers. Cold final tokens are filled by the caller from
        the representative cache.
        """
        from sglang.srt.stream_engine.config import Config

        inner_model = self._resolve_llava_inner_model()
        if inner_model is None:
            return None
        if pixel_values.dim() == 5:
            if int(pixel_values.shape[1]) == 1:
                pixel_values = pixel_values[:, 0]
            elif int(pixel_values.shape[0]) == 1:
                pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:])
            else:
                return None
        if pixel_values.dim() != 4:
            return None

        vision_tower = getattr(inner_model, "vision_tower", None)
        projector = getattr(inner_model, "multi_modal_projector", None)
        vision_model = getattr(vision_tower, "vision_model", None)
        embeddings_module = getattr(vision_model, "embeddings", None)
        if (
            vision_model is None
            or embeddings_module is None
            or getattr(vision_model, "encoder", None) is None
            or projector is None
        ):
            return None

        if sparse:
            raw_hw = self._infer_siglip_patch_grid(
                embeddings_module=embeddings_module,
                pixel_values=pixel_values,
            )
            if raw_hw is None:
                return None
            raw_token_count = int(raw_hw[0] * raw_hw[1])
        else:
            dense_embeddings = embeddings_module(pixel_values)
            raw_hw = self._infer_square_grid(int(dense_embeddings.shape[1]))
            if raw_hw is None:
                return None
            raw_token_count = int(dense_embeddings.shape[1])

        target_token_count = int(getattr(Config, "llava_ov_cache_grid_tokens", self.MAX_TOKENS_PER_IMAGE))
        target_token_count = min(max(target_token_count, 1), raw_token_count)
        target_hw = self._infer_square_grid(target_token_count)
        if target_hw is None:
            return None

        hidden_states: torch.Tensor
        sparse_raw_indices: Optional[torch.Tensor] = None
        if sparse:
            hot_mask = self._normalize_hot_token_mask(hot_token_mask)
            if hot_mask is None or int(hot_mask.numel()) != int(target_hw[0] * target_hw[1]):
                return None
            hot_group_indices = torch.nonzero(hot_mask, as_tuple=False).reshape(-1)
            if int(hot_group_indices.numel()) <= 0:
                return None
            groups = self._build_grid_groups(raw_hw=raw_hw, target_hw=target_hw)
            raw_index_chunks = [groups[int(idx)] for idx in hot_group_indices.tolist()]
            sparse_raw_indices = torch.unique(torch.cat(raw_index_chunks, dim=0), sorted=True)
            if int(sparse_raw_indices.numel()) <= 0:
                return None
            hidden_states = self._build_sparse_siglip_patch_embeddings(
                embeddings_module=embeddings_module,
                pixel_values=pixel_values,
                raw_indices=sparse_raw_indices,
                raw_hw=raw_hw,
            )
            if hidden_states is None:
                return None
        else:
            hidden_states = dense_embeddings

        vision_feature_layer, select_strategy = self._resolve_llava_vision_settings(inner_model)
        selected_feature = self._run_siglip_encoder_for_layers(
            vision_model=vision_model,
            hidden_states=hidden_states,
            vision_feature_layer=vision_feature_layer,
        )
        if selected_feature is None:
            return None

        selected_feature = self._select_llava_vision_tokens(
            selected_feature=selected_feature,
            select_strategy=select_strategy,
            expected_patch_tokens=(
                int(sparse_raw_indices.numel())
                if sparse and sparse_raw_indices is not None
                else int(raw_hw[0] * raw_hw[1])
            ),
        )

        projected_features = projector(selected_feature)
        if sparse and sparse_raw_indices is not None:
            # Re-expand sparse raw-token outputs into the original raw grid
            # slots needed by group pooling. Missing cold raw tokens remain
            # absent and are never consumed because only hot groups are pooled.
            expanded = projected_features.new_zeros(
                (
                    int(projected_features.shape[0]),
                    int(raw_hw[0] * raw_hw[1]),
                    int(projected_features.shape[-1]),
                )
            )
            expanded[:, sparse_raw_indices.to(device=expanded.device)] = projected_features
            projected_features = expanded

        if int(projected_features.shape[1]) != int(raw_hw[0] * raw_hw[1]):
            return None

        per_image_embeddings = self._pool_projected_spatial_tokens(
            projected_features=projected_features,
            raw_hw=raw_hw,
            target_hw=target_hw,
            hot_token_mask=hot_token_mask if sparse else None,
        )
        if per_image_embeddings is None:
            return None

        if sparse:
            grid_rows = [
                [1, int(feature.shape[0]), 1] for feature in per_image_embeddings
            ]
        else:
            grid_rows = [[1, int(target_hw[0]), int(target_hw[1])] for _ in per_image_embeddings]
        grid_thw = torch.tensor(
            grid_rows,
            dtype=torch.long,
            device=per_image_embeddings[0].device,
        )
        return per_image_embeddings, grid_thw

    @staticmethod
    def _normalize_hot_token_mask(
        hot_token_mask: Optional[Any],
    ) -> Optional[torch.Tensor]:
        if hot_token_mask is None:
            return None
        if isinstance(hot_token_mask, torch.Tensor):
            return hot_token_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
        return torch.from_numpy(
            np.asarray(hot_token_mask).astype(np.bool_, copy=False)
        ).reshape(-1)

    def _apply_shared_hot_token_mask(
        self,
        per_image_embeddings: List[torch.Tensor],
        hot_token_mask: Optional[Any],
        mask_mode: str,
    ) -> Optional[List[torch.Tensor]]:
        if str(mask_mode).lower() != "approx":
            return None

        hot_mask = self._normalize_hot_token_mask(hot_token_mask)
        if hot_mask is None:
            return None

        hot_token_count = int(hot_mask.sum().item())
        if hot_token_count <= 0:
            return None

        sparse_embeddings: List[torch.Tensor] = []
        for image_feature in per_image_embeddings:
            token_count = int(image_feature.shape[0])
            if token_count != int(hot_mask.numel()):
                return None
            sparse_embeddings.append(
                image_feature[
                    hot_mask.to(device=image_feature.device)
                ].contiguous()
            )
        return sparse_embeddings

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def family(self) -> ModelFamily:
        return ModelFamily.LLAVA_ONEVISION

    # ------------------------------------------------------------------
    # Chat-template resolution
    # ------------------------------------------------------------------

    def resolve_chat_template(self) -> str:
        # sglang's conversation.py maps 'llava-onevision-qwen2' to 'chatml-llava'.
        return "chatml-llava"

    # ------------------------------------------------------------------
    # Encoding-manager initialization
    # ------------------------------------------------------------------

    def try_init_from_scheduler_model(
        self,
        model: nn.Module,
        hf_config: Any,
    ) -> bool:
        """Wrap the scheduler-side LLaVA model with GlobalEncodingManager."""
        if not any(
            callable(getattr(model, attr_name, None))
            for attr_name in ("get_image_feature", "get_image_features")
        ):
            logger.info(
                "LlavaOneVisionAdapter.try_init_from_scheduler_model: model lacks "
                "a supported LLaVA image-feature interface; skipping."
            )
            return False

        self._scheduler_model = model
        self.encoding_manager = GlobalEncodingManager(
            model=model,
            model_path=self.model_path,
            hf_config=hf_config,
        )
        self.encoding_manager.register_single_frame_encoder(self.embed_frame)
        self.feature_layout = self.encoding_manager.feature_layout

        # Extract hidden size from hf_config.
        text_config = getattr(hf_config, "text_config", None)
        self._hidden_size = getattr(text_config, "hidden_size", None)
        if self._hidden_size is None:
            self._hidden_size = getattr(hf_config, "hidden_size", None)

        # Grab text embedding layer from the language model sub-module.
        lang_model = getattr(model, "language_model", None)
        lang_inner = getattr(lang_model, "model", None) if lang_model else None
        embed_tokens = getattr(lang_inner, "embed_tokens", None) if lang_inner else None
        if embed_tokens is not None:
            self._local_text_embeddings = embed_tokens

        # Determine the device from the vision tower.
        vision_tower = getattr(model, "vision_tower", None)
        try:
            vt_param = next(iter(vision_tower.parameters())) if vision_tower else None
            device = vt_param.device if vt_param is not None else torch.device("cpu")
        except StopIteration:
            device = torch.device("cpu")
        self._target_device = device

        try:
            self._ensure_processor()
        except Exception as exc:
            logger.warning(
                "LlavaOneVisionAdapter: failed to preload processor during "
                "scheduler-side init; query tokenization will retry lazily. %s",
                exc,
            )

        logger.info(
            "LlavaOneVisionAdapter: scheduler-side init successful. "
            "hidden_size=%s device=%s layout=%s",
            self._hidden_size,
            device,
            self.feature_layout.value,
        )
        return True

    def init_local_encoding(self) -> None:
        """Load ``LlavaOnevisionForConditionalGeneration`` as encoding fallback.

        Only vision tower, multi-modal projector, and input embeddings are
        moved to the target device; the LLM decoder remains on CPU to reduce
        memory pressure.
        """
        from transformers import LlavaOnevisionForConditionalGeneration

        local_hf_path = self._resolve_local_hf_path()
        logger.info(
            "LlavaOneVisionAdapter: loading local LLaVA-OneVision model. "
            "model_path=%s local_hf_path=%s",
            self.model_path,
            local_hf_path,
        )
        self._local_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            local_hf_path,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._target_device = target_device

        # Move only the vision-side modules needed by HF get_image_features.
        vision_roots = [self._local_model]
        inner_model = getattr(self._local_model, "model", None)
        if isinstance(inner_model, nn.Module):
            vision_roots.append(inner_model)

        for root in vision_roots:
            vision_tower = getattr(root, "vision_tower", None)
            if vision_tower is not None:
                setattr(
                    root,
                    "vision_tower",
                    vision_tower.to(device=target_device, dtype=torch.float16),
                )
            projector = getattr(root, "multi_modal_projector", None)
            if projector is not None:
                setattr(
                    root,
                    "multi_modal_projector",
                    projector.to(device=target_device, dtype=torch.float16),
                )
            image_newline = getattr(root, "image_newline", None)
            if isinstance(image_newline, torch.Tensor):
                root.image_newline = nn.Parameter(
                    image_newline.to(device=target_device, dtype=torch.float16),
                    requires_grad=False,
                )

        # Freeze all parameters; we only need inference.
        for param in self._local_model.parameters():
            param.requires_grad_(False)
        self._local_model.eval()

        # Keep text embeddings on the target device for query embedding.
        self._local_text_embeddings = self._local_model.get_input_embeddings().to(
            device=target_device
        )

        # Extract hidden size.
        hf_config = getattr(self._local_model, "config", None)
        text_config = getattr(hf_config, "text_config", None)
        self._hidden_size = getattr(text_config, "hidden_size", None)
        if self._hidden_size is None:
            self._hidden_size = getattr(hf_config, "hidden_size", None)

        self.encoding_manager = GlobalEncodingManager(
            model=self._local_model,
            model_path=self.model_path,
            hf_config=hf_config,
        )
        self.encoding_manager.register_single_frame_encoder(self.embed_frame)
        self.feature_layout = self.encoding_manager.feature_layout

        logger.info(
            "LlavaOneVisionAdapter: local encoding ready. "
            "hidden_size=%s device=%s layout=%s",
            self._hidden_size,
            target_device,
            self.feature_layout.value,
        )

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_retrieval_hidden_size(self) -> int:
        if self._hidden_size is not None:
            return int(self._hidden_size)
        if self._local_text_embeddings is not None:
            emb_weight = getattr(self._local_text_embeddings, "weight", None)
            if emb_weight is not None and emb_weight.dim() == 2:
                return int(emb_weight.shape[1])
        raise RuntimeError(
            "LlavaOneVisionAdapter: hidden size is unavailable. "
            "Call init_local_encoding() or try_init_from_scheduler_model() first."
        )

    def embed_frame(
        self,
        frames,
        hot_token_mask: Optional[Any] = None,
        mask_mode: str = "off",
        cache_grid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode frames using LLaVA-OneVision image features.

        Parameters
        ----------
        frames:
            BGR frame array, shape ``(H, W, 3)`` or ``(N, H, W, 3)``.
        hot_token_mask:
            Optional shared boolean mask over the post-processed per-image
            token sequence. When ``mask_mode="approx"``, the same mask is
            applied to every compatible frame in the batch.
        mask_mode:
            ``"off"`` or ``"approx"``. Approx mode returns only the hot-token
            subset for each image when the shared mask matches the per-image
            token count after LLaVA feature normalization/capping.
        cache_grid:
            If true, prefer the fixed spatial cache grid (for example 14x14)
            for dense representative-frame embeddings. This is intentionally
            opt-in so normal recent-frame and cache=False cluster paths keep
            using LLaVA's generation-compatible image-feature layout.

        Returns
        -------
        llm_embedding : Tensor, shape ``(num_tokens, D_llm)``
        dummy_grid_thw : Tensor, shape ``(1, 3)`` — placeholder for Qwen-style
            grid; LLaVA does not use ``image_grid_thw`` / mRoPE.
        retrieval_embedding : Tensor, shape ``(num_tokens, D_ret)``
        """
        active_model = self._get_active_model()
        if active_model is None or self.encoding_manager is None:
            raise NotImplementedError(
                "LlavaOneVisionAdapter: encoding manager is not initialized; "
                "cannot encode frames."
            )

        from PIL import Image as PILImage

        frame_list = self._prepare_resized_rgb_frames(frames)
        pil_images = [PILImage.fromarray(frame) for frame in frame_list]
        inputs = self._build_generation_compatible_image_inputs(pil_images)
        visual = self.encoding_manager.visual
        pixel_values = inputs["pixel_values"].to(
            device=visual.device, dtype=visual.dtype,
        )
        image_sizes = inputs.get("image_sizes", None)
        if image_sizes is not None:
            if not isinstance(image_sizes, torch.Tensor):
                image_sizes = torch.as_tensor(image_sizes, dtype=torch.long)
            image_sizes = image_sizes.to(device=visual.device)
        batch_num_images = inputs.get("batch_num_images", None)
        if batch_num_images is not None:
            if not isinstance(batch_num_images, torch.Tensor):
                batch_num_images = torch.as_tensor(batch_num_images, dtype=torch.long)
            batch_num_images = batch_num_images.to(device=visual.device)

        with torch.no_grad():
            use_approx_mask = (
                hot_token_mask is not None and str(mask_mode).lower() == "approx"
            )
            sparse_spatial_result = None
            if use_approx_mask:
                from sglang.srt.stream_engine.config import Config

                if bool(getattr(Config, "llava_ov_sparse_visual_encode", True)):
                    sparse_spatial_result = self._encode_spatial_image_features(
                        pixel_values=pixel_values,
                        hot_token_mask=hot_token_mask,
                        sparse=True,
                    )
                if sparse_spatial_result is None:
                    raise RuntimeError(
                        "LlavaOneVisionAdapter: sparse cluster cache requested "
                        "mask_mode='approx', but SigLIP hot-patch encoding is "
                        "unavailable for this model/input."
                    )
            elif cache_grid and self._should_use_spatial_cache_grid():
                sparse_spatial_result = self._encode_spatial_image_features(
                    pixel_values=pixel_values,
                    sparse=False,
                )
                if sparse_spatial_result is None:
                    raise RuntimeError(
                        "LlavaOneVisionAdapter: cache_grid=True requires a fixed "
                        "spatial SigLIP cache grid, but it is unavailable for "
                        "this model/input."
                    )

            if sparse_spatial_result is not None:
                per_image_embeddings, dummy_grid_thw = sparse_spatial_result
            else:
                encode_kwargs: Dict[str, Any] = {}
                if image_sizes is not None:
                    encode_kwargs["image_sizes"] = image_sizes
                if batch_num_images is not None:
                    encode_kwargs["batch_num_images"] = batch_num_images
                image_features = self.encoding_manager.encode(
                    pixel_values,
                    **encode_kwargs,
                )
                per_image_embeddings = [
                    self._cap_image_feature_tokens(feature)
                    for feature in self._normalize_image_feature_batches(image_features)
                ]
                dummy_grid_thw = torch.tensor(
                    [[1, int(feature.shape[0]), 1] for feature in per_image_embeddings],
                    dtype=torch.long,
                    device=per_image_embeddings[0].device,
                )

        llm_embedding = torch.cat(per_image_embeddings, dim=0)

        # Project to retrieval dimension.
        retrieval_dim = self.get_retrieval_hidden_size()
        retrieval_embedding = QwenVLAdapter._project_feature_to_dim(
            llm_embedding, retrieval_dim,
        )

        return llm_embedding, dummy_grid_thw, retrieval_embedding

    def get_query_embedding(self, query_text: str) -> torch.Tensor:
        """Return token-level text embeddings for retrieval, shape ``(S, D)``."""
        processor = self._ensure_query_encoder_ready()
        tokenizer = getattr(processor, "tokenizer", processor)
        inputs = tokenizer(text=query_text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        emb_weight = getattr(self._local_text_embeddings, "weight", None)
        embedding_device = (
            emb_weight.device if emb_weight is not None else torch.device("cpu")
        )
        input_ids = input_ids.to(embedding_device)

        with torch.no_grad():
            query_embeddings = self._local_text_embeddings(input_ids)  # (1, S, D)

        return query_embeddings.squeeze(0)  # (S, D)

    # ------------------------------------------------------------------
    # Generation-payload construction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_generation_query(query: str) -> str:
        """Remove upstream multimodal/chat wrappers before building chatml-llava."""
        cleaned = str(query or "").replace("\r\n", "\n")

        cleaned = re.sub(
            r"^\s*<\|im_start\|>system\s*.*?<\|im_end\|>\s*",
            "",
            cleaned,
            count=1,
            flags=re.DOTALL,
        )
        cleaned = re.sub(
            r"^\s*<\|im_start\|>user\s*",
            "",
            cleaned,
            count=1,
        )
        cleaned = re.sub(
            r"\s*<\|im_end\|>\s*(?:<\|im_start\|>assistant\s*)?$",
            "",
            cleaned,
            count=1,
            flags=re.DOTALL,
        )
        cleaned = re.sub(r"<(?:image|video)>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _build_chatml_llava_prompt(num_images: int, query: str) -> str:
        image_prefix = "".join("<image>\n" for _ in range(num_images))
        user_content = image_prefix + query if query else image_prefix.rstrip("\n")
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _validate_prompt_image_alignment(self, prompt: str, expected_count: int) -> None:
        processor = self._ensure_processor()
        tokenizer = getattr(processor, "tokenizer", None)
        image_token_id = getattr(processor, "image_token_id", None)
        if tokenizer is None or image_token_id is None:
            return

        input_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids.flatten().tolist()
        actual_count = sum(1 for token_id in input_ids if token_id == image_token_id)
        if actual_count != int(expected_count):
            raise ValueError(
                "LlavaOneVisionAdapter: prompt/image alignment mismatch after chatml formatting. "
                f"expected_image_tokens={int(expected_count)} actual_image_tokens={int(actual_count)} "
                f"prompt_preview={prompt[:240]!r}"
            )

    def build_generation_payload(
        self,
        all_embeddings: List[torch.Tensor],
        all_grid_thws: List[torch.Tensor],
        query: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Build a multi-image generation payload for LLaVA-OneVision.

        Retrieved image embeddings are concatenated into a single precomputed
        payload item, while ``image_feature_lens`` preserves per-image
        boundaries for the LLaVA multimodal processor.

        The ``format: "precomputed_embedding"`` transport requires T5 to be
        implemented on the SGLang LLaVA processor side.
        """
        if not all_embeddings:
            return query, {}

        image_placeholders: List[str] = []
        image_feature_lens: List[int] = []
        normalized_embeddings: List[torch.Tensor] = []

        for idx, emb in enumerate(all_embeddings):
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            normalized_embeddings.append(emb)

            feature_len = int(emb.shape[0])
            if idx < len(all_grid_thws):
                grid = all_grid_thws[idx]
                grid = grid.squeeze(0) if grid.dim() == 2 else grid
                if grid.numel() >= 3:
                    inferred_len = int(torch.prod(grid.to(dtype=torch.long)).item())
                    if inferred_len > 0 and inferred_len == int(emb.shape[0]):
                        feature_len = inferred_len
            image_feature_lens.append(feature_len)
            image_placeholders.append("<image>")

        normalized_query = self._normalize_generation_query(query)
        prompt = self._build_chatml_llava_prompt(
            num_images=len(image_placeholders),
            query=normalized_query,
        )
        try:
            self._validate_prompt_image_alignment(
                prompt=prompt,
                expected_count=len(image_feature_lens),
            )
        except Exception as exc:
            logger.error(
                "LlavaOneVisionAdapter: invalid chatml-llava prompt/image alignment. %s",
                exc,
            )
            raise

        concatenated_embeddings = torch.cat(normalized_embeddings, dim=0)
        generate_kwargs: Dict[str, Any] = {
            "image_data": [
                {
                    "format": "precomputed_embedding",
                    "feature": concatenated_embeddings,
                    "image_feature_lens": image_feature_lens,
                }
            ]
        }
        return prompt, generate_kwargs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_adapter(model_path: str) -> BaseStreamModelAdapter:
    """Instantiate the correct adapter for *model_path*.

    Falls back to :class:`QwenVLAdapter` for unknown families so that
    existing Qwen-based scripts keep working.
    """
    family = detect_model_family(model_path)
    if family == ModelFamily.QWEN_VL:
        logger.info("create_adapter: selected QwenVLAdapter for '%s'", model_path)
        return QwenVLAdapter(model_path)
    if family == ModelFamily.LLAVA_ONEVISION:
        logger.info(
            "create_adapter: selected LlavaOneVisionAdapter for '%s'", model_path
        )
        return LlavaOneVisionAdapter(model_path)
    logger.warning(
        "create_adapter: unknown model family for '%s'; defaulting to QwenVLAdapter.",
        model_path,
    )
    return QwenVLAdapter(model_path)
