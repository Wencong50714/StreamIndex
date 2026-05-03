from __future__ import annotations

import asyncio
from collections import OrderedDict
import logging
import threading
import time
import uuid
from itertools import count
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np

from sglang.srt.stream_engine.config import Config
from sglang.srt.stream_engine.encoding_mgr import GlobalEncodingManager
from sglang.srt.stream_engine.model_adapter import (
    BaseStreamModelAdapter,
    ModelFamily,
    create_adapter,
    detect_model_family,
)
from sglang.srt.stream_engine.stream_memory import StreamingMemory
from sglang.srt.stream_engine.types import ContextItemKind, ContextItem, Frame
from sglang.srt.stream_engine.bench_parser import get_query_text_for_embedding

logger = logging.getLogger(__name__)

config = Config()


class StreamEngine:
    """
    High-level engine for streaming video inference with memory management.

    Model-family-specific behavior (encoding, prompt formatting, payload
    assembly) is fully delegated to a :class:`BaseStreamModelAdapter`
    instance.  ``StreamEngine`` itself owns only:

    * the sglang ``Engine`` lifecycle
    * stream creation / frame submission / query dispatch
    * retrieval orchestration (``StreamingMemory``, cluster encoding)
    * stats tracking

    Example::

        engine = StreamEngine(model_path="/path/to/model")
        stream_id = engine.create_stream("video.mp4")
        engine.submit_frame(stream_id, Frame(frame_id, frame_array))
        response = engine.submit_query(stream_id, "Describe the scene.")
        engine.close_stream(stream_id)
        engine.shutdown()
    """

    def __init__(
        self,
        model_path: str,
        max_streams: int = 32,
        tp_size: int = 1,
        chat_template: Optional[str] = None,
        **llm_kwargs,
    ):
        self.model_path = model_path
        self.max_streams = max_streams

        # Instantiate the model-family adapter (owns all model-specific logic).
        self._adapter: BaseStreamModelAdapter = create_adapter(model_path)
        logger.info(
            "StreamEngine: model_family=%s adapter=%s",
            self._adapter.family.value,
            type(self._adapter).__name__,
        )
        strategy_msg = (
            "StreamEngine config: "
            f"frame_select_strategy={Config.frame_select_strategy}; "
            f"cache_strategy={Config.token_reuse_strategy}"
        )
        print(strategy_msg)
        logger.info(strategy_msg)

        # Resolve chat template via adapter (caller may override).
        if chat_template is None:
            chat_template = self._adapter.resolve_chat_template()
        self.chat_template = chat_template

        # feature_layout is kept on StreamEngine for quick access;
        # updated from the adapter after encoding init completes.
        self.feature_layout = self._adapter.feature_layout

        # Initialize sglang LLM engine.
        import sglang as sgl

        engine_kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "tp_size": tp_size,
            "chat_template": chat_template,
        }
        engine_kwargs.update(llm_kwargs)
        if (
            "mem_fraction_static" not in engine_kwargs
            and Config.mem_fraction_static is not None
        ):
            engine_kwargs["mem_fraction_static"] = Config.mem_fraction_static
        engine_kwargs.setdefault("disable_cuda_graph", True)

        logger.info("Initializing sglang.Engine with kwargs: %s", engine_kwargs)
        self.llm_engine = sgl.Engine(**engine_kwargs)
        logger.info("sglang.Engine initialized.")

        # Initialize encoding (scheduler-side preferred; local HF fallback).
        self._init_encoding_manager()

        # Stream management.
        self._streams: dict = {}
        self._stream_counter = count()
        self._lock = threading.Lock()
        self._request_counter = count()

        # A dedicated loop/thread for async_generate-based sync submission.
        # This keeps all async submission on one loop and avoids multi-thread
        # run_until_complete re-entry errors.
        self._sync_submit_loop: Optional[asyncio.AbstractEventLoop] = None
        self._sync_submit_thread: Optional[threading.Thread] = None
        self._sync_submit_loop_ready = threading.Event()
        self._sync_submit_init_lock = threading.Lock()

        # Results storage.
        self._results: dict = {}

        logger.info("StreamEngine initialized with model: %s", model_path)

    # ------------------------------------------------------------------
    # Async generate infrastructure
    # ------------------------------------------------------------------

    def _start_sync_submit_loop(self) -> None:
        with self._sync_submit_init_lock:
            if self._sync_submit_loop is not None:
                return

            self._sync_submit_loop_ready.clear()

            loop = asyncio.new_event_loop()

            def _runner() -> None:
                asyncio.set_event_loop(loop)
                self._sync_submit_loop_ready.set()
                loop.run_forever()

            thread = threading.Thread(
                target=_runner,
                name="stream-engine-sync-submit-loop",
                daemon=True,
            )
            thread.start()
            self._sync_submit_loop_ready.wait()
            self._sync_submit_loop = loop
            self._sync_submit_thread = thread

    def _generate_via_sync_submit_loop(
        self,
        *,
        prompt: str,
        sampling_params: Dict[str, Any],
        generate_kwargs: Dict[str, Any],
    ) -> Any:
        if self._sync_submit_loop is None:
            self._start_sync_submit_loop()

        if self._sync_submit_loop is None:
            raise RuntimeError("sync submit event loop is not initialized")

        future = asyncio.run_coroutine_threadsafe(
            self.llm_engine.async_generate(
                prompt=prompt,
                sampling_params=sampling_params,
                **generate_kwargs,
            ),
            self._sync_submit_loop,
        )
        return future.result()

    # ------------------------------------------------------------------
    # Encoding-manager initialization
    # ------------------------------------------------------------------

    def _init_encoding_manager(self) -> None:
        """Try scheduler-side model access first; fall back to local HF model."""
       
        self._adapter.init_local_encoding()
        self.feature_layout = self._adapter.feature_layout
        print(
            "GlobalEncodingManager active (local fallback). layout=%s",
            self.feature_layout.value,
        )

    # ------------------------------------------------------------------
    # Convenience property
    # ------------------------------------------------------------------

    @property
    def encoding_manager(self) -> Optional[GlobalEncodingManager]:
        """Expose the adapter's encoding manager for cluster encoding and stats."""
        return self._adapter.encoding_manager

    # ------------------------------------------------------------------
    # Stream lifecycle
    # ------------------------------------------------------------------

    def create_stream(
        self,
        video_path: str,
        sample_fps: int = 1,
    ) -> str:
        """Create a new stream and return its identifier."""
        with self._lock:
            if len(self._streams) >= self.max_streams:
                raise RuntimeError(
                    f"Max streams ({self.max_streams}) reached. "
                    "Close an existing stream before creating a new one."
                )

            stream_id = f"stream_{next(self._stream_counter)}_{uuid.uuid4().hex[:8]}"

            if self._adapter.encoding_manager is not None:
                embed_fn = self._adapter.encoding_manager.encode_single_frame
            else:
                def embed_fn(image, **kwargs):
                    raise NotImplementedError(
                        "GlobalEncodingManager is not initialized; "
                        "cannot encode frames yet."
                    )

            memory = StreamingMemory(embed_fn=embed_fn)
            self._streams[stream_id] = {
                "memory": memory,
                "video_path": video_path,
                "sample_fps": sample_fps,
                "query_encoded_frames_total": 0.0,
                "recent_m_encoded_frames_total": 0.0,
                "total_encoded_frames": 0.0,
                "fwd_encoded_frames": 0.0,
                "cluster_cpu_cache": {},
                "cluster_gpu_cache": OrderedDict(),
            }

            logger.info("Created stream '%s' for video: %s", stream_id, video_path)
            return stream_id

    def submit_frame(
        self,
        stream_id: str,
        frame: Frame,
    ) -> None:
        """Submit a frame to a stream."""
        with self._lock:
            if stream_id not in self._streams:
                raise ValueError(f"Stream '{stream_id}' not found.")
            self._streams[stream_id]["memory"].add_frame(frame)
            logger.debug("Submitted frame %d to stream '%s'", frame.frame_id, stream_id)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def submit_query(
        self,
        stream_id: str,
        query: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        return_stats: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """Submit a query to a stream and return the generated response.

        Args:
            stream_id: Identifier of the target stream.
            query: The query string.
            temperature: Sampling temperature.
            max_tokens: Maximum number of new tokens to generate.
            return_stats: When True, return ``{"text": ..., "stats": ...}``.

        Returns:
            Response text, or a dict when *return_stats* is True.
        """
        submit_start = time.perf_counter()
        with self._lock:
            if stream_id not in self._streams:
                raise ValueError(f"Stream '{stream_id}' not found.")
            stream_state = self._streams[stream_id]
            memory = stream_state["memory"]
            stream_state["fwd_encoded_frames"] = 0.0
            cluster_cpu_cache = stream_state.setdefault("cluster_cpu_cache", {})
            cluster_gpu_cache = stream_state.setdefault("cluster_gpu_cache", OrderedDict())
            query_encoded_frames_total = float(
                stream_state.get("query_encoded_frames_total", 0.0)
            )
            recent_m_encoded_frames_total = float(
                stream_state.get("recent_m_encoded_frames_total", 0.0)
            )

        # Step 1: Encode query for retrieval.
        retrieval_start = time.perf_counter()
        query_text = get_query_text_for_embedding(query)
        query_embed = self._adapter.get_query_embedding(query_text)

        # Step 2: Retrieve context from memory.
        context_items, history_count = memory.retrieve_info(
            query_embed=query_embed,
            stream_id=stream_id,
        )
        retrieval_s = time.perf_counter() - retrieval_start

        # Step 3: Encode context items into visual embeddings.
        embed_fn = memory.embed_fn

        all_embeddings: List[torch.Tensor] = []
        all_grid_thws: List[torch.Tensor] = []
        cache_reused_token_count = 0.0
        cache_total_token_count = 0.0
        encoded_vision_token_count = 0
        sparse_vit_encoded_token_count = 0.0
        sparse_vit_total_token_count = 0.0
        sparse_vit_fallback_count = 0.0
        fwd_encoded_frames = 0.0
        recent_m_encoded_frames_this_query = 0.0
        recent_frame_encode_s = 0.0
        cluster_encode_s = 0.0
        target_embed_device = (
            self._adapter.encoding_manager.visual.device
            if self._adapter.encoding_manager is not None
            else None
        )
        use_abs_mem = bool(getattr(Config, "use_abs_mem", True))

        def _append_encoded_frame_embedding(
            llm_embedding: torch.Tensor,
            grid_thw: torch.Tensor,
            *,
            count_recent_m_cost: bool = False,
        ) -> None:
            nonlocal encoded_vision_token_count, fwd_encoded_frames, recent_m_encoded_frames_this_query
            if llm_embedding.dim() == 1:
                llm_embedding = llm_embedding.unsqueeze(0)
            all_embeddings.append(llm_embedding)
            encoded_vision_token_count += int(llm_embedding.shape[0])
            fwd_encoded_frames += 1.0
            if count_recent_m_cost:
                recent_m_encoded_frames_this_query += 1.0
            all_grid_thws.append(
                grid_thw.squeeze(0) if grid_thw.dim() == 2 else grid_thw
            )

        def _append_encoded_frames(
            frames: Optional[List[Frame]],
            *,
            count_recent_m_cost: bool = False,
        ) -> None:
            nonlocal recent_frame_encode_s
            if not frames:
                return
            frames_to_embed = [frame for frame in frames if frame.image is not None]
            if not frames_to_embed:
                return

            encode_start = time.perf_counter()

            batch_images: Optional[np.ndarray] = None
            if (
                len(frames_to_embed) > 1
                and all(isinstance(frame.image, np.ndarray) for frame in frames_to_embed)
            ):
                first_shape = frames_to_embed[0].image.shape
                if all(frame.image.shape == first_shape for frame in frames_to_embed):
                    batch_images = np.stack([frame.image for frame in frames_to_embed], axis=0)

            if batch_images is not None:
                try:
                    llm_embedding, grid_thw, _ = embed_fn(batch_images)
                    normalized_grid_thw = (
                        grid_thw.unsqueeze(0) if grid_thw.dim() == 1 else grid_thw
                    )
                    token_counts = _infer_embedding_token_counts(
                        normalized_grid_thw,
                        int(llm_embedding.shape[0]),
                    )

                    if token_counts:
                        start = 0
                        for row, token_count in zip(normalized_grid_thw, token_counts):
                            end = start + token_count
                            _append_encoded_frame_embedding(
                                llm_embedding[start:end],
                                row,
                                count_recent_m_cost=count_recent_m_cost,
                            )
                            start = end
                        recent_frame_encode_s += time.perf_counter() - encode_start
                        return
                except (TypeError, ValueError, NotImplementedError, RuntimeError):
                    # Fall back to the per-frame path when the encoder does not
                    # accept batched inputs or cannot preserve the batch layout.
                    pass

            for frame in frames_to_embed:
                llm_embedding, grid_thw, _ = embed_fn(frame.image)
                _append_encoded_frame_embedding(
                    llm_embedding,
                    grid_thw,
                    count_recent_m_cost=count_recent_m_cost,
                )
            recent_frame_encode_s += time.perf_counter() - encode_start

        def _infer_embedding_token_counts(
            normalized_grid_thw: torch.Tensor,
            total_embedding_tokens: int,
        ) -> List[int]:
            if normalized_grid_thw.dim() != 2 or normalized_grid_thw.shape[0] <= 0:
                return []

            grid_token_counts: List[int] = []
            total_grid_tokens = 0
            for row in normalized_grid_thw:
                token_count = int(torch.prod(row.to(dtype=torch.long)).item())
                if token_count <= 0:
                    return []
                grid_token_counts.append(token_count)
                total_grid_tokens += token_count

            if total_grid_tokens == total_embedding_tokens:
                return grid_token_counts
            if total_embedding_tokens <= 0 or total_grid_tokens % total_embedding_tokens != 0:
                return []

            merge_divisor = total_grid_tokens // total_embedding_tokens
            if merge_divisor <= 0:
                return []

            inferred_counts: List[int] = []
            for token_count in grid_token_counts:
                if token_count % merge_divisor != 0:
                    return []
                inferred_counts.append(token_count // merge_divisor)

            if sum(inferred_counts) != total_embedding_tokens:
                return []
            return inferred_counts

        def _append_precomputed_embedding(
            llm_embedding: Optional[torch.Tensor],
            grid_thw: Optional[torch.Tensor],
        ) -> None:
            if llm_embedding is None:
                return

            if llm_embedding.dim() == 1:
                llm_embedding = llm_embedding.unsqueeze(0)

            if grid_thw is None:
                all_embeddings.append(llm_embedding)
                return

            normalized_grid_thw = (
                grid_thw.unsqueeze(0) if grid_thw.dim() == 1 else grid_thw
            )
            token_counts: List[int] = []
            total_token_count = 0
            for row in normalized_grid_thw:
                token_count = int(torch.prod(row.to(dtype=torch.long)).item())
                if token_count <= 0:
                    token_counts = []
                    break
                token_counts.append(token_count)
                total_token_count += token_count

            if token_counts and total_token_count == int(llm_embedding.shape[0]):
                start = 0
                for row, token_count in zip(normalized_grid_thw, token_counts):
                    end = start + token_count
                    all_embeddings.append(llm_embedding[start:end])
                    all_grid_thws.append(row)
                    start = end
                return

            all_embeddings.append(llm_embedding)
            if normalized_grid_thw.shape[0] == 1:
                all_grid_thws.append(normalized_grid_thw.squeeze(0))
            else:
                all_grid_thws.extend([row for row in normalized_grid_thw])

        def _cluster_cache_key(cluster) -> Optional[tuple]:
            if cluster is None:
                return None
            return (
                int(cluster.cluster_id),
                tuple(cluster.selected_frame_ids or []),
                bool(Config.enable_motion_frame_select),
                str(getattr(Config, "frame_select_strategy", "motion")),
                bool(Config.enable_embedding_cache),
                str(Config.cache_stage_mode),
                str(getattr(Config, "token_reuse_strategy", "cold")),
                int(getattr(Config, "token_reuse_random_seed", 0)),
                float(Config.cold_token_threshold),
                float(Config.cold_token_ratio),
            )

        def _cluster_gpu_cache_enabled() -> bool:
            if target_embed_device is None or target_embed_device.type != "cuda":
                return False
            if not getattr(Config, "enable_cluster_gpu_cache", True):
                return False
            return int(getattr(Config, "cluster_gpu_cache_max_entries", 4)) > 0

        def _store_gpu_cached_cluster_payload(
            cluster_cache_key: Optional[tuple],
            cluster_llm_emb: Optional[torch.Tensor],
            cluster_grid_thw: Optional[torch.Tensor],
            token_stats: Optional[Dict[str, Any]] = None,
        ) -> None:
            if (
                cluster_cache_key is None
                or not _cluster_gpu_cache_enabled()
                or cluster_llm_emb is None
                or cluster_grid_thw is None
            ):
                return

            cluster_gpu_cache[cluster_cache_key] = {
                "llm_embedding_gpu": cluster_llm_emb.detach(),
                "grid_thw_gpu": cluster_grid_thw.detach(),
                "token_stats": dict(token_stats or {}),
            }
            cluster_gpu_cache.move_to_end(cluster_cache_key)

            max_entries = max(int(getattr(Config, "cluster_gpu_cache_max_entries", 4)), 1)
            while len(cluster_gpu_cache) > max_entries:
                cluster_gpu_cache.popitem(last=False)

        def _load_cached_cluster_payload(
            cluster_cache_key: Optional[tuple],
        ) -> Optional[Dict[str, Any]]:
            if cluster_cache_key is None:
                return None

            if _cluster_gpu_cache_enabled():
                gpu_cached_payload = cluster_gpu_cache.get(cluster_cache_key)
                if gpu_cached_payload is not None:
                    cluster_gpu_cache.move_to_end(cluster_cache_key)
                    return {
                        "llm_embedding": gpu_cached_payload.get("llm_embedding_gpu"),
                        "grid_thw": gpu_cached_payload.get("grid_thw_gpu"),
                        "token_stats": gpu_cached_payload.get("token_stats", {}),
                    }

            cpu_cached_payload = cluster_cpu_cache.get(cluster_cache_key)
            if cpu_cached_payload is None:
                return None

            cluster_llm_emb = cpu_cached_payload.get("llm_embedding_cpu")
            cluster_grid_thw = cpu_cached_payload.get("grid_thw_cpu")
            if cluster_llm_emb is not None and target_embed_device is not None:
                cluster_llm_emb = cluster_llm_emb.to(device=target_embed_device)
            if cluster_grid_thw is not None and target_embed_device is not None:
                cluster_grid_thw = cluster_grid_thw.to(device=target_embed_device)

            token_stats = cpu_cached_payload.get("token_stats", {})
            _store_gpu_cached_cluster_payload(
                cluster_cache_key=cluster_cache_key,
                cluster_llm_emb=cluster_llm_emb,
                cluster_grid_thw=cluster_grid_thw,
                token_stats=token_stats,
            )
            return {
                "llm_embedding": cluster_llm_emb,
                "grid_thw": cluster_grid_thw,
                "token_stats": token_stats,
            }

        cluster_runtime: Dict[int, Dict[str, Any]] = {}
        for item in context_items:
            if item.kind != ContextItemKind.CLUSTER or item.cluster is None:
                continue

            if item.selected_frame_ids is not None:
                item.cluster.selected_frame_ids = item.selected_frame_ids

            use_embedding_cache = (
                Config.enable_retrieval_optimizations
                and Config.enable_embedding_cache
            )
            use_cluster_cpu_cache = (
                Config.enable_retrieval_optimizations
                and Config.enable_cluster_cpu_cache
            )
            stage_mode = Config.cache_stage_mode if use_embedding_cache else "OFF"
            cold_token_threshold = (
                Config.cold_token_threshold if use_embedding_cache else -1.0
            )
            cold_token_ratio = (
                Config.cold_token_ratio if use_embedding_cache else -1.0
            )

            cluster_cache_key = _cluster_cache_key(item.cluster) if use_cluster_cpu_cache else None
            cached_cluster_payload = _load_cached_cluster_payload(cluster_cache_key)
            runtime_state: Dict[str, Any] = {
                "cluster_cache_key": cluster_cache_key,
                "use_cluster_cpu_cache": use_cluster_cpu_cache,
                "stage_mode": stage_mode,
                "cold_token_threshold": cold_token_threshold,
                "cold_token_ratio": cold_token_ratio,
            }
            if cached_cluster_payload is not None:
                runtime_state["cached_payload"] = cached_cluster_payload

            cluster_runtime[int(item.cluster.cluster_id)] = runtime_state

        for item in context_items:
            if item.kind == ContextItemKind.FRAME and item.frames:
                _append_encoded_frames(item.frames, count_recent_m_cost=True)

            elif item.kind == ContextItemKind.CLUSTER:
                if item.cluster is None:
                    _append_encoded_frames(item.frames)
                    continue

                runtime_state = cluster_runtime.get(int(item.cluster.cluster_id), {})
                cluster_cache_key = runtime_state.get("cluster_cache_key")
                cached_payload = runtime_state.get("cached_payload")

                if cached_payload is not None:
                    cluster_llm_emb = cached_payload.get("llm_embedding")
                    cluster_grid_thw = cached_payload.get("grid_thw")
                    token_stats = cached_payload.get("token_stats", {})
                    cached_total_tokens = float(token_stats.get("total_token_count", 0.0))
                    cache_reused_token_count += cached_total_tokens
                    cache_total_token_count += cached_total_tokens
                else:
                    if self._adapter.encoding_manager is None:
                        raise RuntimeError(
                            "GlobalEncodingManager is not initialized for cluster encoding."
                        )
                    stage_mode = runtime_state.get("stage_mode", "OFF")
                    cold_token_threshold = runtime_state.get("cold_token_threshold", -1.0)
                    cold_token_ratio = runtime_state.get("cold_token_ratio", -1.0)
                    cluster_encode_start = time.perf_counter()
                    cluster_result = self._adapter.encoding_manager.encode_cluster(
                        stream_id=item.stream_id or stream_id,
                        cluster=item.cluster,
                        stage_mode=stage_mode,
                        cold_token_threshold=cold_token_threshold,
                        cold_token_ratio=cold_token_ratio,
                    )
                    cluster_llm_emb = cluster_result.get("llm_embedding")
                    cluster_grid_thw = cluster_result.get("grid_thw")
                    token_stats = cluster_result.get("token_stats", {})
                    encoded_vision_token_count += int(
                        token_stats.get("encoded_token_count", 0.0)
                    )
                    sparse_vit_encoded_token_count += float(
                        token_stats.get("sparse_vit_encoded_token_count", 0.0)
                    )
                    sparse_vit_total_token_count += float(
                        token_stats.get("sparse_vit_total_token_count", 0.0)
                    )
                    sparse_vit_fallback_count += float(
                        token_stats.get("sparse_vit_fallback_count", 0.0)
                    )
                    cache_reused_token_count += float(
                        token_stats.get("reused_token_count", 0.0)
                    )
                    cache_total_token_count += float(
                        token_stats.get("total_token_count", 0.0)
                    )
                    fwd_encoded_frames += float(
                        cluster_result.get("fwd_encoded_frames", 0.0)
                    )
                    cluster_encode_s += time.perf_counter() - cluster_encode_start
                    if runtime_state.get("use_cluster_cpu_cache") and cluster_cache_key is not None:
                        cluster_cpu_cache[cluster_cache_key] = {
                            "llm_embedding_cpu": (
                                cluster_llm_emb.detach().to(device="cpu")
                                if cluster_llm_emb is not None else None
                            ),
                            "grid_thw_cpu": (
                                cluster_grid_thw.detach().to(device="cpu")
                                if cluster_grid_thw is not None else None
                            ),
                            "retrieval_embedding_cpu": (
                                cluster_result.get("retrieval_embedding").detach().to(device="cpu")
                                if cluster_result.get("retrieval_embedding") is not None else None
                            ),
                            "selected_frame_count": int(
                                cluster_result.get("selected_frame_count", 0)
                            ),
                            "cache_ratio": float(cluster_result.get("cache_ratio", 0.0)),
                            "token_stats": dict(token_stats),
                        }
                        _store_gpu_cached_cluster_payload(
                            cluster_cache_key=cluster_cache_key,
                            cluster_llm_emb=cluster_llm_emb,
                            cluster_grid_thw=cluster_grid_thw,
                            token_stats=token_stats,
                        )

                _append_precomputed_embedding(cluster_llm_emb, cluster_grid_thw)

            elif item.kind == ContextItemKind.ABSTRACT_MEMORY and use_abs_mem:
                abs_embedding = item.abs_mem_llm_emb
                if abs_embedding is None:
                    abs_embedding = item.abs_mem_emb
                _append_precomputed_embedding(abs_embedding, item.grid_thw)

        if cache_total_token_count > 0:
            logger.info(
                "Embedding cache stats: hit_ratio=%.4f reused=%d total=%d",
                cache_reused_token_count / cache_total_token_count,
                int(cache_reused_token_count),
                int(cache_total_token_count),
            )

        if not all_embeddings:
            raise RuntimeError("No multimodal embeddings found in retrieved context.")

        # Step 4: Build prompt + generation payload (adapter-specific).
        sampling_params = {"max_new_tokens": max_tokens, "temperature": temperature}
        prompt, generate_kwargs = self._adapter.build_generation_payload(
            all_embeddings, all_grid_thws, query
        )

        # Step 5: Run generation.
        generate_start = time.perf_counter()
        output = self.llm_engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            **generate_kwargs,
        )
        generate_e2e_s = time.perf_counter() - generate_start
        total_submit_s = time.perf_counter() - submit_start

        output_text = output["text"] if isinstance(output, dict) else str(output)

        # Update per-stream stats.
        query_non_recent_encoded_frames = max(
            float(fwd_encoded_frames) - float(recent_m_encoded_frames_this_query), 0.0
        )
        query_encoded_frames_total += query_non_recent_encoded_frames
        recent_m_encoded_frames_total += float(recent_m_encoded_frames_this_query)
        ingest_encoded_frames = float(getattr(memory, "ingest_encoded_frames", 0.0))
        total_encoded_frames = (
            query_encoded_frames_total
            + ingest_encoded_frames
            + recent_m_encoded_frames_total
        )
        with self._lock:
            if stream_id in self._streams:
                self._streams[stream_id]["query_encoded_frames_total"] = float(
                    query_encoded_frames_total
                )
                self._streams[stream_id]["recent_m_encoded_frames_total"] = float(
                    recent_m_encoded_frames_total
                )
                self._streams[stream_id]["fwd_encoded_frames"] = float(fwd_encoded_frames)
                self._streams[stream_id]["total_encoded_frames"] = float(total_encoded_frames)

        stats: Dict[str, Any] = {
            "encoded_vision_tokens": int(encoded_vision_token_count),
            "sparse_vit_encoded_tokens": int(sparse_vit_encoded_token_count),
            "sparse_vit_total_tokens": int(sparse_vit_total_token_count),
            "sparse_vit_fallback_count": int(sparse_vit_fallback_count),
            "sparse_vit_compute_ratio": (
                float(sparse_vit_encoded_token_count) / float(sparse_vit_total_token_count)
                if sparse_vit_total_token_count > 0 else 0.0
            ),
            "cache_reused_tokens": int(cache_reused_token_count),
            "cache_total_tokens": int(cache_total_token_count),
            "cache_hit_ratio": (
                float(cache_reused_token_count) / float(cache_total_token_count)
                if cache_total_token_count > 0 else 0.0
            ),
            "fwd_encoded_frames": float(fwd_encoded_frames),
            "total_encoded_frames": float(total_encoded_frames),
            "recent_m_encoded_frames": float(recent_m_encoded_frames_this_query),
            "recent_m_encoded_frames_total": float(recent_m_encoded_frames_total),
            "ingest_encoded_frames_total": float(ingest_encoded_frames),
            "ttft_s": None,
            "retrieval_s": float(retrieval_s),
            "recent_frame_encode_s": float(recent_frame_encode_s),
            "cluster_encode_s": float(cluster_encode_s),
            "generate_s": float(generate_e2e_s),
            "submit_query_s": float(total_submit_s),
            "e2e_s": float(generate_e2e_s),
        }

        logger.info(
            "submit_query stats: stream_id=%s encoded_vision_tokens=%d "
            "fwd_encoded_frames=%.3f total_encoded_frames=%.3f "
            "cache_hit_ratio=%.4f retrieval=%.3fs recent_frame_encode=%.3fs "
            "cluster_encode=%.3fs generate=%.3fs submit_query=%.3fs",
            stream_id,
            stats["encoded_vision_tokens"],
            stats["fwd_encoded_frames"],
            stats["total_encoded_frames"],
            stats["cache_hit_ratio"],
            stats["retrieval_s"],
            stats["recent_frame_encode_s"],
            stats["cluster_encode_s"],
            stats["generate_s"],
            stats["submit_query_s"],
        )

        if return_stats:
            return {"text": output_text, "stats": stats}
        return output_text

    def sync_submit_query(
        self,
        stream_id: str,
        query: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        return_stats: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """Submit a query using ``async_generate`` on a dedicated single event loop.

        This API is intended for concurrent multi-stream callers that need a
        thread-safe synchronous interface without changing ``submit_query``.
        The internal encode/cache/retrieval logic is identical to ``submit_query``;
        only the final LLM call routes through ``async_generate``.
        """
        submit_start = time.perf_counter()
        with self._lock:
            if stream_id not in self._streams:
                raise ValueError(f"Stream '{stream_id}' not found.")
            stream_state = self._streams[stream_id]
            memory = stream_state["memory"]
            stream_state["fwd_encoded_frames"] = 0.0
            cluster_cpu_cache = stream_state.setdefault("cluster_cpu_cache", {})
            cluster_gpu_cache = stream_state.setdefault("cluster_gpu_cache", OrderedDict())
            query_encoded_frames_total = float(
                stream_state.get("query_encoded_frames_total", 0.0)
            )
            recent_m_encoded_frames_total = float(
                stream_state.get("recent_m_encoded_frames_total", 0.0)
            )

        # Step 1: Encode query for retrieval.
        retrieval_start = time.perf_counter()
        query_text = get_query_text_for_embedding(query)
        query_embed = self._adapter.get_query_embedding(query_text)

        # Step 2: Retrieve context from memory.
        context_items, history_count = memory.retrieve_info(
            query_embed=query_embed,
            stream_id=stream_id,
        )
        retrieval_s = time.perf_counter() - retrieval_start

        # Step 3: Encode context items into visual embeddings.
        embed_fn = memory.embed_fn

        all_embeddings: List[torch.Tensor] = []
        all_grid_thws: List[torch.Tensor] = []
        cache_reused_token_count = 0.0
        cache_total_token_count = 0.0
        encoded_vision_token_count = 0
        sparse_vit_encoded_token_count = 0.0
        sparse_vit_total_token_count = 0.0
        sparse_vit_fallback_count = 0.0
        fwd_encoded_frames = 0.0
        recent_m_encoded_frames_this_query = 0.0
        recent_frame_encode_s = 0.0
        cluster_encode_s = 0.0
        target_embed_device = (
            self._adapter.encoding_manager.visual.device
            if self._adapter.encoding_manager is not None
            else None
        )
        use_abs_mem = bool(getattr(Config, "use_abs_mem", True))

        def _append_encoded_frame_embedding(
            llm_embedding: torch.Tensor,
            grid_thw: torch.Tensor,
            *,
            count_recent_m_cost: bool = False,
        ) -> None:
            nonlocal encoded_vision_token_count, fwd_encoded_frames, recent_m_encoded_frames_this_query
            if llm_embedding.dim() == 1:
                llm_embedding = llm_embedding.unsqueeze(0)
            all_embeddings.append(llm_embedding)
            encoded_vision_token_count += int(llm_embedding.shape[0])
            fwd_encoded_frames += 1.0
            if count_recent_m_cost:
                recent_m_encoded_frames_this_query += 1.0
            all_grid_thws.append(
                grid_thw.squeeze(0) if grid_thw.dim() == 2 else grid_thw
            )

        def _append_encoded_frames(
            frames: Optional[List[Frame]],
            *,
            count_recent_m_cost: bool = False,
        ) -> None:
            nonlocal recent_frame_encode_s
            if not frames:
                return
            frames_to_embed = [frame for frame in frames if frame.image is not None]
            if not frames_to_embed:
                return

            encode_start = time.perf_counter()

            batch_images: Optional[np.ndarray] = None
            if (
                len(frames_to_embed) > 1
                and all(isinstance(frame.image, np.ndarray) for frame in frames_to_embed)
            ):
                first_shape = frames_to_embed[0].image.shape
                if all(frame.image.shape == first_shape for frame in frames_to_embed):
                    batch_images = np.stack([frame.image for frame in frames_to_embed], axis=0)

            if batch_images is not None:
                try:
                    llm_embedding, grid_thw, _ = embed_fn(batch_images)
                    normalized_grid_thw = (
                        grid_thw.unsqueeze(0) if grid_thw.dim() == 1 else grid_thw
                    )
                    token_counts = _infer_embedding_token_counts(
                        normalized_grid_thw,
                        int(llm_embedding.shape[0]),
                    )

                    if token_counts:
                        start = 0
                        for row, token_count in zip(normalized_grid_thw, token_counts):
                            end = start + token_count
                            _append_encoded_frame_embedding(
                                llm_embedding[start:end],
                                row,
                                count_recent_m_cost=count_recent_m_cost,
                            )
                            start = end
                        recent_frame_encode_s += time.perf_counter() - encode_start
                        return
                except (TypeError, ValueError, NotImplementedError, RuntimeError):
                    # Fall back to the per-frame path when the encoder does not
                    # accept batched inputs or cannot preserve the batch layout.
                    pass

            for frame in frames_to_embed:
                llm_embedding, grid_thw, _ = embed_fn(frame.image)
                _append_encoded_frame_embedding(
                    llm_embedding,
                    grid_thw,
                    count_recent_m_cost=count_recent_m_cost,
                )
            recent_frame_encode_s += time.perf_counter() - encode_start

        def _infer_embedding_token_counts(
            normalized_grid_thw: torch.Tensor,
            total_embedding_tokens: int,
        ) -> List[int]:
            if normalized_grid_thw.dim() != 2 or normalized_grid_thw.shape[0] <= 0:
                return []

            grid_token_counts: List[int] = []
            total_grid_tokens = 0
            for row in normalized_grid_thw:
                token_count = int(torch.prod(row.to(dtype=torch.long)).item())
                if token_count <= 0:
                    return []
                grid_token_counts.append(token_count)
                total_grid_tokens += token_count

            if total_grid_tokens == total_embedding_tokens:
                return grid_token_counts
            if total_embedding_tokens <= 0 or total_grid_tokens % total_embedding_tokens != 0:
                return []

            merge_divisor = total_grid_tokens // total_embedding_tokens
            if merge_divisor <= 0:
                return []

            inferred_counts: List[int] = []
            for token_count in grid_token_counts:
                if token_count % merge_divisor != 0:
                    return []
                inferred_counts.append(token_count // merge_divisor)

            if sum(inferred_counts) != total_embedding_tokens:
                return []
            return inferred_counts

        def _append_precomputed_embedding(
            llm_embedding: Optional[torch.Tensor],
            grid_thw: Optional[torch.Tensor],
        ) -> None:
            if llm_embedding is None:
                return

            if llm_embedding.dim() == 1:
                llm_embedding = llm_embedding.unsqueeze(0)

            if grid_thw is None:
                all_embeddings.append(llm_embedding)
                return

            normalized_grid_thw = (
                grid_thw.unsqueeze(0) if grid_thw.dim() == 1 else grid_thw
            )
            token_counts: List[int] = []
            total_token_count = 0
            for row in normalized_grid_thw:
                token_count = int(torch.prod(row.to(dtype=torch.long)).item())
                if token_count <= 0:
                    token_counts = []
                    break
                token_counts.append(token_count)
                total_token_count += token_count

            if token_counts and total_token_count == int(llm_embedding.shape[0]):
                start = 0
                for row, token_count in zip(normalized_grid_thw, token_counts):
                    end = start + token_count
                    all_embeddings.append(llm_embedding[start:end])
                    all_grid_thws.append(row)
                    start = end
                return

            all_embeddings.append(llm_embedding)
            if normalized_grid_thw.shape[0] == 1:
                all_grid_thws.append(normalized_grid_thw.squeeze(0))
            else:
                all_grid_thws.extend([row for row in normalized_grid_thw])

        def _cluster_cache_key(cluster) -> Optional[tuple]:
            if cluster is None:
                return None
            return (
                int(cluster.cluster_id),
                tuple(cluster.selected_frame_ids or []),
                bool(Config.enable_motion_frame_select),
                str(getattr(Config, "frame_select_strategy", "motion")),
                bool(Config.enable_embedding_cache),
                str(Config.cache_stage_mode),
                str(getattr(Config, "token_reuse_strategy", "cold")),
                int(getattr(Config, "token_reuse_random_seed", 0)),
                float(Config.cold_token_threshold),
                float(Config.cold_token_ratio),
            )

        def _cluster_gpu_cache_enabled() -> bool:
            if target_embed_device is None or target_embed_device.type != "cuda":
                return False
            if not getattr(Config, "enable_cluster_gpu_cache", True):
                return False
            return int(getattr(Config, "cluster_gpu_cache_max_entries", 4)) > 0

        def _store_gpu_cached_cluster_payload(
            cluster_cache_key: Optional[tuple],
            cluster_llm_emb: Optional[torch.Tensor],
            cluster_grid_thw: Optional[torch.Tensor],
            token_stats: Optional[Dict[str, Any]] = None,
        ) -> None:
            if (
                cluster_cache_key is None
                or not _cluster_gpu_cache_enabled()
                or cluster_llm_emb is None
                or cluster_grid_thw is None
            ):
                return

            cluster_gpu_cache[cluster_cache_key] = {
                "llm_embedding_gpu": cluster_llm_emb.detach(),
                "grid_thw_gpu": cluster_grid_thw.detach(),
                "token_stats": dict(token_stats or {}),
            }
            cluster_gpu_cache.move_to_end(cluster_cache_key)

            max_entries = max(int(getattr(Config, "cluster_gpu_cache_max_entries", 4)), 1)
            while len(cluster_gpu_cache) > max_entries:
                cluster_gpu_cache.popitem(last=False)

        def _load_cached_cluster_payload(
            cluster_cache_key: Optional[tuple],
        ) -> Optional[Dict[str, Any]]:
            if cluster_cache_key is None:
                return None

            if _cluster_gpu_cache_enabled():
                gpu_cached_payload = cluster_gpu_cache.get(cluster_cache_key)
                if gpu_cached_payload is not None:
                    cluster_gpu_cache.move_to_end(cluster_cache_key)
                    return {
                        "llm_embedding": gpu_cached_payload.get("llm_embedding_gpu"),
                        "grid_thw": gpu_cached_payload.get("grid_thw_gpu"),
                        "token_stats": gpu_cached_payload.get("token_stats", {}),
                    }

            cpu_cached_payload = cluster_cpu_cache.get(cluster_cache_key)
            if cpu_cached_payload is None:
                return None

            cluster_llm_emb = cpu_cached_payload.get("llm_embedding_cpu")
            cluster_grid_thw = cpu_cached_payload.get("grid_thw_cpu")
            if cluster_llm_emb is not None and target_embed_device is not None:
                cluster_llm_emb = cluster_llm_emb.to(device=target_embed_device)
            if cluster_grid_thw is not None and target_embed_device is not None:
                cluster_grid_thw = cluster_grid_thw.to(device=target_embed_device)

            token_stats = cpu_cached_payload.get("token_stats", {})
            _store_gpu_cached_cluster_payload(
                cluster_cache_key=cluster_cache_key,
                cluster_llm_emb=cluster_llm_emb,
                cluster_grid_thw=cluster_grid_thw,
                token_stats=token_stats,
            )
            return {
                "llm_embedding": cluster_llm_emb,
                "grid_thw": cluster_grid_thw,
                "token_stats": token_stats,
            }

        cluster_runtime: Dict[int, Dict[str, Any]] = {}
        for item in context_items:
            if item.kind != ContextItemKind.CLUSTER or item.cluster is None:
                continue

            if item.selected_frame_ids is not None:
                item.cluster.selected_frame_ids = item.selected_frame_ids

            use_embedding_cache = (
                Config.enable_retrieval_optimizations
                and Config.enable_embedding_cache
            )
            use_cluster_cpu_cache = (
                Config.enable_retrieval_optimizations
                and Config.enable_cluster_cpu_cache
            )
            stage_mode = Config.cache_stage_mode if use_embedding_cache else "OFF"
            cold_token_threshold = (
                Config.cold_token_threshold if use_embedding_cache else -1.0
            )
            cold_token_ratio = (
                Config.cold_token_ratio if use_embedding_cache else -1.0
            )

            cluster_cache_key = _cluster_cache_key(item.cluster) if use_cluster_cpu_cache else None
            cached_cluster_payload = _load_cached_cluster_payload(cluster_cache_key)
            runtime_state: Dict[str, Any] = {
                "cluster_cache_key": cluster_cache_key,
                "use_cluster_cpu_cache": use_cluster_cpu_cache,
                "stage_mode": stage_mode,
                "cold_token_threshold": cold_token_threshold,
                "cold_token_ratio": cold_token_ratio,
            }
            if cached_cluster_payload is not None:
                runtime_state["cached_payload"] = cached_cluster_payload

            cluster_runtime[int(item.cluster.cluster_id)] = runtime_state

        for item in context_items:
            if item.kind == ContextItemKind.FRAME and item.frames:
                _append_encoded_frames(item.frames, count_recent_m_cost=True)

            elif item.kind == ContextItemKind.CLUSTER:
                if item.cluster is None:
                    _append_encoded_frames(item.frames)
                    continue

                runtime_state = cluster_runtime.get(int(item.cluster.cluster_id), {})
                cluster_cache_key = runtime_state.get("cluster_cache_key")
                cached_payload = runtime_state.get("cached_payload")

                if cached_payload is not None:
                    cluster_llm_emb = cached_payload.get("llm_embedding")
                    cluster_grid_thw = cached_payload.get("grid_thw")
                    token_stats = cached_payload.get("token_stats", {})
                    cached_total_tokens = float(token_stats.get("total_token_count", 0.0))
                    cache_reused_token_count += cached_total_tokens
                    cache_total_token_count += cached_total_tokens
                else:
                    if self._adapter.encoding_manager is None:
                        raise RuntimeError(
                            "GlobalEncodingManager is not initialized for cluster encoding."
                        )
                    stage_mode = runtime_state.get("stage_mode", "OFF")
                    cold_token_threshold = runtime_state.get("cold_token_threshold", -1.0)
                    cold_token_ratio = runtime_state.get("cold_token_ratio", -1.0)
                    cluster_encode_start = time.perf_counter()
                    cluster_result = self._adapter.encoding_manager.encode_cluster(
                        stream_id=item.stream_id or stream_id,
                        cluster=item.cluster,
                        stage_mode=stage_mode,
                        cold_token_threshold=cold_token_threshold,
                        cold_token_ratio=cold_token_ratio,
                    )
                    cluster_llm_emb = cluster_result.get("llm_embedding")
                    cluster_grid_thw = cluster_result.get("grid_thw")
                    token_stats = cluster_result.get("token_stats", {})
                    encoded_vision_token_count += int(
                        token_stats.get("encoded_token_count", 0.0)
                    )
                    sparse_vit_encoded_token_count += float(
                        token_stats.get("sparse_vit_encoded_token_count", 0.0)
                    )
                    sparse_vit_total_token_count += float(
                        token_stats.get("sparse_vit_total_token_count", 0.0)
                    )
                    sparse_vit_fallback_count += float(
                        token_stats.get("sparse_vit_fallback_count", 0.0)
                    )
                    cache_reused_token_count += float(
                        token_stats.get("reused_token_count", 0.0)
                    )
                    cache_total_token_count += float(
                        token_stats.get("total_token_count", 0.0)
                    )
                    fwd_encoded_frames += float(
                        cluster_result.get("fwd_encoded_frames", 0.0)
                    )
                    cluster_encode_s += time.perf_counter() - cluster_encode_start
                    if runtime_state.get("use_cluster_cpu_cache") and cluster_cache_key is not None:
                        cluster_cpu_cache[cluster_cache_key] = {
                            "llm_embedding_cpu": (
                                cluster_llm_emb.detach().to(device="cpu")
                                if cluster_llm_emb is not None else None
                            ),
                            "grid_thw_cpu": (
                                cluster_grid_thw.detach().to(device="cpu")
                                if cluster_grid_thw is not None else None
                            ),
                            "retrieval_embedding_cpu": (
                                cluster_result.get("retrieval_embedding").detach().to(device="cpu")
                                if cluster_result.get("retrieval_embedding") is not None else None
                            ),
                            "selected_frame_count": int(
                                cluster_result.get("selected_frame_count", 0)
                            ),
                            "cache_ratio": float(cluster_result.get("cache_ratio", 0.0)),
                            "token_stats": dict(token_stats),
                        }
                        _store_gpu_cached_cluster_payload(
                            cluster_cache_key=cluster_cache_key,
                            cluster_llm_emb=cluster_llm_emb,
                            cluster_grid_thw=cluster_grid_thw,
                            token_stats=token_stats,
                        )

                _append_precomputed_embedding(cluster_llm_emb, cluster_grid_thw)

            elif item.kind == ContextItemKind.ABSTRACT_MEMORY and use_abs_mem:
                abs_embedding = item.abs_mem_llm_emb
                if abs_embedding is None:
                    abs_embedding = item.abs_mem_emb
                _append_precomputed_embedding(abs_embedding, item.grid_thw)

        if cache_total_token_count > 0:
            logger.info(
                "Embedding cache stats: hit_ratio=%.4f reused=%d total=%d",
                cache_reused_token_count / cache_total_token_count,
                int(cache_reused_token_count),
                int(cache_total_token_count),
            )

        if not all_embeddings:
            raise RuntimeError("No multimodal embeddings found in retrieved context.")

        # Step 4: Build prompt + generation payload (adapter-specific).
        sampling_params = {"max_new_tokens": max_tokens, "temperature": temperature}
        prompt, generate_kwargs = self._adapter.build_generation_payload(
            all_embeddings, all_grid_thws, query
        )

        # Step 5: Run generation via async_generate on the dedicated loop.
        generate_start = time.perf_counter()
        output = self._generate_via_sync_submit_loop(
            prompt=prompt,
            sampling_params=sampling_params,
            generate_kwargs=generate_kwargs,
        )
        generate_e2e_s = time.perf_counter() - generate_start
        total_submit_s = time.perf_counter() - submit_start

        output_text = output["text"] if isinstance(output, dict) else str(output)

        # Update per-stream stats.
        query_non_recent_encoded_frames = max(
            float(fwd_encoded_frames) - float(recent_m_encoded_frames_this_query), 0.0
        )
        query_encoded_frames_total += query_non_recent_encoded_frames
        recent_m_encoded_frames_total += float(recent_m_encoded_frames_this_query)
        ingest_encoded_frames = float(getattr(memory, "ingest_encoded_frames", 0.0))
        total_encoded_frames = (
            query_encoded_frames_total
            + ingest_encoded_frames
            + recent_m_encoded_frames_total
        )
        with self._lock:
            if stream_id in self._streams:
                self._streams[stream_id]["query_encoded_frames_total"] = float(
                    query_encoded_frames_total
                )
                self._streams[stream_id]["recent_m_encoded_frames_total"] = float(
                    recent_m_encoded_frames_total
                )
                self._streams[stream_id]["fwd_encoded_frames"] = float(fwd_encoded_frames)
                self._streams[stream_id]["total_encoded_frames"] = float(total_encoded_frames)

        stats: Dict[str, Any] = {
            "encoded_vision_tokens": int(encoded_vision_token_count),
            "sparse_vit_encoded_tokens": int(sparse_vit_encoded_token_count),
            "sparse_vit_total_tokens": int(sparse_vit_total_token_count),
            "sparse_vit_fallback_count": int(sparse_vit_fallback_count),
            "sparse_vit_compute_ratio": (
                float(sparse_vit_encoded_token_count) / float(sparse_vit_total_token_count)
                if sparse_vit_total_token_count > 0 else 0.0
            ),
            "cache_reused_tokens": int(cache_reused_token_count),
            "cache_total_tokens": int(cache_total_token_count),
            "cache_hit_ratio": (
                float(cache_reused_token_count) / float(cache_total_token_count)
                if cache_total_token_count > 0 else 0.0
            ),
            "fwd_encoded_frames": float(fwd_encoded_frames),
            "total_encoded_frames": float(total_encoded_frames),
            "recent_m_encoded_frames": float(recent_m_encoded_frames_this_query),
            "recent_m_encoded_frames_total": float(recent_m_encoded_frames_total),
            "ingest_encoded_frames_total": float(ingest_encoded_frames),
            "ttft_s": None,
            "retrieval_s": float(retrieval_s),
            "recent_frame_encode_s": float(recent_frame_encode_s),
            "cluster_encode_s": float(cluster_encode_s),
            "generate_s": float(generate_e2e_s),
            "submit_query_s": float(total_submit_s),
            "e2e_s": float(generate_e2e_s),
        }

        logger.info(
            "sync_submit_query stats: stream_id=%s encoded_vision_tokens=%d "
            "fwd_encoded_frames=%.3f total_encoded_frames=%.3f "
            "cache_hit_ratio=%.4f retrieval=%.3fs recent_frame_encode=%.3fs "
            "cluster_encode=%.3fs generate=%.3fs submit_query=%.3fs",
            stream_id,
            stats["encoded_vision_tokens"],
            stats["fwd_encoded_frames"],
            stats["total_encoded_frames"],
            stats["cache_hit_ratio"],
            stats["retrieval_s"],
            stats["recent_frame_encode_s"],
            stats["cluster_encode_s"],
            stats["generate_s"],
            stats["submit_query_s"],
        )

        if return_stats:
            return {"text": output_text, "stats": stats}
        return output_text

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close_stream(self, stream_id: str) -> None:
        """Close and remove a stream."""
        with self._lock:
            if stream_id in self._streams:
                del self._streams[stream_id]
                logger.info("Closed stream '%s'", stream_id)

    def shutdown(self) -> None:
        """Shut down the engine and release resources."""
        with self._lock:
            self._streams.clear()

        with self._sync_submit_init_lock:
            sync_submit_loop = self._sync_submit_loop
            sync_submit_thread = self._sync_submit_thread
            self._sync_submit_loop = None
            self._sync_submit_thread = None

        if sync_submit_loop is not None:
            sync_submit_loop.call_soon_threadsafe(sync_submit_loop.stop)
        if sync_submit_thread is not None:
            sync_submit_thread.join(timeout=2.0)

        if hasattr(self, "llm_engine") and self.llm_engine is not None:
            self.llm_engine.shutdown()
            logger.info("LLM engine shut down.")
