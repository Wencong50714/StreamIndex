from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from sglang.srt.stream_engine.config import Config
from sglang.srt.stream_engine.types import Cluster


@dataclass
class TokenCacheEntry:
    token_pos: int
    token_embedding: torch.Tensor
    last_used_ts: float
    motion_confidence: float = 1.0


@dataclass
class ClusterTokenCache:
    representative_tokens: torch.Tensor
    representative_retrieval_tokens: torch.Tensor
    representative_grid_thw: torch.Tensor
    grid_hw: Tuple[int, int]
    token_count: int

    @classmethod
    def from_cluster(cls, cluster: Cluster) -> "ClusterTokenCache":
        if cluster.representative_embedding is None:
            raise ValueError("Cluster representative_embedding is required for cache.")

        rep_llm_embedding, rep_grid_thw, rep_retrieval_embedding = cluster.representative_embedding
        grid = rep_grid_thw.squeeze(0) if rep_grid_thw.dim() == 2 else rep_grid_thw
        h_token, w_token = int(grid[-2]), int(grid[-1])
        rep_tokens = rep_llm_embedding.reshape(-1, rep_llm_embedding.shape[-1]).detach().clone()
        rep_retrieval_tokens = (
            rep_retrieval_embedding.reshape(-1, rep_retrieval_embedding.shape[-1])
            .detach()
            .clone()
        )
        return cls(
            representative_tokens=rep_tokens,
            representative_retrieval_tokens=rep_retrieval_tokens,
            representative_grid_thw=grid.detach().clone(),
            grid_hw=(h_token, w_token),
            token_count=int(rep_tokens.shape[0]),
        )

    def is_compatible(self, grid_thw: torch.Tensor, token_count: int) -> bool:
        grid = grid_thw.squeeze(0) if grid_thw.dim() == 2 else grid_thw
        h_token, w_token = int(grid[-2]), int(grid[-1])
        return self.grid_hw == (h_token, w_token) and self.token_count == token_count


class EmbeddingCacheManager:
    """Ephemeral cluster-level token cache helper for a single encode pass."""

    @staticmethod
    def _resolve_selected_frames(cluster: Cluster) -> List:
        selected = cluster.get_selected_frames()
        return selected if selected else cluster.frames

    @staticmethod
    def _iter_dense_batch_groups(frames: List) -> List[List]:
        groups: List[List] = []
        current_group: List = []
        current_signature = None

        for frame in frames:
            image = frame.image
            if not isinstance(image, np.ndarray):
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_signature = None
                groups.append([frame])
                continue

            signature = (image.shape, image.dtype.str)
            if current_group and signature != current_signature:
                groups.append(current_group)
                current_group = []
            if not current_group:
                current_signature = signature
            current_group.append(frame)

        if current_group:
            groups.append(current_group)
        return groups

    @staticmethod
    def _split_batched_embeddings(
        llm_embedding: torch.Tensor,
        grid_thw: torch.Tensor,
        retrieval_embedding: torch.Tensor,
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        normalized_grid_thw = grid_thw.unsqueeze(0) if grid_thw.dim() == 1 else grid_thw
        if normalized_grid_thw.dim() != 2 or normalized_grid_thw.shape[0] <= 0:
            return None

        grid_token_counts: List[int] = []
        total_grid_token_count = 0
        for row in normalized_grid_thw:
            token_count = int(torch.prod(row.to(dtype=torch.long)).item())
            if token_count <= 0:
                return None
            grid_token_counts.append(token_count)
            total_grid_token_count += token_count

        total_embedding_tokens = int(llm_embedding.shape[0])
        total_retrieval_tokens = int(retrieval_embedding.shape[0])
        if total_embedding_tokens != total_retrieval_tokens:
            return None

        if total_grid_token_count == total_embedding_tokens:
            token_counts = grid_token_counts
        else:
            if (
                total_embedding_tokens <= 0
                or total_grid_token_count % total_embedding_tokens != 0
            ):
                return None

            merge_divisor = total_grid_token_count // total_embedding_tokens
            if merge_divisor <= 0:
                return None

            token_counts = []
            for token_count in grid_token_counts:
                if token_count % merge_divisor != 0:
                    return None
                token_counts.append(token_count // merge_divisor)

        if sum(token_counts) != total_embedding_tokens:
            return None

        split_results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        start = 0
        for row, token_count in zip(normalized_grid_thw, token_counts):
            end = start + token_count
            split_results.append(
                (
                    llm_embedding[start:end],
                    row,
                    retrieval_embedding[start:end],
                )
            )
            start = end
        return split_results

    @classmethod
    def _encode_dense_frame_group(
        cls,
        cluster: Cluster,
        frame_group: List,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if len(frame_group) == 1:
            frame = frame_group[0]
            return [cluster.embed_fn(frame.image)]

        if not all(isinstance(frame.image, np.ndarray) for frame in frame_group):
            return [cluster.embed_fn(frame.image) for frame in frame_group]

        try:
            batch_images = np.stack([frame.image for frame in frame_group], axis=0)
            batch_llm_embedding, batch_grid_thw, batch_retrieval_embedding = cluster.embed_fn(
                batch_images
            )
        except (TypeError, ValueError, RuntimeError):
            return [cluster.embed_fn(frame.image) for frame in frame_group]

        split_results = cls._split_batched_embeddings(
            llm_embedding=batch_llm_embedding,
            grid_thw=batch_grid_thw,
            retrieval_embedding=batch_retrieval_embedding,
        )
        if split_results is None or len(split_results) != len(frame_group):
            return [cluster.embed_fn(frame.image) for frame in frame_group]
        return split_results

    @classmethod
    def _iter_sparse_batch_groups(
        cls,
        frames: List,
        hot_token_count: int,
        max_batch_frames: int,
        max_batch_tokens: int,
    ) -> List[List]:
        if not frames:
            return []

        max_batch_frames = max(int(max_batch_frames), 1)
        max_batch_tokens = max(int(max_batch_tokens), 1)
        tokens_per_frame = max(int(hot_token_count), 1)
        token_limited_frames = max(max_batch_tokens // tokens_per_frame, 1)
        group_limit = max(1, min(max_batch_frames, token_limited_frames))

        batches: List[List] = []
        for frame_group in cls._iter_dense_batch_groups(frames):
            for start in range(0, len(frame_group), group_limit):
                batches.append(frame_group[start : start + group_limit])
        return batches

    @classmethod
    def _encode_sparse_frame_group(
        cls,
        cluster: Cluster,
        frame_group: List,
        hot_mask_np: np.ndarray,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if len(frame_group) > 1 and all(isinstance(f.image, np.ndarray) for f in frame_group):
            try:
                batch_images = np.stack([f.image for f in frame_group], axis=0)
                try:
                    batch_llm, batch_grid, batch_ret = cluster.embed_fn(
                        batch_images, hot_token_mask=hot_mask_np, mask_mode="approx"
                    )
                except TypeError:
                    batch_llm, batch_grid, batch_ret = cluster.embed_fn(batch_images)
                split = cls._split_batched_embeddings(batch_llm, batch_grid, batch_ret)
                if split is not None and len(split) == len(frame_group):
                    return split
            except (TypeError, ValueError, RuntimeError):
                pass

        outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for frame in frame_group:
            try:
                outputs.append(
                    cluster.embed_fn(frame.image, hot_token_mask=hot_mask_np, mask_mode="approx")
                )
            except TypeError:
                outputs.append(cluster.embed_fn(frame.image))
        return outputs

    @staticmethod
    def _scatter_sparse_frame_result(
        cache_entry: ClusterTokenCache,
        hot_token_indices: torch.Tensor,
        hot_token_count: int,
        hot_llm_embedding: Optional[torch.Tensor],
        hot_retrieval_embedding: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        full_token_count = cache_entry.token_count
        full_grid = cache_entry.representative_grid_thw.to(
            dtype=torch.long,
            device=cache_entry.representative_grid_thw.device,
        )
        rep_tokens = cache_entry.representative_tokens
        rep_retrieval_tokens = cache_entry.representative_retrieval_tokens

        if hot_token_count <= 0:
            return (
                rep_tokens,
                full_grid,
                rep_retrieval_tokens,
                {
                    "encoded_token_count": 0.0,
                    "reused_token_count": float(full_token_count),
                    "total_token_count": float(full_token_count),
                    "sparse_vit_encoded_token_count": 0.0,
                    "sparse_vit_total_token_count": float(full_token_count),
                    "sparse_vit_fallback_count": 0.0,
                },
            )

        if hot_llm_embedding is None or hot_retrieval_embedding is None:
            raise ValueError("Sparse scatter requires both llm and retrieval embeddings.")

        raw_hot_llm_tokens = hot_llm_embedding.reshape(-1, hot_llm_embedding.shape[-1])
        raw_hot_retrieval_tokens = hot_retrieval_embedding.reshape(
            -1, hot_retrieval_embedding.shape[-1]
        )
        encoded_token_count = float(raw_hot_llm_tokens.shape[0])
        used_actual_sparse_vit = int(raw_hot_llm_tokens.shape[0]) == int(hot_token_count)
        sparse_vit_encoded_token_count = (
            float(hot_token_count)
            if used_actual_sparse_vit
            else float(raw_hot_llm_tokens.shape[0])
        )
        sparse_vit_fallback_count = 0.0 if used_actual_sparse_vit else 1.0

        scatter_indices = hot_token_indices.to(dtype=torch.long)
        hot_llm_tokens = raw_hot_llm_tokens
        hot_retrieval_tokens = raw_hot_retrieval_tokens
        if int(raw_hot_llm_tokens.shape[0]) != hot_token_count:
            hot_llm_tokens = raw_hot_llm_tokens[
                scatter_indices.to(device=raw_hot_llm_tokens.device)
            ]
        if int(raw_hot_retrieval_tokens.shape[0]) != hot_token_count:
            hot_retrieval_tokens = raw_hot_retrieval_tokens[
                scatter_indices.to(device=raw_hot_retrieval_tokens.device)
            ]

        llm_tokens = rep_tokens.to(
            device=hot_llm_tokens.device,
            dtype=hot_llm_tokens.dtype,
        ).clone()
        retrieval_tokens = rep_retrieval_tokens.to(
            device=hot_retrieval_tokens.device,
            dtype=hot_retrieval_tokens.dtype,
        ).clone()
        llm_tokens[scatter_indices.to(device=llm_tokens.device)] = hot_llm_tokens
        retrieval_tokens[
            scatter_indices.to(device=retrieval_tokens.device)
        ] = hot_retrieval_tokens

        return (
            llm_tokens,
            full_grid,
            retrieval_tokens,
            {
                "encoded_token_count": encoded_token_count,
                "reused_token_count": float(full_token_count - hot_token_count),
                "total_token_count": float(full_token_count),
                "sparse_vit_encoded_token_count": sparse_vit_encoded_token_count,
                "sparse_vit_total_token_count": float(full_token_count),
                "sparse_vit_fallback_count": sparse_vit_fallback_count,
            },
        )

    @classmethod
    def _encode_sparse_cluster_frames(
        cls,
        cluster: Cluster,
        valid_frames: List,
        cache_entry: ClusterTokenCache,
        hot_mask_np: np.ndarray,
        hot_token_indices: torch.Tensor,
        max_batch_frames: int,
        max_batch_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        if cluster.embed_fn is None:
            raise ValueError("cluster.embed_fn is required for encoding.")

        hot_token_count = int(hot_token_indices.numel())
        llm_embeddings: List[torch.Tensor] = []
        grid_thws: List[torch.Tensor] = []
        retrieval_embeddings: List[torch.Tensor] = []
        encoded_token_count = 0.0
        reused_token_count = 0.0
        total_token_count = 0.0
        sparse_vit_encoded_token_count = 0.0
        sparse_vit_total_token_count = 0.0
        sparse_vit_fallback_count = 0.0

        if hot_token_count <= 0:
            for _ in valid_frames:
                llm_tokens, grid_thw, retrieval_tokens, token_stats = cls._scatter_sparse_frame_result(
                    cache_entry=cache_entry,
                    hot_token_indices=hot_token_indices,
                    hot_token_count=hot_token_count,
                    hot_llm_embedding=None,
                    hot_retrieval_embedding=None,
                )
                llm_embeddings.append(llm_tokens)
                grid_thws.append(grid_thw)
                retrieval_embeddings.append(retrieval_tokens)
                encoded_token_count += float(token_stats["encoded_token_count"])
                reused_token_count += float(token_stats["reused_token_count"])
                total_token_count += float(token_stats["total_token_count"])
                sparse_vit_encoded_token_count += float(
                    token_stats.get("sparse_vit_encoded_token_count", 0.0)
                )
                sparse_vit_total_token_count += float(
                    token_stats.get("sparse_vit_total_token_count", 0.0)
                )
                sparse_vit_fallback_count += float(
                    token_stats.get("sparse_vit_fallback_count", 0.0)
                )
        else:
            for frame_group in cls._iter_sparse_batch_groups(
                valid_frames,
                hot_token_count=hot_token_count,
                max_batch_frames=max_batch_frames,
                max_batch_tokens=max_batch_tokens,
            ):
                encoded_group = cls._encode_sparse_frame_group(
                    cluster=cluster,
                    frame_group=frame_group,
                    hot_mask_np=hot_mask_np,
                )
                for hot_llm_embedding, _grid_thw, hot_retrieval_embedding in encoded_group:
                    llm_tokens, grid_thw, retrieval_tokens, token_stats = cls._scatter_sparse_frame_result(
                        cache_entry=cache_entry,
                        hot_token_indices=hot_token_indices,
                        hot_token_count=hot_token_count,
                        hot_llm_embedding=hot_llm_embedding,
                        hot_retrieval_embedding=hot_retrieval_embedding,
                    )
                    llm_embeddings.append(llm_tokens)
                    grid_thws.append(grid_thw)
                    retrieval_embeddings.append(retrieval_tokens)
                    encoded_token_count += float(token_stats["encoded_token_count"])
                    reused_token_count += float(token_stats["reused_token_count"])
                    total_token_count += float(token_stats["total_token_count"])
                    sparse_vit_encoded_token_count += float(
                        token_stats.get("sparse_vit_encoded_token_count", 0.0)
                    )
                    sparse_vit_total_token_count += float(
                        token_stats.get("sparse_vit_total_token_count", 0.0)
                    )
                    sparse_vit_fallback_count += float(
                        token_stats.get("sparse_vit_fallback_count", 0.0)
                    )

        if not llm_embeddings:
            return None

        selected_frame_count = len(valid_frames)
        cache_ratio = (
            reused_token_count / total_token_count
            if total_token_count > 0
            else 0.0
        )
        return {
            "llm_embedding": torch.cat(llm_embeddings, dim=0),
            "grid_thw": torch.stack(grid_thws, dim=0),
            "retrieval_embedding": torch.cat(retrieval_embeddings, dim=0),
            "cache_ratio": cache_ratio,
            "selected_frame_count": selected_frame_count,
            "fwd_encoded_frames": float(selected_frame_count) * (1.0 - cache_ratio),
            "token_stats": {
                "cache_hit_ratio": cache_ratio,
                "encoded_token_count": float(encoded_token_count),
                "reused_token_count": float(reused_token_count),
                "total_token_count": float(total_token_count),
                "sparse_vit_encoded_token_count": float(sparse_vit_encoded_token_count),
                "sparse_vit_total_token_count": float(sparse_vit_total_token_count),
                "sparse_vit_fallback_count": float(sparse_vit_fallback_count),
            },
        }

    @staticmethod
    def _get_cluster_spatial_motion_map(
        cluster: Cluster,
        aggregation: str = "mean",
    ) -> Optional[np.ndarray]:
        normalized_aggregation = str(aggregation or "mean").strip().lower()
        if normalized_aggregation == "mean" and cluster.spatial_motion_map is not None:
            return cluster.spatial_motion_map

        motion_tensor = cluster.motion_tensor
        if motion_tensor is None or motion_tensor.size == 0:
            return None

        if cluster.selected_frame_ids:
            selected_id_set = set(cluster.selected_frame_ids)
            selected_indices = [
                idx for idx, frame in enumerate(cluster.frames)
                if frame.frame_id in selected_id_set
            ]
        else:
            selected_indices = list(range(len(cluster.frames)))

        selected_indices = [
            idx for idx in selected_indices
            if idx < int(motion_tensor.shape[0])
        ]
        if not selected_indices:
            return None

        selected_motion = motion_tensor[selected_indices]
        if normalized_aggregation == "max":
            return np.max(selected_motion, axis=0).astype(np.float32, copy=False)
        if normalized_aggregation == "median":
            return np.median(selected_motion, axis=0).astype(np.float32, copy=False)
        return np.mean(selected_motion, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _heatmap_to_token_motion(
        heatmap: np.ndarray,
        h_token: int,
        w_token: int,
    ) -> np.ndarray:
        if heatmap.ndim == 3:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            heatmap.astype(np.float32),
            dsize=(w_token, h_token),
            interpolation=cv2.INTER_AREA,
        )
        max_val = float(np.max(resized))
        if max_val > 1.0:
            resized = resized / 255.0
        elif max_val > 0:
            resized = resized / max_val
        return resized

    @staticmethod
    def _motion_values_for_token_count(
        motion_map: np.ndarray,
        token_count: int,
    ) -> np.ndarray:
        values = motion_map.reshape(-1).astype(np.float32)
        if int(values.size) == int(token_count):
            return values
        # Some vision backbones spatially merge patches, so H*W from grid_thw
        # can be larger than the final encoded token count. Resize to token_count.
        resized = cv2.resize(
            motion_map.astype(np.float32),
            dsize=(int(token_count), 1),
            interpolation=cv2.INTER_AREA,
        )
        return resized.reshape(-1)

    @staticmethod
    def _budgeted_reuse_token_count(
        token_count: int,
        cold_token_ratio: float,
    ) -> Optional[int]:
        if cold_token_ratio < 0.0:
            return None
        ratio = min(max(float(cold_token_ratio), 0.0), 1.0)
        return min(token_count, int(round(token_count * ratio)))

    @staticmethod
    def _build_cold_mask(
        motion_values: np.ndarray,
        cold_token_threshold: float,
        cold_token_ratio: float,
        strategy: str = "cold",
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        token_count = int(motion_values.size)
        if token_count <= 0:
            return np.zeros((0,), dtype=bool)

        normalized_strategy = str(strategy or "cold").strip().lower()
        if normalized_strategy in {"motion", "median", "max"}:
            normalized_strategy = "cold"
        reuse_count = EmbeddingCacheManager._budgeted_reuse_token_count(
            token_count=token_count,
            cold_token_ratio=cold_token_ratio,
        )
        if reuse_count is not None:
            cold_mask = np.zeros(token_count, dtype=bool)
            if reuse_count <= 0:
                return cold_mask

            if normalized_strategy == "cold":
                cold_indices = np.argsort(motion_values, kind="stable")[:reuse_count]
            elif normalized_strategy == "hot":
                cold_indices = np.argsort(-motion_values, kind="stable")[:reuse_count]
            elif normalized_strategy == "random":
                seed = int(getattr(Config, "token_reuse_random_seed", 0))
                rng = np.random.default_rng(seed)
                cold_indices = rng.choice(token_count, size=reuse_count, replace=False)
            else:
                raise ValueError(
                    f"Unsupported token reuse strategy: {strategy}. "
                    "Expected one of: cold, random, hot."
                )
            cold_mask[cold_indices] = True
            return cold_mask

        if normalized_strategy != "cold":
            return np.zeros(token_count, dtype=bool)
        return motion_values < cold_token_threshold

    def encode_cluster(
        self,
        stream_id: str,
        cluster: Cluster,
        stage_mode: str,
        cold_token_threshold: float,
        cold_token_ratio: float = -1.0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        """Encode a retrieved cluster and apply cache policy.

        Stage A: full-frame encode then overwrite cold tokens.
        Stage B: derive one cluster-level hot mask from the motion map, batch
        encode only those hot visual tokens for selected frames, and scatter
        them back into the representative cache tokens.
        """
        if cluster.embed_fn is None:
            raise ValueError("cluster.embed_fn is required for encoding.")

        frames = self._resolve_selected_frames(cluster)
        if not frames:
            return None, None, None, {
                "cache_hit_ratio": 0.0,
                "encoded_token_count": 0.0,
                "reused_token_count": 0.0,
                "total_token_count": 0.0,
                "sparse_vit_encoded_token_count": 0.0,
                "sparse_vit_total_token_count": 0.0,
                "sparse_vit_fallback_count": 0.0,
            }

        effective_stage_mode = stage_mode if stage_mode in {"A", "B"} else "OFF"
        use_cache_overlay = (
            effective_stage_mode in {"A", "B"}
            and (cold_token_ratio >= 0.0 or cold_token_threshold >= 0.0)
        )
        cache_entry = ClusterTokenCache.from_cluster(cluster=cluster) if use_cache_overlay else None

        all_llm_embeddings: List[torch.Tensor] = []
        all_grid_thws: List[torch.Tensor] = []
        all_retrieval_embeddings: List[torch.Tensor] = []

        reused_token_count = 0
        total_token_count = 0
        encoded_token_count = 0
        sparse_vit_encoded_token_count = 0.0
        sparse_vit_total_token_count = 0.0
        sparse_vit_fallback_count = 0.0

        token_reuse_strategy = str(
            getattr(Config, "token_reuse_strategy", "cold") or "cold"
        ).strip().lower()
        if token_reuse_strategy in {"motion", "median", "max"}:
            token_reuse_strategy = "cold"
        motion_aggregation = "median" if token_reuse_strategy in {"cold", "hot"} else "mean"
        cluster_motion_map = self._get_cluster_spatial_motion_map(
            cluster,
            aggregation=motion_aggregation,
        )
        token_motion_cache: Dict[Tuple[int, int], np.ndarray] = {}

        valid_frames = [frame for frame in frames if frame.image is not None]

        if (
            effective_stage_mode == "B"
            and cache_entry is not None
            and (
                token_reuse_strategy == "random"
                or cluster_motion_map is not None
            )
        ):
            if cluster_motion_map is not None:
                motion_map = self._heatmap_to_token_motion(
                    cluster_motion_map,
                    h_token=cache_entry.grid_hw[0],
                    w_token=cache_entry.grid_hw[1],
                )
                motion_values = self._motion_values_for_token_count(
                    motion_map=motion_map,
                    token_count=cache_entry.token_count,
                )
            else:
                motion_values = np.zeros(cache_entry.token_count, dtype=np.float32)
            cold_mask_np = self._build_cold_mask(
                motion_values=motion_values,
                cold_token_threshold=cold_token_threshold,
                cold_token_ratio=cold_token_ratio,
                strategy=token_reuse_strategy,
                grid_hw=cache_entry.grid_hw,
            )
            hot_mask_np = np.logical_not(cold_mask_np)
            hot_token_indices = torch.from_numpy(np.nonzero(hot_mask_np)[0]).to(dtype=torch.long)
            sparse_result = self._encode_sparse_cluster_frames(
                cluster=cluster,
                valid_frames=valid_frames,
                cache_entry=cache_entry,
                hot_mask_np=hot_mask_np,
                hot_token_indices=hot_token_indices,
                max_batch_frames=int(getattr(Config, "query_sparse_max_batch_frames", 8)),
                max_batch_tokens=int(getattr(Config, "query_sparse_max_batch_tokens", 1024)),
            )
            if sparse_result is not None:
                all_llm_embeddings.append(sparse_result["llm_embedding"])
                all_grid_thws.extend([row for row in sparse_result["grid_thw"]])
                all_retrieval_embeddings.append(sparse_result["retrieval_embedding"])
                sparse_token_stats = sparse_result.get("token_stats", {})
                encoded_token_count += int(sparse_token_stats.get("encoded_token_count", 0.0))
                reused_token_count += int(sparse_token_stats.get("reused_token_count", 0.0))
                total_token_count += int(sparse_token_stats.get("total_token_count", 0.0))
                sparse_vit_encoded_token_count += float(
                    sparse_token_stats.get("sparse_vit_encoded_token_count", 0.0)
                )
                sparse_vit_total_token_count += float(
                    sparse_token_stats.get("sparse_vit_total_token_count", 0.0)
                )
                sparse_vit_fallback_count += float(
                    sparse_token_stats.get("sparse_vit_fallback_count", 0.0)
                )
            else:
                raise RuntimeError(
                    "EmbeddingCacheManager: Stage-B sparse cluster encode produced "
                    "no result; refusing to fall back to full-frame visual encode."
                )
        else:
            for frame_group in self._iter_dense_batch_groups(valid_frames):
                for llm_embedding, grid_thw, retrieval_embedding in self._encode_dense_frame_group(
                    cluster,
                    frame_group,
                ):
                    llm_tokens = llm_embedding.reshape(-1, llm_embedding.shape[-1])
                    token_count = int(llm_tokens.shape[0])
                    total_token_count += token_count
                    encoded_token_count += token_count

                    if (
                        effective_stage_mode == "A"
                        and cache_entry is not None
                        and cache_entry.is_compatible(grid_thw, token_count)
                        and (
                            token_reuse_strategy == "random"
                            or cluster_motion_map is not None
                        )
                    ):
                        grid = grid_thw.squeeze(0) if grid_thw.dim() == 2 else grid_thw
                        h_token, w_token = int(grid[-2]), int(grid[-1])
                        token_grid_key = (h_token, w_token)
                        if cluster_motion_map is not None:
                            motion_map = token_motion_cache.get(token_grid_key)
                            if motion_map is None:
                                motion_map = self._heatmap_to_token_motion(
                                    cluster_motion_map,
                                    h_token=h_token,
                                    w_token=w_token,
                                )
                                token_motion_cache[token_grid_key] = motion_map

                            motion_values = self._motion_values_for_token_count(
                                motion_map=motion_map,
                                token_count=token_count,
                            )
                        else:
                            motion_values = np.zeros(token_count, dtype=np.float32)
                        frame_cold_mask_np = self._build_cold_mask(
                            motion_values=motion_values,
                            cold_token_threshold=cold_token_threshold,
                            cold_token_ratio=cold_token_ratio,
                            strategy=token_reuse_strategy,
                            grid_hw=(h_token, w_token),
                        )
                        cold_mask = torch.from_numpy(frame_cold_mask_np).to(
                            device=llm_tokens.device,
                            dtype=torch.bool,
                        )
                        reusable_count = int(cold_mask.sum().item())
                        if reusable_count > 0:
                            rep_tokens = cache_entry.representative_tokens.to(
                                device=llm_tokens.device,
                                dtype=llm_tokens.dtype,
                            )
                            llm_tokens[cold_mask] = rep_tokens[cold_mask]
                            reused_token_count += reusable_count

                    all_llm_embeddings.append(llm_tokens)
                    all_grid_thws.append(
                        grid_thw.squeeze(0) if grid_thw.dim() == 2 else grid_thw
                    )
                    all_retrieval_embeddings.append(
                        retrieval_embedding.reshape(-1, retrieval_embedding.shape[-1])
                    )

        if not all_llm_embeddings:
            return None, None, None, {
                "cache_hit_ratio": 0.0,
                "encoded_token_count": float(encoded_token_count),
                "reused_token_count": 0.0,
                "total_token_count": float(total_token_count),
                "sparse_vit_encoded_token_count": float(
                    sparse_vit_encoded_token_count
                ),
                "sparse_vit_total_token_count": float(sparse_vit_total_token_count),
                "sparse_vit_fallback_count": float(sparse_vit_fallback_count),
            }

        stats = {
            "cache_hit_ratio": (float(reused_token_count) / float(total_token_count))
            if total_token_count > 0
            else 0.0,
            "encoded_token_count": float(encoded_token_count),
            "reused_token_count": float(reused_token_count),
            "total_token_count": float(total_token_count),
            "sparse_vit_encoded_token_count": float(sparse_vit_encoded_token_count),
            "sparse_vit_total_token_count": float(sparse_vit_total_token_count),
            "sparse_vit_fallback_count": float(sparse_vit_fallback_count),
        }
        return (
            torch.cat(all_llm_embeddings, dim=0),
            torch.stack(all_grid_thws, dim=0),
            torch.cat(all_retrieval_embeddings, dim=0),
            stats,
        )
