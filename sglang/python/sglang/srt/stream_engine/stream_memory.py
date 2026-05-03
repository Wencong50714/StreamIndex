"""
Streaming Memory Module for Video Frame Management.

This module provides a memory system that organizes video frames into clusters
based on visual similarity using Bhattacharyya distance on frame histograms.
It supports streaming ingestion of frames and provides methods for retrieving
cluster embeddings for downstream retrieval tasks.
"""

import logging
from typing import Callable, List, Optional, Tuple

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from sglang.srt.stream_engine.types import (
    ContextItemKind,
    ContextItem,
    Frame,
    Cluster,
)
from sglang.srt.stream_engine.config import Config
from sglang.srt.stream_engine.motion_analyzer import CPUMotionAnalyzer
from sglang.srt.stream_engine.frame_selector import MotionFrameSelector

logger = logging.getLogger(__name__)

EmbedFunction = Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
QwenEmbedding = Tuple[torch.Tensor, torch.Tensor]

class StreamingMemory:
    """
    A streaming memory system that organizes video frames into visual clusters.
    
    The memory maintains a buffer for incoming frames and groups them into
    clusters based on visual similarity. Each cluster stores a representative
    frame embedding for efficient retrieval.
    
    Attributes:
        embed_fn: Function to compute embeddings from frame images.
        buffer: Temporary buffer for incoming frames.
        buffer_size: Maximum size of the buffer before processing.
        clusters: List of completed clusters.
        current_cluster: The cluster currently being built.
        bhattacharyya_threshold: Threshold for determining cluster boundaries.
    """
    
    def __init__(self, embed_fn: EmbedFunction):
        """
        Initialize the streaming memory.
        
        Args:
            embed_fn: A callable that takes a frame image (np.ndarray) 
                     and returns its embedding (torch.Tensor).
            query_embed_fn: A callable that takes a query string and returns
                           its embedding (torch.Tensor). Required for history retrieval.
        """
        self._validate_embed_fn(embed_fn)
        self.embed_fn = embed_fn
        
        # Frame buffer for delayed processing
        self.buffer: List[Frame] = []
        self.buffer_size: int = Config.buffer_size
        
        # Cluster management
        self.clusters: List[Cluster] = []
        self.current_cluster: Optional[Cluster] = None
        self._cluster_id_counter: int = 0
        
        # Similarity threshold
        self.bhattacharyya_threshold: float = Config.bhattacharyya_threshold
        self.max_cluster_frames: int = Config.max_cluster_frames
        
        self.alpha: float = Config.alpha if hasattr(Config, 'alpha') else 0.1  # momentum parameter
        self.motion_analyzer = CPUMotionAnalyzer(score_method=Config.motion_score_method)
        self.frame_selector = MotionFrameSelector(
            min_frames=Config.motion_select_min_frames,
            temporal_bins=Config.motion_select_temporal_bins,
            significant_threshold=Config.motion_significant_threshold,
        )
        # Count representative-frame encodes performed during ingestion.
        self.ingest_encoded_frames: float = 0.0
        # Resident retrieval cache (stacked representative embeddings).
        self._cluster_lifecycle_version: int = 0
        self._resident_retrieval_version: int = -1
        self._resident_cluster_count: int = 0
        self._resident_rep_versions: Tuple[int, ...] = tuple()
        self._resident_cluster_embeddings: Optional[torch.Tensor] = None
        self._resident_cluster_grid_thws: Optional[torch.Tensor] = None
        self._resident_cluster_embeddings_norm: Optional[torch.Tensor] = None
    
    @staticmethod
    def _validate_embed_fn(embed_fn: EmbedFunction) -> None:
        """Validate that embed_fn is callable."""
        if not callable(embed_fn):
            raise TypeError("embed_fn must be a callable function.")

    @staticmethod
    def _should_sync_cuda_in_gen_mask() -> bool:
        """Return whether debug/profile sync is enabled for gen_mask()."""
        return bool(
            Config.enable_gen_mask_cuda_sync_debug
            or Config.enable_gen_mask_cuda_sync_profile
        )

    def _invalidate_resident_retrieval_cache(self) -> None:
        """Invalidate resident retrieval caches after lifecycle/reset changes."""
        self._resident_retrieval_version = -1
        self._resident_cluster_count = 0
        self._resident_rep_versions = tuple()
        self._resident_cluster_embeddings = None
        self._resident_cluster_grid_thws = None
        self._resident_cluster_embeddings_norm = None

    def _try_append_resident_retrieval_cache(self, cluster: Cluster, old_version: int) -> bool:
        """Incrementally append a finalized cluster into resident retrieval cache."""
        if not Config.enable_resident_retrieval_cache:
            return False
        if (
            self._resident_retrieval_version != old_version
            or self._resident_cluster_embeddings is None
            or self._resident_cluster_grid_thws is None
        ):
            return False
        if self._resident_cluster_count != len(self.clusters) - 1:
            return False

        cluster_emb = cluster.get_representative_embedding().unsqueeze(0)
        cluster_grid_thw = cluster.get_representative_grid_thw().unsqueeze(0)
        self._resident_cluster_embeddings = torch.cat(
            [self._resident_cluster_embeddings, cluster_emb], dim=0
        )
        self._resident_cluster_grid_thws = torch.cat(
            [self._resident_cluster_grid_thws, cluster_grid_thw], dim=0
        )
        self._resident_rep_versions = (
            *self._resident_rep_versions,
            cluster.representative_embedding_version,
        )
        self._resident_cluster_count += 1
        if (
            Config.enable_resident_retrieval_norm_cache
            and self._resident_cluster_embeddings_norm is not None
        ):
            cluster_emb_norm = F.normalize(cluster_emb, p=2, dim=-1)
            self._resident_cluster_embeddings_norm = torch.cat(
                [self._resident_cluster_embeddings_norm, cluster_emb_norm], dim=0
            )
        self._resident_retrieval_version = self._cluster_lifecycle_version
        return True

    def _rebuild_resident_retrieval_cache(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build resident stacked retrieval caches from completed clusters."""
        if not self.clusters:
            self._invalidate_resident_retrieval_cache()
            self._resident_retrieval_version = self._cluster_lifecycle_version
            return None, None, None

        embeddings: List[torch.Tensor] = []
        grid_thws: List[torch.Tensor] = []
        rep_versions: List[int] = []

        for cluster in self.clusters:
            emb = cluster.get_representative_embedding()
            if emb is None:
                continue
            embeddings.append(emb)
            grid_thws.append(cluster.get_representative_grid_thw())
            rep_versions.append(cluster.representative_embedding_version)

        if not embeddings:
            self._invalidate_resident_retrieval_cache()
            self._resident_retrieval_version = self._cluster_lifecycle_version
            return None, None, None

        stacked_embeddings = torch.stack(embeddings, dim=0)
        stacked_grid_thws = torch.stack(grid_thws, dim=0)

        self._resident_cluster_embeddings = stacked_embeddings
        self._resident_cluster_grid_thws = stacked_grid_thws
        self._resident_cluster_count = len(embeddings)
        self._resident_rep_versions = tuple(rep_versions)
        self._resident_retrieval_version = self._cluster_lifecycle_version
        if Config.enable_resident_retrieval_norm_cache:
            self._resident_cluster_embeddings_norm = F.normalize(
                stacked_embeddings, p=2, dim=-1
            )
        else:
            self._resident_cluster_embeddings_norm = None
        return (
            self._resident_cluster_embeddings,
            self._resident_cluster_grid_thws,
            self._resident_cluster_embeddings_norm,
        )
    
    # =========================================================================
    # Histogram Computation Methods
    # =========================================================================
    
    def compute_normalized_histogram(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute a normalized grayscale histogram for a frame.
        
        Args:
            frame: Input frame image in BGR format.
            
        Returns:
            Normalized histogram with 256 bins, values sum to 1.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray_frame.shape
        
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        normalized_hist = hist.flatten() / (width * height)
        
        return normalized_hist

    def bhattacharyya_distance(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Calculate Bhattacharyya distance between two histograms.
        
        Lower values indicate more similar histograms.
        
        Args:
            hist1: First normalized histogram.
            hist2: Second normalized histogram.
            
        Returns:
            Bhattacharyya distance. Returns infinity if coefficient is zero.
        """
        bhattacharyya_coeff = np.sum(np.sqrt(hist1 * hist2))
        
        if bhattacharyya_coeff == 0:
            return float('inf')
        
        distance = -np.log(bhattacharyya_coeff)
        return distance
    
    # =========================================================================
    # Cluster Management Methods
    # =========================================================================
    
    def _create_new_cluster(self, frame: Frame, histogram: np.ndarray) -> Cluster:
        """
        Create a new cluster with the given frame as its representative.
        
        This method creates a new cluster, sets the frame as representative,
        and computes the representative embedding using embed_fn.
        
        Args:
            frame: The first frame of the new cluster.
            histogram: Precomputed histogram of the frame.
            
        Returns:
            The newly created cluster with computed embedding.
        """
        cluster = Cluster(
            cluster_id=self._cluster_id_counter,
            embed_fn=self.embed_fn
        )
        self._cluster_id_counter += 1
        
        # Set histogram and add frame (this will compute embedding)
        cluster.representative_histogram = histogram
        cluster.add_frame(frame)
        self.ingest_encoded_frames += 1.0
        
        return cluster
    
    def _finalize_current_cluster(self) -> None:
        """Move the current cluster to the completed clusters list."""
        if self.current_cluster is not None:
            self._prepare_cluster_motion_selection(self.current_cluster)
            cluster = self.current_cluster
            self.clusters.append(cluster)
            self.current_cluster = None
            old_version = self._cluster_lifecycle_version
            self._cluster_lifecycle_version += 1
            if not self._try_append_resident_retrieval_cache(cluster, old_version):
                self._invalidate_resident_retrieval_cache()

    def _prepare_cluster_motion_selection(self, cluster: Cluster) -> None:
        """Compute and cache motion-selection metadata for a cluster when enabled."""
        if cluster.finalize_cpu_precomputed:
            return
        if not cluster.frames:
            cluster.motion_tensor = None
            cluster.temporal_motion_scores = []
            cluster.spatial_motion_map = None
            cluster.selected_frame_ids = []
            cluster.finalize_cpu_precomputed = True
            return

        need_motion_analysis = (
            Config.enable_retrieval_optimizations
            and (Config.enable_motion_frame_select or Config.enable_embedding_cache)
        )
        if not need_motion_analysis:
            cluster.finalize_cpu_precomputed = True
            return

        if cluster.motion_tensor is None:
            result = self.motion_analyzer.analyze(cluster.frames)
            cluster.motion_tensor = result.motion_tensor
            cluster.temporal_motion_scores = self._project_temporal_motion_scores(
                cluster.motion_tensor,
                num_frames=len(cluster.frames),
            )

        if Config.enable_motion_frame_select and cluster.selected_frame_ids is None:
            selected_indices = self.frame_selector.select(
                motion_scores=cluster.temporal_motion_scores or [],
                num_frames=len(cluster.frames),
                topk=Config.motion_select_topk,
                significant_threshold=Config.motion_significant_threshold,
                strategy=getattr(Config, "frame_select_strategy", "motion"),
            )
            cluster.selected_frame_ids = [cluster.frames[idx].frame_id for idx in selected_indices]
        if cluster.spatial_motion_map is None:
            cluster.spatial_motion_map = self._project_spatial_motion_map(cluster)
        cluster.finalize_cpu_precomputed = True

    @staticmethod
    def _project_temporal_motion_scores(
        motion_tensor: Optional[np.ndarray],
        num_frames: int,
    ) -> List[float]:
        if motion_tensor is None or motion_tensor.size == 0:
            return [0.0] * max(int(num_frames), 0)

        temporal_scores = motion_tensor.reshape(motion_tensor.shape[0], -1).mean(axis=1)
        scores = temporal_scores.astype(np.float32, copy=False).tolist()
        if len(scores) < num_frames:
            scores.extend([0.0] * (num_frames - len(scores)))
        return scores[:num_frames]

    @staticmethod
    def _project_spatial_motion_map(cluster: Cluster) -> Optional[np.ndarray]:
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

        if not selected_indices:
            return None

        selected_indices = [
            idx for idx in selected_indices
            if idx < int(motion_tensor.shape[0])
        ]
        if not selected_indices:
            return None

        selected_motion = motion_tensor[selected_indices]
        return np.mean(selected_motion, axis=0).astype(np.float32, copy=False)
    
    def _should_create_new_cluster(self, frame_histogram: np.ndarray) -> bool:
        """
        Determine if a new cluster should be created for the frame.
        
        Args:
            frame_histogram: Histogram of the frame to evaluate.
            
        Returns:
            True if a new cluster should be created, False otherwise.
        """
        if self.current_cluster is None:
            return True

        # Hard cap cluster length to keep each cluster bounded.
        if len(self.current_cluster) >= self.max_cluster_frames:
            return True

        distance = self.bhattacharyya_distance(
            frame_histogram,
            self.current_cluster.representative_histogram
        )
        
        return distance >= self.bhattacharyya_threshold
    
    # =========================================================================
    # Frame Processing Methods
    # =========================================================================
    
    def add_frame(self, frame_item: Frame) -> None:
        """
        Add a frame to the streaming memory.
        
        Frames are first added to a buffer. When the buffer is full,
        the oldest frame is processed and assigned to an appropriate cluster.
        
        Args:
            frame_item: The frame to add.
        """
        # Process oldest frame if buffer is full
        if len(self.buffer) >= self.buffer_size:
            self._process_buffered_frame()
        
        self.buffer.append(frame_item)
    
    def _process_buffered_frame(self) -> None:
        """Process the oldest frame in the buffer and assign to a cluster."""
        if not self.buffer:
            return
        
        frame = self.buffer.pop(0)
        
        if frame.image is None:
            raise ValueError(f"Frame {frame.frame_id} has no image data.")
        
        frame_histogram = self.compute_normalized_histogram(frame.image)
        
        if self._should_create_new_cluster(frame_histogram):
            # Finalize current cluster and create new one
            self._finalize_current_cluster()
            self.current_cluster = self._create_new_cluster(frame, frame_histogram)
        else:
            # Add to current cluster and do not update histogram
            self.current_cluster.add_frame(frame)
            # self.current_cluster.update_histogram(frame_histogram)
    
    # =========================================================================
    # Retrieval Methods
    # =========================================================================
    
    def _get_all_cluster_embeddings(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get resident stacked representative embeddings from completed clusters.

        Returns:
            Tuple of:
            - embeddings: (num_clusters, img_tokens, embedding_dim), or None.
            - grid_thws: (num_clusters, 3), or None.
            - normalized_embeddings: same shape as embeddings, or None.
        """
        if not self.clusters:
            return None, None, None

        if not Config.enable_resident_retrieval_cache:
            return self._rebuild_resident_retrieval_cache()

        rep_versions = tuple(
            cluster.representative_embedding_version for cluster in self.clusters
        )
        cache_valid = (
            self._resident_retrieval_version == self._cluster_lifecycle_version
            and self._resident_cluster_count == len(self.clusters)
            and self._resident_cluster_embeddings is not None
            and self._resident_cluster_grid_thws is not None
            and self._resident_rep_versions == rep_versions
        )
        if cache_valid:
            if (
                Config.enable_resident_retrieval_norm_cache
                and self._resident_cluster_embeddings_norm is None
            ):
                self._resident_cluster_embeddings_norm = F.normalize(
                    self._resident_cluster_embeddings, p=2, dim=-1
                )
            return (
                self._resident_cluster_embeddings,
                self._resident_cluster_grid_thws,
                self._resident_cluster_embeddings_norm,
            )
        return self._rebuild_resident_retrieval_cache()
    
    def retrieve_info(
        self,
        query_embed: Optional[torch.Tensor] = None,
        stream_id: Optional[str] = None,
    ) -> Tuple[List[ContextItem], int]:
        """
        Collect interleaved context with abstract memories and retrieved frames.
        
        This method implements the logic:
        1. Generate retrieval mask for clusters
        2. For continuous 0-segments in mask, update abstract memory with momentum
        3. Return interleaved list of ContextItem objects
        
        Args:
            query_embed: Query embedding for retrieval
        
        Returns:
            Tuple of:
                - context_items: List[ContextItem], each item is either:
                    - CLUSTER item with selected frames and cluster metadata
                    - FRAME item with frames list populated
                    - ABSTRACT_MEMORY item with retrieval embedding (abs_mem_emb),
                      LLM embedding (abs_mem_llm_emb), and grid_thw populated
                - history_frame_count: Number of retrieved history frames
        """
        self._finalize_current_cluster()

        context_items: List[ContextItem] = []
        history_frame_count = 0
        
        # Process history context with interleaving
        cluster_embeddings, _, cluster_embeddings_norm = self._get_all_cluster_embeddings()
        
        # Handle empty clusters case - skip history processing
        if cluster_embeddings is not None and len(self.clusters) > 0:
            mask = self.gen_mask(
                query_embed,
                cluster_embeddings,
                key_embeddings_normalized=cluster_embeddings_norm,
            )  # (num_clusters,)
            
            # Process clusters and create interleaved structure
            i = 0
            while i < len(self.clusters):
                if mask[i]:  # Retrieved cluster
                    cluster = self.clusters[i]
                    frames_to_use = self._get_cluster_frames_for_retrieval(cluster)
                    selected_frame_ids = [frame.frame_id for frame in frames_to_use]
                    if frames_to_use:
                        context_items.append(ContextItem(
                            kind=ContextItemKind.CLUSTER,
                            stream_id=stream_id,
                            cluster_id=cluster.cluster_id,
                            cluster=cluster,
                            selected_frame_ids=selected_frame_ids,
                            frames=frames_to_use,
                        ))
                        history_frame_count += len(frames_to_use)
                    i += 1
                else:  # Non-retrieved cluster - use momentum update for abstract memory
                    # Collect consecutive non-retrieved clusters
                    start_idx = i
                    j = i
                    while j < len(self.clusters) and not mask[j]:
                        j += 1
                    end_idx = j
                    
                    # Compute aggregated embedding for this segment using momentum update
                    aggregated_embedding = None
                    aggregated_llm_embedding = None
                    aggregated_grid_thw = None
                    
                    for k in range(start_idx, end_idx):
                        cluster = self.clusters[k]
                        cluster_emb = cluster.get_representative_embedding()  # (img_tokens, D)
                        if cluster_emb is None:
                            continue
                        cluster_llm_emb = cluster.get_representative_llm_embedding()
                        cluster_grid_thw = (
                            cluster.get_representative_grid_thw()
                            if cluster.representative_embedding
                            else None
                        )
                        
                        if aggregated_embedding is None:
                            aggregated_embedding = cluster_emb.clone()
                            aggregated_llm_embedding = cluster_llm_emb.clone()
                            aggregated_grid_thw = cluster_grid_thw
                        else:
                            # Momentum update: new = alpha * current + (1 - alpha) * accumulated
                            aggregated_embedding = self.alpha * cluster_emb + (1 - self.alpha) * aggregated_embedding
                            aggregated_llm_embedding = (
                                self.alpha * cluster_llm_emb
                                + (1 - self.alpha) * aggregated_llm_embedding
                            )

                    # Add abstract memory to context
                    if aggregated_embedding is not None:
                        context_items.append(ContextItem(
                            kind=ContextItemKind.ABSTRACT_MEMORY,
                            abs_mem_emb=aggregated_embedding,
                            abs_mem_llm_emb=aggregated_llm_embedding,
                            grid_thw=aggregated_grid_thw,
                        ))
                    
                    i = j
        
        # Add recent frames from buffer
        if self.buffer:
            recent_frame_objects = [frame for frame in self.buffer if frame.image is not None]
            if recent_frame_objects:
                context_items.append(ContextItem(
                    kind=ContextItemKind.FRAME,
                    frames=recent_frame_objects,
                ))
        
        return context_items, history_frame_count

    def _get_cluster_frames_for_retrieval(self, cluster: Cluster) -> List[Frame]:
        """Build frame list for retrieved cluster with optional motion selection."""
        fallback_frames = cluster.frames

        if not Config.enable_retrieval_optimizations or not Config.enable_motion_frame_select:
            return fallback_frames

        if not cluster.frames:
            return []

        self._prepare_cluster_motion_selection(cluster)
        selected_frames = cluster.get_selected_frames()
        if not selected_frames:
            selected_frames = fallback_frames
        cluster.selected_frame_ids = [frame.frame_id for frame in selected_frames]

        logger.debug(
            "Cluster %d frame selection: raw=%d selected=%d",
            cluster.cluster_id,
            len(cluster.frames),
            len(selected_frames),
        )
        return selected_frames

    def gen_mask(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        key_embeddings_normalized: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compare the Query embedding with the sequence embeddings and return a mask.

        Args:
            query_embeddings (S, D): The embedding of the query.
            key_embeddings (num_clusters, img_tokens, D): The embeddings of the key frames.
            key_embeddings_normalized: Optional pre-normalized key embeddings.

        Returns:
            mask (num_clusters,): A boolean mask indicating which key frames to select.
        """
        need_sync = key_embeddings.is_cuda and self._should_sync_cuda_in_gen_mask()
        if need_sync:
            torch.cuda.synchronize()

        N = key_embeddings.shape[0]
        k = min(Config.topk, N)

        # Normalize embeddings for stable attention computation
        q_norm = F.normalize(query_embeddings, p=2, dim=-1)  # (S, D)
        k_norm = key_embeddings_normalized
        if k_norm is None:
            k_norm = F.normalize(key_embeddings, p=2, dim=-1)    # (N, img_tokens, D)
        q_expanded = q_norm.unsqueeze(0)
        sim_matrix = torch.matmul(
            q_expanded.expand(N, -1, -1),
            k_norm.transpose(1, 2),
        )
        max_sim_per_query, _ = sim_matrix.max(dim=-1)
        frame_relevance = max_sim_per_query.mean(dim=1)

        similarities = frame_relevance

        if need_sync:
            torch.cuda.synchronize()

        # select top-k key frames
        _, topk_indices = torch.topk(similarities, k)
        mask = torch.zeros(N, dtype=torch.bool, device=key_embeddings.device)
        mask[topk_indices] = True
        return mask


    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_recent_frames(self) -> List[np.ndarray]:
        """
        Get raw frames from the buffer for batch processing.
        
        Returns:
            List of frame images (np.ndarray) from the buffer.
        
        Raises:
            ValueError: If the buffer is empty.
        """
        if not self.buffer:
            raise ValueError("Buffer is empty, cannot get recent frames.")
        return [frame.image for frame in self.buffer]

    def reset(self) -> None:
        """
        Reset the streaming memory to its initial state.
        
        Clears all buffered frames, clusters, and resets the cluster ID counter.
        The embed_fn and router are preserved.
        """
        self.buffer.clear()
        self.clusters.clear()
        self.current_cluster = None
        self._cluster_id_counter = 0
        self._cluster_lifecycle_version += 1
        self._invalidate_resident_retrieval_cache()
    
    def __repr__(self) -> str:
        self._finalize_current_cluster()
        cluster_frame_ids: List[List[int]] = [
            cluster.frame_ids for cluster in self.clusters
        ]
        buffer_frame_ids: List[int] = [frame.frame_id for frame in self.buffer]
        return (
            f"StreamingMemory("
            f"clusters={cluster_frame_ids}, "
            f"buffer={buffer_frame_ids})"
        )
