from __future__ import annotations

from enum import Enum
from typing import List, Union, Optional, Tuple, Callable
from dataclasses import dataclass, field

from PIL import Image
import torch
import numpy as np

QwenEmbedding = Tuple[torch.Tensor, torch.Tensor]
FrameEmbeddingResult = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
EmbedFunction = Callable[..., FrameEmbeddingResult]

@dataclass
class Frame:
    """
    Represents a single video frame.
    
    Attributes:
        frame_id: Unique identifier for the frame (typically the frame index).
        image: The actual image data, can be PIL Image or numpy array.
    """
    frame_id: int
    image: Optional[Union[Image.Image, np.ndarray]] = None
    
    def __repr__(self) -> str:
        has_image = self.image is not None
        return f"Frame(id={self.frame_id}, has_image={has_image})"


@dataclass
class Cluster:
    """
    A cluster of visually similar frames.
    
    Clusters group consecutive frames that share similar visual characteristics
    (measured by Bhattacharyya distance on histograms). Each cluster maintains
    a representative embedding computed from its first frame.
    
    Attributes:
        cluster_id: Unique identifier for the cluster.
        embed_fn: Function to compute embeddings from frame images.
        frame_ids: List of frame IDs belonging to this cluster.
        representative_frame: The first frame added, used as the representative.
        representative_histogram: Normalized histogram for similarity comparison.
        representative_embedding: Precomputed embedding of the representative frame.
    """
    cluster_id: int
    embed_fn: Optional[EmbedFunction] = None
    frame_ids: List[int] = field(default_factory=list)
    frames: List[Frame] = field(default_factory=list)
    
    representative_id: int = -1
    representative_frame: Optional[Frame] = None
    representative_histogram: Optional[np.ndarray] = None
    representative_embedding: Optional[FrameEmbeddingResult] = None
    motion_tensor: Optional[np.ndarray] = None
    temporal_motion_scores: Optional[List[float]] = None
    spatial_motion_map: Optional[np.ndarray] = None
    selected_frame_ids: Optional[List[int]] = None
    finalize_cpu_precomputed: bool = False
    representative_embedding_version: int = 0
    
    def add_frame(self, frame: Frame) -> None:
        """
        Add a frame to this cluster.
        
        If this is the first frame, it becomes the representative frame
        and its embedding is computed using embed_fn.
        
        Args:
            frame: The frame to add to the cluster.
        """
        self.frame_ids.append(frame.frame_id)
        self.frames.append(frame)
        self.finalize_cpu_precomputed = False
        # Any new frame invalidates previous motion/selection-derived metadata.
        self.motion_tensor = None
        self.temporal_motion_scores = None
        self.spatial_motion_map = None
        self.selected_frame_ids = None
        # First frame becomes the representative
        if self.representative_frame is None:
            self.representative_id = frame.frame_id
            self.representative_frame = frame
            self._compute_representative_embedding()
    
    def _compute_representative_embedding(self) -> None:
        """Compute and cache the embedding of the representative frame."""
        if self.representative_frame is None:
            return
        
        if self.embed_fn is None:
            raise ValueError("embed_fn is not set, cannot compute embedding.")
        
        if self.representative_frame.image is None:
            raise ValueError("Representative frame has no image data.")

        try:
            from sglang.srt.stream_engine.config import Config

            use_cache_grid = bool(getattr(Config, "enable_embedding_cache", True))
        except Exception:
            use_cache_grid = False

        try:
            self.representative_embedding = self.embed_fn(
                self.representative_frame.image,
                cache_grid=use_cache_grid,
            )
        except TypeError:
            self.representative_embedding = self.embed_fn(self.representative_frame.image)
        self.representative_embedding_version += 1
    
    def get_representative_embedding(self) -> torch.Tensor:
        """
        Get the representative embedding used for retrieval.
        
        Returns:
            The precomputed retrieval embedding tensor of the representative frame.
            
        Raises:
            ValueError: If no representative embedding exists.
        """
        if self.representative_embedding is None:
            raise ValueError("Cluster has no representative embedding.")
        return self.representative_embedding[2]

    def get_representative_llm_embedding(self) -> torch.Tensor:
        """Get the representative embedding used by LLM precomputed path."""
        if self.representative_embedding is None:
            raise ValueError("Cluster has no representative embedding.")
        return self.representative_embedding[0]

    def get_representative_grid_thw(self) -> torch.Tensor:
        """Get the representative frame grid_thw."""
        if self.representative_embedding is None:
            raise ValueError("Cluster has no representative embedding.")
        return self.representative_embedding[1]
    
    def get_cluster_context(self) -> QwenEmbedding:
        """
        Return a bounded cluster context for callers outside the query path.

        When explicit selection metadata exists, use it. Otherwise fall back to
        the representative frame rather than silently expanding to every frame
        in a long cluster.
        
        Returns:
            embed: (num_frames, img_tokens, embedding_dim), for LLM path.
            grid_thw: (num_frames, 3)
        """
        if self.embed_fn is None:
            raise ValueError("embed_fn is not set, cannot compute embeddings.")

        if self.selected_frame_ids:
            frames_to_embed = self.get_selected_frames()
        elif self.representative_frame is not None:
            frames_to_embed = [self.representative_frame]
        else:
            frames_to_embed = self.frames

        embeddings = []
        grid_thws = []
        for frame in frames_to_embed:

            if frame.frame_id == self.representative_id:
                embeddings.append(self.representative_embedding[0])
                grid_thws.append(self.representative_embedding[1])
                continue
            embedding = self.embed_fn(frame.image)
            embeddings.append(embedding[0])
            grid_thws.append(embedding[1])
        
        return torch.stack(embeddings, dim=0), torch.stack(grid_thws, dim=0)

    def get_selected_frames(self) -> List[Frame]:
        """
        Return selected frames when available, otherwise return all frames.

        Selection is represented by frame IDs to keep stable identity when
        frame list objects are copied/reordered by callers.
        """
        if not self.selected_frame_ids:
            return self.frames
        selected_id_set = set(self.selected_frame_ids)
        return [frame for frame in self.frames if frame.frame_id in selected_id_set]
        
    
    def update_histogram(self, new_histogram: np.ndarray) -> None:
        """
        Update the representative histogram using moving average.
        
        Args:
            new_histogram: The histogram of a newly added frame.
        """
        if self.representative_histogram is None:
            self.representative_histogram = new_histogram
        else:
            n = len(self.frame_ids)
            self.representative_histogram = (
                self.representative_histogram * (n - 1) + new_histogram
            ) / n
    
    def get_frame_ids(self) -> List[int]:
        """
        Get all frame IDs in this cluster.
        
        Returns:
            List of frame IDs belonging to this cluster.
        """
        return self.frame_ids.copy()
    
    def __len__(self) -> int:
        """Return the number of frames in this cluster."""
        return len(self.frame_ids)
    
    def __repr__(self) -> str:
        return f"Cluster(id={self.cluster_id}, num_frames={len(self)})"


class ContextItemKind(str, Enum):
    ABSTRACT_MEMORY = "abstract_memory"
    FRAME = "frame"
    CLUSTER = "cluster"


@dataclass
class ContextItem:
    kind: ContextItemKind
    stream_id: Optional[str] = None
    cluster_id: Optional[int] = None
    cluster: Optional[Cluster] = None
    selected_frame_ids: Optional[List[int]] = None
    frames: Optional[List[Frame]] = None
    abs_mem_emb: Optional[torch.Tensor] = None
    abs_mem_llm_emb: Optional[torch.Tensor] = None
    grid_thw: Optional[torch.Tensor] = None
