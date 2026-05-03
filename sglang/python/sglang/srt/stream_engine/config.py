"""Configuration defaults for stream engine components."""


class Config:
    """Default runtime configuration used across stream engine modules."""

    # SGLang engine static GPU memory fraction. This controls the budget for
    # static allocations including the KV cache pool. ``None`` keeps the
    # upstream SGLang default / auto-tuned behavior.
    mem_fraction_static: float = 0.85 # for 32b model, set 0.85 otherwise 0.4

    # Global retrieval optimization switch
    enable_retrieval_optimizations: bool = True
    # Whether to include abstract-memory embeddings during context assembly.
    use_abs_mem: bool = True

    # Retrieval
    topk: int = 2
    # When enabled, keep a resident stacked representative-embedding cache.
    enable_resident_retrieval_cache: bool = True
    # Also keep a resident normalized key cache for gen_mask() hot path.
    enable_resident_retrieval_norm_cache: bool = True
    # Optional debug/profile sync points around retrieval similarity compute.
    # Keep disabled by default to avoid forced CPU/GPU serialization.
    enable_gen_mask_cuda_sync_debug: bool = False
    enable_gen_mask_cuda_sync_profile: bool = False

    # Streaming memory
    buffer_size: int = 4
    max_cluster_frames: int = 16
    bhattacharyya_threshold: float = 0.1
    alpha: float = 0.1

    enable_motion_frame_select: bool = True
    frame_select_strategy: str = "low_motion"
    motion_select_min_frames: int = 2
    motion_select_topk: int = 4
    motion_select_temporal_bins: int = 4
    motion_significant_threshold: float = 10.0
    motion_score_method: str = "mean_absdiff"

    # Cluster-level embedding cache
    enable_embedding_cache: bool = True
    # Stream-level CPU cache for whole-cluster precomputed embeddings.
    enable_cluster_cpu_cache: bool = True
    token_reuse_strategy: str = "cold"
    token_reuse_random_seed: int = 0
    cold_token_threshold: float = 0.08
    # Negative means "disabled" and falls back to threshold-based cold-token selection.
    cold_token_ratio: float = 0.8
    cache_stage_mode: str = "B"
    # LLaVA-OneVision sparse cache path. When enabled, LLaVA representative
    # frame embeddings use a spatial token grid and mask-approx encodes only
    # hot visual tokens through the SigLIP transformer when the local model
    # exposes the expected internals. Fallbacks keep correctness but are
    # reported separately in token stats.
    llava_ov_sparse_visual_encode: bool = True
    llava_ov_spatial_cache_grid: bool = True
    llava_ov_cache_grid_tokens: int = 196

    # Image resize
    min_pixels: int = 50176 # 224 * 224
    max_pixels: int = 151200 # 360 * 420
    enable_frame_resize: bool = True
