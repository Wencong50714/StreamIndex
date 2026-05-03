from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ClusterMotionAnalysis:
    motion_volume: np.ndarray
    temporal_scores: np.ndarray
    temporal_threshold: float
    selected_frame_indices: List[int]
    spatial_score_map: np.ndarray
    spatial_threshold: float


def _ensure_odd_kernel_size(kernel_size: int) -> int:
    kernel_size = max(int(kernel_size), 1)
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def prepare_gray_image(image: Any, *, max_side: int = 64) -> np.ndarray:
    if image is None:
        raise ValueError("Frame image is None; cannot run cluster motion analysis.")

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported frame shape for motion analysis: {image.shape}")
    else:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for PIL image inputs.") from exc

        if isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            raise TypeError(f"Unsupported frame type for motion analysis: {type(image)!r}")

    gray = np.asarray(gray, dtype=np.float32)
    h, w = gray.shape[:2]
    max_side = max(int(max_side), 1)
    if max(h, w) > max_side:
        scale = float(max_side) / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32, copy=False)


def build_motion_volume(
    gray_frames: Sequence[np.ndarray],
    *,
    gaussian_kernel_size: int = 5,
    noise_threshold: float = 8.0,
) -> np.ndarray:
    if not gray_frames:
        return np.zeros((0, 0, 0), dtype=np.float32)

    base_h, base_w = gray_frames[0].shape[:2]
    normalized_frames: List[np.ndarray] = []
    for gray_frame in gray_frames:
        frame = np.asarray(gray_frame, dtype=np.float32)
        if frame.shape[:2] != (base_h, base_w):
            frame = cv2.resize(frame, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
        normalized_frames.append(frame)

    if len(normalized_frames) < 2:
        return np.zeros((0, base_h, base_w), dtype=np.float32)

    gaussian_kernel_size = _ensure_odd_kernel_size(gaussian_kernel_size)
    diffs: List[np.ndarray] = []
    for prev_frame, curr_frame in zip(normalized_frames, normalized_frames[1:]):
        diff = cv2.absdiff(prev_frame, curr_frame)
        if gaussian_kernel_size > 1:
            diff = cv2.GaussianBlur(diff, (gaussian_kernel_size, gaussian_kernel_size), 0)
        if noise_threshold > 0:
            diff = np.where(diff >= float(noise_threshold), diff, 0.0)
        diffs.append(diff.astype(np.float32, copy=False))

    if not diffs:
        return np.zeros((0, base_h, base_w), dtype=np.float32)
    return np.stack(diffs, axis=0)


def otsu_threshold_1d(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0

    min_val = float(np.min(flat))
    max_val = float(np.max(flat))
    if max_val <= min_val + 1e-6:
        return max_val

    scaled = np.clip(
        np.round((flat - min_val) * 255.0 / (max_val - min_val)),
        0,
        255,
    ).astype(np.uint8)
    threshold_uint8, _ = cv2.threshold(
        scaled.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return float(min_val + (threshold_uint8 / 255.0) * (max_val - min_val))


def compute_temporal_scores(motion_volume: np.ndarray, num_frames: int) -> np.ndarray:
    num_frames = max(int(num_frames), 0)
    if num_frames == 0:
        return np.zeros((0,), dtype=np.float32)

    temporal_scores = np.zeros((num_frames,), dtype=np.float32)
    if motion_volume.size == 0:
        return temporal_scores

    diff_scores = motion_volume.mean(axis=(1, 2)).astype(np.float32, copy=False)
    valid_count = min(int(diff_scores.shape[0]), max(num_frames - 1, 0))
    if valid_count > 0:
        temporal_scores[1 : valid_count + 1] = diff_scores[:valid_count]
    return temporal_scores


def select_keyframe_indices(
    temporal_scores: np.ndarray,
    *,
    min_frames: int = 1,
) -> Tuple[List[int], float]:
    num_frames = int(np.asarray(temporal_scores).size)
    if num_frames <= 0:
        return [], 0.0
    if num_frames == 1:
        return [0], 0.0

    projected_scores = np.asarray(temporal_scores, dtype=np.float32).reshape(-1)
    threshold = otsu_threshold_1d(projected_scores[1:])
    # Cluster frame selection always keeps the boundary frames.
    selected = {0, num_frames - 1}

    motion_frame_indices = np.flatnonzero(projected_scores[1:] > threshold) + 1
    selected.update(int(idx) for idx in motion_frame_indices.tolist())

    target_count = min(max(int(min_frames), 1), num_frames)
    ranked_candidates = np.argsort(projected_scores[1:], kind="stable")[::-1] + 1
    for idx in ranked_candidates.tolist():
        if len(selected) >= target_count:
            break
        selected.add(int(idx))

    return sorted(selected), float(threshold)


def compute_spatial_score_map(
    motion_volume: np.ndarray,
    selected_frame_indices: Sequence[int],
) -> np.ndarray:
    if motion_volume.ndim != 3 or motion_volume.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    slice_indices = [
        int(frame_idx - 1)
        for frame_idx in selected_frame_indices
        if int(frame_idx) > 0 and int(frame_idx - 1) < int(motion_volume.shape[0])
    ]
    if not slice_indices:
        return np.zeros(motion_volume.shape[1:], dtype=np.float32)

    return motion_volume[np.asarray(slice_indices, dtype=np.int64)].max(axis=0).astype(
        np.float32,
        copy=False,
    )


def resize_score_map_to_token_scores(
    spatial_score_map: np.ndarray,
    *,
    token_h: int,
    token_w: int,
    token_count: int,
) -> np.ndarray:
    token_count = max(int(token_count), 0)
    if token_count == 0:
        return np.zeros((0,), dtype=np.float32)

    token_h = max(int(token_h), 1)
    token_w = max(int(token_w), 1)
    if spatial_score_map.ndim != 2 or spatial_score_map.size == 0:
        resized_map = np.zeros((token_h, token_w), dtype=np.float32)
    else:
        resized_map = cv2.resize(
            spatial_score_map.astype(np.float32, copy=False),
            dsize=(token_w, token_h),
            interpolation=cv2.INTER_AREA,
        )

    token_scores = resized_map.reshape(-1).astype(np.float32, copy=False)
    if int(token_scores.size) == token_count:
        return token_scores

    remapped = cv2.resize(
        token_scores.reshape(1, -1),
        dsize=(token_count, 1),
        interpolation=cv2.INTER_AREA,
    )
    return remapped.reshape(-1).astype(np.float32, copy=False)


def build_token_hot_mask(
    spatial_score_map: np.ndarray,
    *,
    token_h: int,
    token_w: int,
    token_count: int,
) -> Tuple[np.ndarray, float]:
    token_scores = resize_score_map_to_token_scores(
        spatial_score_map,
        token_h=token_h,
        token_w=token_w,
        token_count=token_count,
    )
    if token_scores.size == 0 or float(np.max(token_scores)) <= 0.0:
        return np.zeros((int(token_scores.size),), dtype=bool), 0.0

    threshold = otsu_threshold_1d(token_scores)
    hot_mask = token_scores > threshold
    if not np.any(hot_mask):
        hot_mask[int(np.argmax(token_scores))] = True
    return hot_mask.astype(bool, copy=False), float(threshold)


def build_background_median(frames: Sequence[Any]) -> Optional[np.ndarray]:
    bgr_frames: List[np.ndarray] = []
    for frame in frames:
        image = getattr(frame, "image", frame)
        if image is None:
            continue
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 3:
                bgr = image.astype(np.uint8)
            elif image.ndim == 3 and image.shape[2] == 4:
                bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR).astype(np.uint8)
            else:
                continue
        else:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("Pillow is required for PIL image inputs.") from exc
            if not isinstance(image, Image.Image):
                continue
            rgb = np.asarray(image.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr_frames.append(bgr)

    if not bgr_frames:
        return None
    return np.median(np.stack(bgr_frames, axis=0).astype(np.float32), axis=0).astype(np.uint8)


def build_visualization_maps(spatial_score_map: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if spatial_score_map.ndim != 2 or spatial_score_map.size == 0:
        return None, None

    normalized = cv2.normalize(
        spatial_score_map.astype(np.float32, copy=False),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return heatmap, normalized


def analyze_motion_frames(
    frames: Sequence[Any],
    *,
    max_side: int = 64,
    gaussian_kernel_size: int = 5,
    noise_threshold: float = 8.0,
    min_frames: int = 1,
) -> ClusterMotionAnalysis:
    gray_frames = [
        prepare_gray_image(getattr(frame, "image", frame), max_side=max_side)
        for frame in frames
        if getattr(frame, "image", frame) is not None
    ]
    motion_volume = build_motion_volume(
        gray_frames,
        gaussian_kernel_size=gaussian_kernel_size,
        noise_threshold=noise_threshold,
    )
    temporal_scores = compute_temporal_scores(motion_volume, num_frames=len(gray_frames))
    selected_frame_indices, temporal_threshold = select_keyframe_indices(
        temporal_scores,
        min_frames=min_frames,
    )
    spatial_score_map = compute_spatial_score_map(motion_volume, selected_frame_indices)
    spatial_threshold = (
        otsu_threshold_1d(spatial_score_map.reshape(-1))
        if spatial_score_map.size > 0
        else 0.0
    )
    return ClusterMotionAnalysis(
        motion_volume=motion_volume,
        temporal_scores=temporal_scores,
        temporal_threshold=temporal_threshold,
        selected_frame_indices=selected_frame_indices,
        spatial_score_map=spatial_score_map,
        spatial_threshold=spatial_threshold,
    )
