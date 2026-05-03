from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from sglang.srt.stream_engine.types import Frame


@dataclass
class MotionAnalysisResult:
    motion_tensor: Optional[np.ndarray]


class CPUMotionAnalyzer:
    """Analyze inter-frame motion on CPU using grayscale abs-diff."""

    def __init__(
        self,
        downsample_scale: float = 0.25,
        blur_kernel_size: int = 3,
        threshold: float = 5.0,
        score_method: str = "mean_absdiff",
    ) -> None:
        self.downsample_scale = downsample_scale
        self.blur_kernel_size = blur_kernel_size
        self.threshold = threshold
        self.score_method = score_method

    def analyze(self, frames: List[Frame]) -> MotionAnalysisResult:
        if not frames:
            return MotionAnalysisResult(motion_tensor=None)

        first_gray = self._prepare_gray(frames[0].image)
        motion_slices: List[np.ndarray] = [
            np.zeros(first_gray.shape, dtype=np.float32)
        ]

        prev_gray = first_gray
        for frame in frames[1:]:
            curr_gray = self._prepare_gray(frame.image)
            diff = cv2.absdiff(curr_gray, prev_gray)
            diff = self._post_process(diff)
            motion_slices.append(diff.astype(np.float32, copy=False))
            prev_gray = curr_gray

        motion_tensor = np.stack(motion_slices, axis=0).astype(
            np.float32, copy=False
        )
        return MotionAnalysisResult(motion_tensor=motion_tensor)

    def _prepare_gray(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Frame image is None; cannot run motion analysis.")

        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported frame shape for motion analysis: {image.shape}")

        if self.downsample_scale > 0 and self.downsample_scale < 1.0:
            gray = cv2.resize(
                gray,
                dsize=None,
                fx=self.downsample_scale,
                fy=self.downsample_scale,
                interpolation=cv2.INTER_AREA,
            )

        return gray.astype(np.float32)

    def _post_process(self, diff: np.ndarray) -> np.ndarray:
        if self.blur_kernel_size and self.blur_kernel_size > 1:
            k = int(self.blur_kernel_size)
            if k % 2 == 0:
                k += 1
            diff = cv2.GaussianBlur(diff, (k, k), sigmaX=0)

        if self.threshold and self.threshold > 0:
            diff = np.where(diff >= self.threshold, diff, 0.0)

        return diff
