from __future__ import annotations

import random
from typing import List


class MotionFrameSelector:
    """Select key frames using adjacent-frame motion scores."""

    def __init__(
        self,
        min_frames: int = 2,
        temporal_bins: int = 4,
        significant_threshold: float = 8.0,
    ) -> None:
        self.min_frames = min_frames
        self.temporal_bins = temporal_bins
        self.significant_threshold = significant_threshold

    def select(
        self,
        motion_scores: List[float],
        num_frames: int,
        topk: int | None = None,
        significant_threshold: float | None = None,
        strategy: str = "motion",
    ) -> List[int]:
        if num_frames <= 0:
            return []
        if num_frames == 1:
            return [0]

        target_topk = num_frames if topk is None else max(2, int(topk))
        target_topk = min(target_topk, num_frames)
        if target_topk >= num_frames:
            return list(range(num_frames))

        normalized_strategy = str(strategy or "motion").strip().lower()
        if normalized_strategy == "uniform":
            return self._select_uniform(num_frames=num_frames, topk=target_topk)
        if normalized_strategy == "random":
            return self._select_random(num_frames=num_frames, topk=target_topk)
        if normalized_strategy == "low_motion":
            return self._select_low_motion(
                motion_scores=motion_scores,
                num_frames=num_frames,
                topk=target_topk,
            )
        if normalized_strategy == "hybrid":
            return self._select_hybrid(
                motion_scores=motion_scores,
                num_frames=num_frames,
                topk=target_topk,
            )
        if normalized_strategy != "motion":
            raise ValueError(
                f"Unsupported frame selection strategy: {strategy}. "
                "Expected one of: motion, uniform, random, low_motion, hybrid."
            )

        selected = set()
        if target_topk <= len(selected):
            return sorted(selected)

        threshold = (
            self.significant_threshold
            if significant_threshold is None
            else significant_threshold
        )
        candidate_indices = list(range(1, num_frames - 1))
        candidate_indices.sort(
            key=lambda idx: (
                self._score_at(motion_scores, idx) >= threshold,
                self._score_at(motion_scores, idx),
                -idx,
            ),
            reverse=True,
        )

        remaining_slots = max(0, target_topk - len(selected))
        selected.update(candidate_indices[:remaining_slots])

        min_needed = min(max(self.min_frames, target_topk), num_frames)
        if len(selected) < min_needed:
            remaining = [idx for idx in range(num_frames) if idx not in selected]
            selected.update(remaining[: min_needed - len(selected)])
        return sorted(selected)

    def _select_uniform(self, num_frames: int, topk: int) -> List[int]:
        if topk <= 2:
            return [0, num_frames - 1]

        denom = max(topk - 1, 1)
        selected = {
            int(round(i * (num_frames - 1) / denom))
            for i in range(topk)
        }
        return self._dedupe_and_fill(selected, num_frames=num_frames, topk=topk)

    def _select_random(self, num_frames: int, topk: int) -> List[int]:
        selected = {0, num_frames - 1}
        remaining_slots = max(0, topk - len(selected))
        candidates = list(range(1, num_frames - 1))
        if remaining_slots > 0:
            selected.update(
                random.sample(candidates, k=min(remaining_slots, len(candidates)))
            )
        return self._dedupe_and_fill(selected, num_frames=num_frames, topk=topk)

    def _select_low_motion(
        self,
        motion_scores: List[float],
        num_frames: int,
        topk: int,
    ) -> List[int]:
        selected = set()
        if topk <= len(selected):
            return sorted(selected)

        candidate_indices = list(range(1, num_frames - 1))
        candidate_indices.sort(
            key=lambda idx: (self._score_at(motion_scores, idx), idx)
        )
        remaining_slots = max(0, topk - len(selected))
        selected.update(candidate_indices[:remaining_slots])
        return self._dedupe_and_fill(selected, num_frames=num_frames, topk=topk)

    def _select_hybrid(
        self,
        motion_scores: List[float],
        num_frames: int,
        topk: int,
    ) -> List[int]:
        selected = {0, num_frames - 1}
        if topk <= len(selected):
            return sorted(selected)

        interior_indices = list(range(1, num_frames - 1))
        high_motion_order = sorted(
            interior_indices,
            key=lambda idx: (self._score_at(motion_scores, idx), -idx),
            reverse=True,
        )
        low_motion_order = sorted(
            interior_indices,
            key=lambda idx: (self._score_at(motion_scores, idx), idx),
        )

        for order in (high_motion_order, low_motion_order):
            for idx in order:
                selected.add(idx)
                if len(selected) >= topk:
                    return sorted(selected)
                break

        high_pos = 0
        low_pos = 0
        use_high = True
        while len(selected) < topk and (high_pos < len(high_motion_order) or low_pos < len(low_motion_order)):
            order = high_motion_order if use_high else low_motion_order
            pos = high_pos if use_high else low_pos
            while pos < len(order) and order[pos] in selected:
                pos += 1
            if pos < len(order):
                selected.add(order[pos])
                pos += 1
            if use_high:
                high_pos = pos
            else:
                low_pos = pos
            use_high = not use_high

        return self._dedupe_and_fill(selected, num_frames=num_frames, topk=topk)

    @staticmethod
    def _dedupe_and_fill(selected: set[int], num_frames: int, topk: int) -> List[int]:
        if len(selected) < topk:
            for idx in range(num_frames):
                selected.add(idx)
                if len(selected) >= topk:
                    break
        return sorted(selected)

    @staticmethod
    def _score_at(
        motion_scores: List[float],
        idx: int,
    ) -> float:
        return motion_scores[idx] if idx < len(motion_scores) else 0.0
