"""Counting semantics.

Three different quantities are tracked and must NOT be treated as interchangeable:

1. **Detections per frame**: how many boxes the detector emits in one frame (a snapshot).
2. **Unique tracks**: how many distinct confirmed track IDs the tracker created. ID switches
   and fragmentation inflate it; missed associations split one product into several tracks.
3. **Estimated unique products passing through a scene**: here estimated by a virtual-line
   crossing counter (each track ID counted once when its centre crosses the line). It is only
   meaningful for flow scenarios (e.g. a conveyor or a basket moving past a camera), and is an
   estimate, not a verified count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CountingSummary:
    frames: int
    detections_total: int
    mean_detections_per_frame: float
    max_detections_in_frame: int
    unique_tracks: int
    line_crossings: int | None = None


@dataclass
class FrameCounter:
    """Accumulates per-frame detection counts and unique track IDs."""

    per_frame: list[int] = field(default_factory=list)
    track_ids: set[int] = field(default_factory=set)

    def update(self, n_detections: int, ids: np.ndarray | None = None) -> None:
        self.per_frame.append(int(n_detections))
        if ids is not None:
            self.track_ids.update(int(i) for i in ids)

    def summary(self, line_crossings: int | None = None) -> CountingSummary:
        n = len(self.per_frame)
        return CountingSummary(
            frames=n,
            detections_total=int(sum(self.per_frame)),
            mean_detections_per_frame=float(np.mean(self.per_frame)) if n else 0.0,
            max_detections_in_frame=max(self.per_frame, default=0),
            unique_tracks=len(self.track_ids),
            line_crossings=line_crossings,
        )


class LineCrossingCounter:
    """Counts each track ID once when its box centre crosses a virtual line.

    The line is given by two points in normalized [0,1] image coordinates. Crossing is
    detected by a sign change of the centre's side relative to the infinite line through the
    points, so place the line to span the whole flow path.
    """

    def __init__(self, p1: tuple[float, float], p2: tuple[float, float]) -> None:
        if p1 == p2:
            raise ValueError("line endpoints must differ")
        self.p1, self.p2 = p1, p2
        self._last_side: dict[int, float] = {}
        self.crossed: set[int] = set()

    def _side(self, cx: float, cy: float) -> float:
        (x1, y1), (x2, y2) = self.p1, self.p2
        return (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)

    def update(self, boxes_xyxy: np.ndarray, ids: np.ndarray, frame_shape: tuple[int, int]) -> int:
        """Update with this frame's tracked boxes; returns total distinct IDs crossed so far."""
        h, w = frame_shape
        for box, tid in zip(boxes_xyxy, ids, strict=True):
            cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
            side = self._side(cx, cy)
            prev = self._last_side.get(int(tid))
            if prev is not None and prev * side < 0:
                self.crossed.add(int(tid))
            if side != 0:
                self._last_side[int(tid)] = side
        return len(self.crossed)


def count_mae(predicted: list[int], ground_truth: list[int]) -> float:
    """Mean absolute error between per-image predicted and ground-truth object counts."""
    if len(predicted) != len(ground_truth):
        raise ValueError("predicted and ground_truth must have the same length")
    if not predicted:
        raise ValueError("need at least one image")
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(ground_truth))))
