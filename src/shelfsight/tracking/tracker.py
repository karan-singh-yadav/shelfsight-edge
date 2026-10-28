"""ByteTrack-style multi-object tracker (own implementation of the published algorithm).

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box",
ECCV 2022. This is a compact re-implementation (constant-velocity Kalman filter, IoU-based
Hungarian association, two-stage high/low-score matching); it is NOT a copy of the reference
code and has NOT been validated on MOT benchmarks. Matching is class-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from shelfsight.inference.base import Detections
from shelfsight.inference.postprocess import iou_matrix


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over (cx, cy, aspect, height) + velocities."""

    _WP, _WV = 1.0 / 20, 1.0 / 160

    def __init__(self) -> None:
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.eye(4, 8)

    @staticmethod
    def to_measurement(xyxy: np.ndarray) -> np.ndarray:
        w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        return np.array([xyxy[0] + w / 2, xyxy[1] + h / 2, w / max(h, 1e-6), h])

    def initiate(self, meas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = meas[3]
        mean = np.r_[meas, np.zeros(4)]
        std = [2 * self._WP * h, 2 * self._WP * h, 1e-2, 2 * self._WP * h,
               10 * self._WV * h, 10 * self._WV * h, 1e-5, 10 * self._WV * h]  # fmt: skip
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._WP * h, self._WP * h, 1e-2, self._WP * h,
               self._WV * h, self._WV * h, 1e-5, self._WV * h]  # fmt: skip
        return self.F @ mean, self.F @ cov @ self.F.T + np.diag(np.square(std))

    def update(
        self, mean: np.ndarray, cov: np.ndarray, meas: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        r = np.diag(np.square([self._WP * h, self._WP * h, 1e-1, self._WP * h]))
        s = self.H @ cov @ self.H.T + r
        gain = np.linalg.solve(s, self.H @ cov.T).T  # cov H^T S^-1
        new_mean = mean + gain @ (meas - self.H @ mean)
        new_cov = cov - gain @ s @ gain.T
        return new_mean, new_cov


def mean_to_xyxy(mean: np.ndarray) -> np.ndarray:
    cx, cy, a, h = mean[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class TrackState(Enum):
    TENTATIVE = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class Track:
    track_id: int
    mean: np.ndarray
    cov: np.ndarray
    box: np.ndarray
    score: float
    class_id: int
    start_frame: int
    last_frame: int
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    activated: bool = False

    @property
    def xyxy(self) -> np.ndarray:
        return mean_to_xyxy(self.mean)


@dataclass
class TrackedDetections:
    """Confirmed tracks matched in the current frame: detections + aligned track ids."""

    detections: Detections
    track_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def _associate(
    tracks: list[Track], boxes: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching on 1-IoU cost with gating. Returns (matches, unmatched_t, unmatched_d)."""
    if not tracks or len(boxes) == 0:
        return [], list(range(len(tracks))), list(range(len(boxes)))
    cost = 1.0 - iou_matrix(np.stack([t.xyxy for t in tracks]), boxes)
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= max_cost]
    mt, md = {r for r, _ in matches}, {c for _, c in matches}
    return (
        matches,
        [i for i in range(len(tracks)) if i not in mt],
        [j for j in range(len(boxes)) if j not in md],
    )


class ByteTracker:
    """Two-stage association tracker.

    Parameters mirror the paper: ``high_thresh`` splits detections into high/low score,
    ``match_thresh`` is the max (1-IoU) for the first association, ``track_buffer`` is how many
    frames a lost track is kept for re-identification.
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        new_track_thresh: float | None = None,
        match_thresh: float = 0.8,
        low_match_thresh: float = 0.5,
        unconfirmed_match_thresh: float = 0.7,
        track_buffer: int = 30,
    ) -> None:
        self.high_thresh, self.low_thresh = high_thresh, low_thresh
        self.new_track_thresh = high_thresh if new_track_thresh is None else new_track_thresh
        self.match_thresh = match_thresh
        self.low_match_thresh = low_match_thresh
        self.unconfirmed_match_thresh = unconfirmed_match_thresh
        self.track_buffer = track_buffer
        self.kf = KalmanBoxFilter()
        self.frame_id = 0
        self._next_id = 1
        self.tracked: list[Track] = []
        self.lost: list[Track] = []
        self.activated_ids: set[int] = set()

    @property
    def unique_tracks(self) -> int:
        """Number of distinct confirmed track IDs created so far (NOT unique products)."""
        return len(self.activated_ids)

    def reset(self) -> None:
        self.__init__(  # type: ignore[misc]
            self.high_thresh, self.low_thresh, self.new_track_thresh, self.match_thresh,
            self.low_match_thresh, self.unconfirmed_match_thresh, self.track_buffer,
        )  # fmt: skip

    def _new_track(self, box: np.ndarray, score: float, class_id: int) -> Track:
        mean, cov = self.kf.initiate(self.kf.to_measurement(box))
        t = Track(self._next_id, mean, cov, box, score, class_id, self.frame_id, self.frame_id)
        self._next_id += 1
        return t

    def _hit(self, track: Track, box: np.ndarray, score: float, class_id: int) -> None:
        track.mean, track.cov = self.kf.update(track.mean, track.cov, self.kf.to_measurement(box))
        track.box, track.score, track.class_id = box, score, class_id
        track.last_frame = self.frame_id
        track.hits += 1
        track.state = TrackState.TRACKED
        if not track.activated and (track.hits >= 2 or self.frame_id == 1):
            track.activated = True
            self.activated_ids.add(track.track_id)

    def update(self, det: Detections) -> TrackedDetections:
        """Advance one (processed) frame and return confirmed, currently-matched tracks."""
        self.frame_id += 1
        boxes, scores, cls = det.boxes, det.scores, det.class_ids
        high = np.flatnonzero(scores >= self.high_thresh)
        low = np.flatnonzero((scores >= self.low_thresh) & (scores < self.high_thresh))

        confirmed = [t for t in self.tracked if t.activated]
        unconfirmed = [t for t in self.tracked if not t.activated]
        pool = confirmed + self.lost
        for t in pool + unconfirmed:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)

        # Stage 1: confirmed + lost tracks vs high-score detections.
        m1, ut1, ud1 = _associate(pool, boxes[high], self.match_thresh)
        for ti, di in m1:
            k = high[di]
            self._hit(pool[ti], boxes[k], float(scores[k]), int(cls[k]))
        # Stage 2: still-unmatched tracked (not lost) vs low-score detections.
        rem = [pool[i] for i in ut1 if pool[i].state == TrackState.TRACKED]
        m2, ut2, _ = _associate(rem, boxes[low], self.low_match_thresh)
        for ti, di in m2:
            k = low[di]
            self._hit(rem[ti], boxes[k], float(scores[k]), int(cls[k]))
        for i in ut2:
            rem[i].state = TrackState.LOST
        # Unconfirmed tracks vs leftover high detections.
        left_high = high[ud1]
        m3, ut3, ud3 = _associate(unconfirmed, boxes[left_high], self.unconfirmed_match_thresh)
        for ti, di in m3:
            k = left_high[di]
            self._hit(unconfirmed[ti], boxes[k], float(scores[k]), int(cls[k]))
        for i in ut3:
            unconfirmed[i].state = TrackState.REMOVED
        # New tracks from leftover high detections.
        new_tracks = []
        for di in ud3:
            k = left_high[di]
            if scores[k] >= self.new_track_thresh:
                t = self._new_track(boxes[k], float(scores[k]), int(cls[k]))
                if self.frame_id == 1:
                    t.activated, t.state = True, TrackState.TRACKED
                    self.activated_ids.add(t.track_id)
                new_tracks.append(t)

        everyone = pool + unconfirmed + new_tracks
        self.tracked = [
            t for t in everyone if t.state in (TrackState.TRACKED, TrackState.TENTATIVE)
        ]
        self.lost = [
            t
            for t in everyone
            if t.state == TrackState.LOST and self.frame_id - t.last_frame <= self.track_buffer
        ]
        out = [t for t in self.tracked if t.activated and t.last_frame == self.frame_id]
        if not out:
            return TrackedDetections(Detections.empty(det.class_names))
        return TrackedDetections(
            Detections(
                np.stack([t.box for t in out]),
                np.array([t.score for t in out]),
                np.array([t.class_id for t in out]),
                det.class_names,
            ),
            np.array([t.track_id for t in out], dtype=np.int64),
        )
