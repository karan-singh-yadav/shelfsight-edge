"""Rendering helpers: detections, tracks and shelf regions on frames."""

from __future__ import annotations

import cv2
import numpy as np

from shelfsight.inference.base import Detections

_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
]  # fmt: skip


def color_for(idx: int) -> tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


def draw_detections(
    image: np.ndarray,
    det: Detections,
    track_ids: np.ndarray | None = None,
    show_conf: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Return an annotated copy. ``track_ids`` (len == len(det)) adds ``#id`` labels."""
    out = image.copy()
    for i in range(len(det)):
        x1, y1, x2, y2 = (round(float(v)) for v in det.boxes[i])
        cid = int(det.class_ids[i])
        key = int(track_ids[i]) if track_ids is not None else cid
        color = color_for(key)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        parts = []
        if track_ids is not None:
            parts.append(f"#{int(track_ids[i])}")
        if show_conf or len(det.class_names) > 1:
            parts.append(det.name_of(cid))
        if show_conf:
            parts.append(f"{float(det.scores[i]):.2f}")
        label = " ".join(parts)
        if label and (x2 - x1) > 30:
            cv2.putText(
                out, label, (x1 + 1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
    return out


def draw_regions(
    image: np.ndarray,
    regions_px: list[tuple[str, tuple[int, int, int, int]]],
    low_flags: dict[str, bool] | None = None,
    labels: dict[str, str] | None = None,
) -> np.ndarray:
    """Draw named rectangular regions; low-occupancy regions are drawn in red."""
    out = image.copy()
    for name, (x1, y1, x2, y2) in regions_px:
        low = bool(low_flags and low_flags.get(name))
        color = (0, 0, 255) if low else (255, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = (labels or {}).get(name, name)
        cv2.putText(out, text, (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out
