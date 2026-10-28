"""Draw ground-truth boxes for visual inspection of annotations."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def draw_boxes(
    image: np.ndarray, boxes_xyxy: np.ndarray, color: tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Return a copy of ``image`` with pixel-space xyxy boxes drawn."""
    out = image.copy()
    for x1, y1, x2, y2 in np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4):
        if not np.isfinite([x1, y1, x2, y2]).all():
            continue
        cv2.rectangle(out, (round(x1), round(y1)), (round(x2), round(y2)), color, 1)
    return out


def render_samples(
    df: pd.DataFrame, image_dir: Path, out_dir: Path, n: int = 8, seed: int = 42
) -> list[Path]:
    """Render ``n`` randomly chosen annotated images into ``out_dir``; returns written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(df["image_name"].unique())
    rng = np.random.default_rng(seed)
    picks = rng.choice(names, size=min(n, len(names)), replace=False)
    written: list[Path] = []
    for name in picks:
        img = cv2.imread(str(image_dir / str(name)))
        if img is None:
            logger.warning("cannot read %s, skipped", name)
            continue
        boxes = df.loc[df["image_name"] == name, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        vis = draw_boxes(img, boxes)
        cv2.putText(
            vis, f"{len(boxes)} boxes", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        path = out_dir / f"{Path(str(name)).stem}_gt.jpg"
        cv2.imwrite(str(path), vis)
        written.append(path)
    return written
