"""Dataset statistics: class distribution, boxes per image, relative box sizes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _describe(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def dataset_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute statistics over valid (finite, positive-size) boxes only."""
    coords = df[["x1", "y1", "x2", "y2", "image_width", "image_height"]].to_numpy(dtype=float)
    ok = np.isfinite(coords).all(axis=1)
    x1, y1, x2, y2, w, h = coords[ok].T
    valid = (x2 > x1) & (y2 > y1) & (w > 0) & (h > 0)
    good = df[ok].reset_index(drop=True)[valid]
    bw = ((x2 - x1) / w)[valid]
    bh = ((y2 - y1) / h)[valid]
    per_image = good.groupby("image_name").size().to_numpy(dtype=float)
    return {
        "n_images": int(df["image_name"].nunique()),
        "n_boxes_total": int(len(df)),
        "n_boxes_valid": int(valid.sum()),
        "class_counts": {str(k): int(v) for k, v in good["class"].value_counts().items()},
        "boxes_per_image": _describe(per_image),
        "box_width_rel": _describe(bw),
        "box_height_rel": _describe(bh),
        "box_area_rel": _describe(bw * bh),
    }


def write_stats(stats: dict[str, Any], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out
