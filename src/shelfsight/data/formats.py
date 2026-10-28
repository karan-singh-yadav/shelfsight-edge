"""Annotation loading and box-format math (SKU-110K CSV <-> YOLO)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

SKU_COLUMNS = [
    "image_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "class",
    "image_width",
    "image_height",
]
NUMERIC_COLUMNS = ["x1", "y1", "x2", "y2", "image_width", "image_height"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetError(Exception):
    """Raised for problems that make a dataset unusable (not for per-row quality issues)."""


def load_annotations(csv_path: str | Path) -> pd.DataFrame:
    """Load a SKU-110K-style annotation CSV.

    Expected columns: image_name, x1, y1, x2, y2, class, image_width, image_height.
    The released SKU-110K CSVs are understood to be headerless with this column order; this
    loader also accepts a header row. Non-numeric cells become NaN so the validator can
    report them instead of the loader crashing.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise DatasetError(f"Annotation file not found: {path}")
    with open(path, encoding="utf-8", newline="") as fh:
        first = fh.readline().strip()
    if not first:
        raise DatasetError(f"Annotation file is empty: {path}")
    has_header = first.split(",")[0].strip().lower() == "image_name"
    df = pd.read_csv(path, header=0 if has_header else None, dtype=str, keep_default_na=False)
    if has_header:
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = [c for c in SKU_COLUMNS if c not in df.columns]
        if missing:
            raise DatasetError(f"{path} is missing columns: {missing}")
        df = df[SKU_COLUMNS].copy()
    else:
        if df.shape[1] != len(SKU_COLUMNS):
            raise DatasetError(
                f"{path}: expected {len(SKU_COLUMNS)} columns {SKU_COLUMNS}, got {df.shape[1]}"
            )
        df.columns = SKU_COLUMNS
    df["image_name"] = df["image_name"].str.strip()
    df["class"] = df["class"].str.strip()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def select_subset(df: pd.DataFrame, n_images: int, seed: int = 42) -> pd.DataFrame:
    """Deterministically select ``n_images`` distinct images (all their boxes)."""
    import numpy as np

    names = np.array(sorted(df["image_name"].unique()))
    if n_images >= len(names):
        return df
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(names, size=n_images, replace=False).tolist())
    return df[df["image_name"].isin(chosen)].reset_index(drop=True)


def clip_box(
    x1: float, y1: float, x2: float, y2: float, width: float, height: float
) -> tuple[float, float, float, float]:
    """Clip a pixel-space xyxy box to the image rectangle."""
    return (
        min(max(x1, 0.0), width),
        min(max(y1, 0.0), height),
        min(max(x2, 0.0), width),
        min(max(y2, 0.0), height),
    )


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, width: float, height: float
) -> tuple[float, float, float, float]:
    """Pixel xyxy -> normalized (x_center, y_center, w, h). Does not clip."""
    if width <= 0 or height <= 0:
        raise ValueError("image width/height must be positive")
    return (
        (x1 + x2) / 2.0 / width,
        (y1 + y2) / 2.0 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def yolo_to_xyxy(
    xc: float, yc: float, bw: float, bh: float, width: float, height: float
) -> tuple[float, float, float, float]:
    """Normalized YOLO box -> pixel xyxy."""
    return (
        (xc - bw / 2.0) * width,
        (yc - bh / 2.0) * height,
        (xc + bw / 2.0) * width,
        (yc + bh / 2.0) * height,
    )


def is_finite_box(*vals: float) -> bool:
    return all(math.isfinite(v) for v in vals)
