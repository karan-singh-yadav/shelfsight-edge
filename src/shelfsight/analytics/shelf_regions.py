"""Rectangular shelf regions, detection assignment and occupancy estimation.

Occupancy = fraction of a region's area covered by the UNION of assigned detection boxes.
It is an estimate of visible product coverage, not a stock quantity: occlusion, perspective,
product overlap, tall/short packaging and camera movement all bias it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from shelfsight.inference.base import Detections

_MAX_MASK_SIDE = 256


class RegionError(ValueError):
    """Invalid shelf-region definition."""


@dataclass(frozen=True)
class ShelfRegion:
    """Rectangle in NORMALIZED coordinates (0..1), so it is resolution independent."""

    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    low_threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise RegionError("region name must be non-empty")
        if not (0 <= self.x1 < self.x2 <= 1 and 0 <= self.y1 < self.y2 <= 1):
            raise RegionError(f"region '{self.name}' must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1")
        if self.low_threshold is not None and not 0 <= self.low_threshold <= 1:
            raise RegionError(f"region '{self.name}' low_threshold must be in [0,1]")

    def to_pixels(self, shape: tuple[int, int]) -> tuple[int, int, int, int]:
        h, w = shape
        return (round(self.x1 * w), round(self.y1 * h), round(self.x2 * w), round(self.y2 * h))


@dataclass
class RegionStats:
    name: str
    count: int
    occupancy_ratio: float
    low_occupancy: bool
    class_counts: dict[str, int] = field(default_factory=dict)


def validate_regions(regions: list[ShelfRegion]) -> None:
    names = [r.name for r in regions]
    if len(names) != len(set(names)):
        raise RegionError("region names must be unique")


def load_regions(path: str | Path) -> list[ShelfRegion]:
    """Load regions from YAML/JSON: a list (or ``{regions: [...]}``) of dicts with
    name, x1, y1, x2, y2 and optional low_threshold."""
    p = Path(path)
    if not p.is_file():
        raise RegionError(f"Region file not found: {p}")
    text = p.read_text(encoding="utf-8")
    data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    if isinstance(data, dict):
        data = data.get("regions")
    if not isinstance(data, list) or not data:
        raise RegionError("region file must contain a non-empty list of regions")
    try:
        regions = [ShelfRegion(**item) for item in data]
    except TypeError as exc:
        raise RegionError(f"invalid region entry: {exc}") from exc
    validate_regions(regions)
    return regions


def _union_ratio(boxes_px: np.ndarray, region_px: tuple[int, int, int, int]) -> float:
    rx1, ry1, rx2, ry2 = region_px
    rw, rh = rx2 - rx1, ry2 - ry1
    if rw <= 0 or rh <= 0 or len(boxes_px) == 0:
        return 0.0
    s = min(1.0, _MAX_MASK_SIDE / max(rw, rh))
    mw, mh = max(1, round(rw * s)), max(1, round(rh * s))
    mask = np.zeros((mh, mw), dtype=bool)
    for x1, y1, x2, y2 in boxes_px:
        a = int(np.floor((max(x1, rx1) - rx1) * s))
        b = int(np.floor((max(y1, ry1) - ry1) * s))
        c = int(np.ceil((min(x2, rx2) - rx1) * s))
        d = int(np.ceil((min(y2, ry2) - ry1) * s))
        if c > a and d > b:
            mask[max(b, 0) : min(d, mh), max(a, 0) : min(c, mw)] = True
    return float(mask.mean())


def analyze_regions(
    det: Detections,
    regions: list[ShelfRegion],
    frame_shape: tuple[int, int],
    default_low_threshold: float = 0.2,
) -> list[RegionStats]:
    """Assign each detection (by box centre) to the FIRST region containing it and compute
    per-region counts and approximate occupancy. Detections outside all regions are ignored."""
    validate_regions(regions)
    px = [r.to_pixels(frame_shape) for r in regions]
    owner = np.full(len(det), -1, dtype=np.int64)
    if len(det):
        cx = (det.boxes[:, 0] + det.boxes[:, 2]) / 2
        cy = (det.boxes[:, 1] + det.boxes[:, 3]) / 2
        for idx in range(len(regions) - 1, -1, -1):  # reverse so earlier regions win
            x1, y1, x2, y2 = px[idx]
            owner[(cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)] = idx
    stats = []
    for idx, region in enumerate(regions):
        sel = np.flatnonzero(owner == idx)
        sub = Detections(det.boxes[sel], det.scores[sel], det.class_ids[sel], det.class_names)
        ratio = _union_ratio(det.boxes[sel], px[idx])
        thr = default_low_threshold if region.low_threshold is None else region.low_threshold
        stats.append(RegionStats(region.name, len(sel), ratio, ratio < thr, sub.counts()))
    return stats
