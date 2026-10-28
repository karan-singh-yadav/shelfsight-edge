"""Dataset integrity and bounding-box validation.

Severity model: ``error`` = would corrupt training/evaluation if ignored; ``warning`` =
recoverable (e.g. slightly out-of-bounds boxes that conversion clips and reports).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from shelfsight.data.formats import load_annotations

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" | "warning"
    code: str
    image: str
    detail: str


@dataclass
class ValidationReport:
    n_images: int = 0
    n_boxes: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    severities: dict[str, str] = field(default_factory=dict)
    examples: list[Issue] = field(default_factory=list)
    max_examples_per_code: int = 20

    def add(self, severity: str, code: str, image: str, detail: str) -> None:
        self.counts[code] = self.counts.get(code, 0) + 1
        self.severities[code] = severity
        if sum(1 for i in self.examples if i.code == code) < self.max_examples_per_code:
            self.examples.append(Issue(severity, code, image, detail))

    @property
    def n_errors(self) -> int:
        return sum(n for c, n in self.counts.items() if self.severities[c] == "error")

    @property
    def n_warnings(self) -> int:
        return sum(n for c, n in self.counts.items() if self.severities[c] == "warning")

    @property
    def ok(self) -> bool:
        return self.n_errors == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_images": self.n_images,
            "n_boxes": self.n_boxes,
            "ok": self.ok,
            "counts": self.counts,
            "severities": self.severities,
            "examples": [i.__dict__ for i in self.examples],
        }

    def summary(self) -> str:
        lines = [
            f"images={self.n_images} boxes={self.n_boxes} "
            f"errors={self.n_errors} warnings={self.n_warnings}"
        ]
        for code, n in sorted(self.counts.items()):
            lines.append(f"  [{self.severities[code]}] {code}: {n}")
        return "\n".join(lines)


def validate_boxes(df: pd.DataFrame, report: ValidationReport, tolerance_px: float = 1.0) -> None:
    """Check box geometry. Vectorized; reports every offending row (examples are capped)."""
    coords = df[["x1", "y1", "x2", "y2", "image_width", "image_height"]].to_numpy(dtype=float)
    finite = np.isfinite(coords).all(axis=1)
    x1, y1, x2, y2, w, h = coords.T
    with np.errstate(invalid="ignore"):
        degenerate = finite & ((x2 <= x1) | (y2 <= y1))
        bad_dims = finite & ((w <= 0) | (h <= 0))
        oob = (
            finite
            & ~degenerate
            & (
                (x1 < -tolerance_px)
                | (y1 < -tolerance_px)
                | (x2 > w + tolerance_px)
                | (y2 > h + tolerance_px)
            )
        )
    for mask, sev, code, msg in (
        (~finite, "error", "non_finite_value", "missing or non-numeric coordinate/dimension"),
        (degenerate, "error", "degenerate_box", "x2<=x1 or y2<=y1 (zero/negative size)"),
        (bad_dims, "error", "invalid_image_dims", "image width/height <= 0"),
        (oob, "warning", "box_out_of_bounds", f"box extends >{tolerance_px}px outside image"),
    ):
        for idx in np.flatnonzero(mask):
            row = df.iloc[idx]
            report.add(sev, code, str(row["image_name"]), f"row {idx}: {msg}")
    empty_cls = df["class"] == ""
    for idx in np.flatnonzero(empty_cls.to_numpy()):
        report.add("error", "empty_class", str(df.iloc[idx]["image_name"]), f"row {idx}")
    dup = df.duplicated(keep="first")
    for idx in np.flatnonzero(dup.to_numpy()):
        report.add("warning", "duplicate_row", str(df.iloc[idx]["image_name"]), f"row {idx}")


def validate_images(
    df: pd.DataFrame, image_dir: Path, report: ValidationReport, deep: bool = False
) -> None:
    """Check each referenced image exists, opens, and matches the CSV dimensions.

    ``deep=True`` fully decodes every image (slow on large datasets); the default verifies
    the file structure and reads dimensions from the header.
    """
    dims = df.groupby("image_name")[["image_width", "image_height"]].agg(["min", "max"])
    for name, row in dims.iterrows():
        name = str(name)
        w_min, w_max, h_min, h_max = row.to_numpy(dtype=float)
        if w_min != w_max or h_min != h_max:
            report.add("error", "inconsistent_dims_in_csv", name, "rows disagree on image size")
        path = image_dir / name
        if not path.is_file():
            report.add("error", "missing_image", name, f"not found: {path}")
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                actual = im.size  # (w, h)
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad: importing Ultralytics patches PIL.Image.open, which then raises
            # ModuleNotFoundError (pi_heif) instead of UnidentifiedImageError on corrupt files.
            report.add("error", "corrupt_image", name, f"{type(exc).__name__}: {exc}")
            continue
        if deep and cv2.imread(str(path)) is None:
            report.add("error", "corrupt_image", name, "cv2 failed to decode")
            continue
        if (actual[0], actual[1]) != (int(w_min), int(h_min)):
            report.add(
                "error",
                "dimension_mismatch",
                name,
                f"CSV says {int(w_min)}x{int(h_min)}, file is {actual[0]}x{actual[1]}",
            )


def validate_dataset(
    csv_path: str | Path,
    image_dir: str | Path,
    deep: bool = False,
    check_images: bool = True,
) -> ValidationReport:
    """Validate one annotation CSV against an image directory."""
    df = load_annotations(csv_path)
    report = ValidationReport(n_images=df["image_name"].nunique(), n_boxes=len(df))
    validate_boxes(df, report)
    if check_images:
        validate_images(df, Path(image_dir), report, deep=deep)
    logger.info("validated %s: %s", csv_path, report.summary().splitlines()[0])
    return report
