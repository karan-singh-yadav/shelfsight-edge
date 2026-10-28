"""Synthetic SKU-110K-format dataset for pipeline development and tests.

These images are procedurally generated rectangles on a gradient. They exist ONLY to exercise
the tooling without downloading licensed data. Never report model metrics measured on them
as retail performance.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_synthetic_sku_dataset(
    out_dir: str | Path,
    n_train: int = 8,
    n_val: int = 3,
    n_test: int = 3,
    seed: int = 0,
    size: tuple[int, int] = (320, 240),
) -> dict[str, Path]:
    """Write images/ and headerless annotations_{split}.csv in SKU-110K column order."""
    root = Path(out_dir)
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    width, height = size
    csvs: dict[str, Path] = {}
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        lines: list[str] = []
        for i in range(count):
            name = f"{split}_{i}.jpg"
            gradient = np.linspace(60, 200, width, dtype=np.uint8)
            img = np.tile(gradient, (height, 1))[:, :, None].repeat(3, axis=2).copy()
            rows, cols = int(rng.integers(2, 4)), int(rng.integers(4, 8))
            cell_w, cell_h = width // cols, height // rows
            for r in range(rows):
                for c in range(cols):
                    x1, y1 = c * cell_w + 2, r * cell_h + 2
                    x2, y2 = (c + 1) * cell_w - 2, (r + 1) * cell_h - 2
                    color = tuple(int(v) for v in rng.integers(0, 255, 3))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                    lines.append(f"{name},{x1},{y1},{x2},{y2},object,{width},{height}")
            cv2.imwrite(str(root / "images" / name), img)
        csv_path = root / f"annotations_{split}.csv"
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        csvs[split] = csv_path
    return csvs
