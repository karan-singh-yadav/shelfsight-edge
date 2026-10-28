"""Convert SKU-110K-style annotations into the YOLO (Ultralytics) dataset layout.

Output layout::

    <out_root>/images/<split>/<image>
    <out_root>/labels/<split>/<image stem>.txt     # "class xc yc w h", normalized
    <out_root>/dataset.yaml

Every dropped or clipped box is counted and returned; nothing is changed silently.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from shelfsight.data.formats import DatasetError, clip_box, is_finite_box, xyxy_to_yolo

logger = logging.getLogger(__name__)


@dataclass
class ConversionStats:
    split: str
    images: int = 0
    boxes_written: int = 0
    boxes_dropped_invalid: int = 0
    boxes_clipped: int = 0
    images_without_boxes: int = 0
    images_missing: int = 0


def _place_image(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            logger.debug("hardlink failed for %s, falling back to copy", src)
    shutil.copy2(src, dst)


def convert_split(
    df: pd.DataFrame,
    image_dir: Path,
    out_root: Path,
    split: str,
    class_names: list[str],
    image_mode: str = "copy",
    min_box_px: float = 1.0,
    strict: bool = True,
) -> ConversionStats:
    """Convert one split. ``strict`` raises on missing images or size mismatches."""
    if image_mode not in ("copy", "hardlink"):
        raise ValueError("image_mode must be 'copy' or 'hardlink'")
    cls_map = {n: i for i, n in enumerate(class_names)}
    unknown = sorted(set(df["class"]) - set(cls_map))
    if unknown:
        raise DatasetError(f"Classes not in mapping {class_names}: {unknown}")
    stems = [Path(n).stem for n in df["image_name"].unique()]
    if len(stems) != len(set(stems)):
        raise DatasetError("Two images share a filename stem; YOLO labels would collide")

    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    stats = ConversionStats(split=split)

    for name, group in df.groupby("image_name", sort=True):
        name = str(name)
        src = image_dir / name
        if not src.is_file():
            stats.images_missing += 1
            if strict:
                raise DatasetError(f"Missing image for split '{split}': {src}")
            logger.warning("skipping missing image %s", src)
            continue
        w = float(group["image_width"].iloc[0])
        h = float(group["image_height"].iloc[0])
        try:
            with Image.open(src) as im:
                actual_size = im.size
        except Exception as exc:  # noqa: BLE001 - see validate_dataset: PIL may be patched
            if strict:
                raise DatasetError(
                    f"{name}: cannot read image ({type(exc).__name__}: {exc})"
                ) from exc
            logger.warning("skipping unreadable image %s: %s", src, exc)
            stats.images_missing += 1
            continue
        if strict and actual_size != (int(w), int(h)):
            raise DatasetError(f"{name}: CSV size {int(w)}x{int(h)} != file size {actual_size}")
        lines: list[str] = []
        for row in group.itertuples(index=False):
            raw = (row.x1, row.y1, row.x2, row.y2)
            if not is_finite_box(*raw, w, h) or w <= 0 or h <= 0:
                stats.boxes_dropped_invalid += 1
                continue
            cx1, cy1, cx2, cy2 = clip_box(*raw, w, h)
            if (cx1, cy1, cx2, cy2) != raw:
                stats.boxes_clipped += 1
            if (cx2 - cx1) < min_box_px or (cy2 - cy1) < min_box_px:
                stats.boxes_dropped_invalid += 1
                continue
            xc, yc, bw, bh = xyxy_to_yolo(cx1, cy1, cx2, cy2, w, h)
            lines.append(f"{cls_map[row[5]]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (lbl_out / f"{Path(name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        _place_image(src, img_out / Path(name).name, image_mode)
        stats.images += 1
        stats.boxes_written += len(lines)
        stats.images_without_boxes += int(not lines)
    logger.info("converted split %s: %s", split, stats)
    return stats


def write_dataset_yaml(out_root: Path, class_names: list[str], splits: list[str]) -> Path:
    """Write an Ultralytics dataset YAML with no absolute paths.

    The ``path`` key is deliberately omitted so the dataset root is the YAML's own folder,
    keeping the dataset relocatable between laptop and cloud. (Ultralytics behaviour to be
    re-verified when it is installed in Phase 3.)
    """
    data: dict[str, object] = {"names": dict(enumerate(class_names))}
    for split in splits:
        data[split] = f"images/{split}"
    path = out_root / "dataset.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file (used by tests and visualization)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c, xc, yc, bw, bh = line.split()
            out.append((int(c), float(xc), float(yc), float(bw), float(bh)))
    return out
