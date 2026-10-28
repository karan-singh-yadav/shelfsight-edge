"""Run a detector over a YOLO-format split and report accuracy, counting error and latency."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shelfsight.analytics.counting import count_mae
from shelfsight.data.convert import read_yolo_labels
from shelfsight.data.formats import IMAGE_EXTENSIONS, DatasetError, yolo_to_xyxy
from shelfsight.errors import InvalidImageError
from shelfsight.evaluation.metrics import DetectionAccumulator
from shelfsight.inference.base import Detections, Detector, load_image

logger = logging.getLogger(__name__)


@dataclass
class ImageRecord:
    name: str
    shape: tuple[int, int]
    gt_boxes: np.ndarray
    gt_cls: np.ndarray
    detections: Detections  # at eval_conf (low), class-remapped if agnostic


def load_gt(label_path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Read YOLO labels into pixel xyxy boxes + class ids (empty arrays if file is missing)."""
    if not label_path.is_file():
        return np.zeros((0, 4)), np.zeros(0, dtype=np.int64)
    h, w = shape
    rows = read_yolo_labels(label_path)
    boxes = np.array([yolo_to_xyxy(xc, yc, bw, bh, w, h) for _, xc, yc, bw, bh in rows])
    return boxes.reshape(-1, 4), np.array([r[0] for r in rows], dtype=np.int64)


def latency_stats(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {}
    a = np.asarray(values_ms)
    return {
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def evaluate_detector(
    detector: Detector,
    images_dir: str | Path,
    labels_dir: str | Path,
    class_agnostic: bool = False,
    pr_conf: float = 0.25,
    eval_conf: float = 0.001,
    max_images: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], list[ImageRecord]]:
    """Evaluate on every image in ``images_dir`` (labels looked up by stem in ``labels_dir``).

    Inference runs ONCE at ``eval_conf`` (low) for mAP; precision/recall and count error are
    derived from the same predictions filtered at ``pr_conf``. ``class_agnostic`` collapses all
    classes to one (needed to score a COCO-pretrained model on single-class SKU-110K).
    """
    img_root, lbl_root = Path(images_dir), Path(labels_dir)
    files = sorted(p for p in img_root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise DatasetError(f"No images in {img_root}")
    if max_images:
        files = files[:max_images]
    acc = DetectionAccumulator()
    records: list[ImageRecord] = []
    wall: list[float] = []
    stage: dict[str, list[float]] = {"preprocess_ms": [], "inference_ms": [], "postprocess_ms": []}
    pred_counts: list[int] = []
    gt_counts: list[int] = []
    no_label = skipped = 0
    old_conf = detector.conf_threshold
    detector.conf_threshold = eval_conf
    import time

    try:
        for i, path in enumerate(files):
            try:
                img = load_image(path)
            except InvalidImageError as exc:
                logger.warning("skipping unreadable image: %s", exc)
                skipped += 1
                continue
            shape = img.shape[:2]
            label_path = lbl_root / f"{path.stem}.txt"
            no_label += int(not label_path.is_file())
            gt_boxes, gt_cls = load_gt(label_path, shape)
            t0 = time.perf_counter()
            det = detector.predict(img)
            wall.append((time.perf_counter() - t0) * 1000.0)
            for k in stage:
                stage[k].append(getattr(detector.last_timings, k))
            if class_agnostic:
                gt_cls = np.zeros_like(gt_cls)
                det = Detections(det.boxes, det.scores, np.zeros_like(det.class_ids), ("object",))
            acc.add(det.boxes, det.scores, det.class_ids, gt_boxes, gt_cls)
            pred_counts.append(int((det.scores >= pr_conf).sum()))
            gt_counts.append(len(gt_boxes))
            records.append(ImageRecord(path.name, shape, gt_boxes, gt_cls, det))
            if progress:
                progress(i + 1, len(files))
    finally:
        detector.conf_threshold = old_conf

    if not records:
        raise DatasetError("No readable images were evaluated")
    metrics = acc.compute(pr_conf)
    result: dict[str, Any] = {
        **metrics,
        "n_images": len(records),
        "n_images_unreadable": skipped,
        "n_images_without_label_file": no_label,
        "class_agnostic": class_agnostic,
        "eval_conf": eval_conf,
        "iou_nms": detector.iou_threshold,
        "max_det": detector.max_det,
        "imgsz": detector.imgsz,
        "count_mae": count_mae(pred_counts, gt_counts),
        "mean_gt_count": float(np.mean(gt_counts)),
        "latency_wall_ms": latency_stats(wall),
        "latency_stage_median_ms": {k: float(np.median(v)) for k, v in stage.items()},
        "note": (
            "Latency here was measured while evaluating at a very low confidence threshold, "
            "which inflates postprocessing time; use the benchmark script for latency claims."
        ),
    }
    return result, records
