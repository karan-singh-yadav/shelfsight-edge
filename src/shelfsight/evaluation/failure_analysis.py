"""Automatic triage of detector failures from evaluation records.

Automatically detectable categories: missed detections (false negatives), false positives,
dense scenes, tiny products and poorly localised boxes (matched but low IoU). Occlusion and
glare/reflections cannot be detected reliably from boxes alone: tag those by hand in the
generated CSV after reviewing the overlays.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from shelfsight.drawing import draw_detections
from shelfsight.evaluation.evaluate import ImageRecord
from shelfsight.inference.postprocess import iou_matrix


def analyze_image(
    rec: ImageRecord, conf: float = 0.25, dense_gt: int = 100, tiny_rel_area: float = 0.0005
) -> dict[str, object]:
    det = rec.detections.filter(conf)
    n_gt, n_pred = len(rec.gt_boxes), len(det)
    ious = iou_matrix(det.boxes, rec.gt_boxes)
    matched_gt: set[int] = set()
    tp = 0
    low_iou = 0
    for i in np.argsort(-det.scores):
        if ious.size == 0:
            break
        row = ious[i].copy()
        row[list(matched_gt)] = 0.0
        j = int(row.argmax())
        if row[j] >= 0.5:
            matched_gt.add(j)
            tp += 1
            low_iou += int(row[j] < 0.7)
    h, w = rec.shape
    areas = (
        (rec.gt_boxes[:, 2] - rec.gt_boxes[:, 0])
        * (rec.gt_boxes[:, 3] - rec.gt_boxes[:, 1])
        / (h * w)
        if n_gt
        else np.zeros(0)
    )
    return {
        "image": rec.name,
        "gt": n_gt,
        "pred": n_pred,
        "tp": tp,
        "false_negatives": n_gt - tp,
        "false_positives": n_pred - tp,
        "poorly_localised": low_iou,
        "count_error": n_pred - n_gt,
        "dense_scene": n_gt >= dense_gt,
        "tiny_products": int((areas < tiny_rel_area).sum()),
        "manual_tags": "",  # fill by hand: occlusion, glare, ...
    }


def failure_report(
    records: list[ImageRecord],
    out_dir: str | Path,
    conf: float = 0.25,
    top_k: int = 10,
    images_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Write ``failures.csv`` and overlay images (GT green, predictions red) for the worst
    images by absolute count error and by missed detections. Returns the summary table."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([analyze_image(r, conf) for r in records])
    df.to_csv(out / "failures.csv", index=False)
    if images_dir is None:
        return df
    by_name = {r.name: r for r in records}
    picks = {
        "worst_count_error": df.reindex(df["count_error"].abs().sort_values(ascending=False).index),
        "most_missed": df.sort_values("false_negatives", ascending=False),
        "most_false_positives": df.sort_values("false_positives", ascending=False),
    }
    for tag, table in picks.items():
        for name in table["image"].head(top_k):
            rec = by_name[name]
            img = cv2.imread(str(Path(images_dir) / name))
            if img is None:
                continue
            for x1, y1, x2, y2 in rec.gt_boxes:
                cv2.rectangle(img, (round(x1), round(y1)), (round(x2), round(y2)), (0, 200, 0), 1)
            from shelfsight.inference.base import Detections

            shown = rec.detections.filter(conf)
            img = draw_detections(
                img,
                Detections(shown.boxes, shown.scores, shown.class_ids),
                show_conf=False,
                thickness=1,
            )
            cv2.imwrite(str(out / f"{tag}_{Path(name).stem}.jpg"), img)
    return df
