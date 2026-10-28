"""Detection metrics implemented from scratch so every backend is scored identically.

* AP uses COCO-style 101-point interpolation over the precision envelope (see compute_ap).
* mAP50 / mAP50-95 average over classes that have ground truth (and, for 50-95, 10 IoU
  thresholds 0.50:0.05:0.95).
* Precision / recall are MICRO-averaged at IoU 0.5 for predictions with score >= ``pr_conf``.
  They are therefore not directly comparable with the macro P/R at best-F1 that Ultralytics
  prints during validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shelfsight.inference.postprocess import iou_matrix

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101-point AP.

    For each recall threshold r in {0, 0.01, ..., 1}, take the maximum precision achieved at
    recall >= r (0 if that recall is never reached) and average the 101 values. A perfect
    detector scores exactly 1.0. Ultralytics' validator integrates with the trapezoid rule
    instead (perfect = 0.995), so absolute values can differ slightly from its printouts.
    """
    if len(recall) == 0:
        return 0.0
    envelope = np.maximum.accumulate(np.asarray(precision)[::-1])[::-1]
    thresholds = np.linspace(0.0, 1.0, 101) - 1e-9
    idx = np.searchsorted(np.asarray(recall), thresholds, side="left")
    q = np.zeros(101)
    valid = idx < len(envelope)
    q[valid] = envelope[idx[valid]]
    return float(q.mean())


def match_image(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_cls: np.ndarray,
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
    thresholds: np.ndarray = IOU_THRESHOLDS,
) -> np.ndarray:
    """Greedy per-class matching. Returns bool (n_pred, n_thr): is prediction a true positive."""
    n = len(pred_boxes)
    tp = np.zeros((n, len(thresholds)), dtype=bool)
    if n == 0 or len(gt_boxes) == 0:
        return tp
    order = np.argsort(-pred_scores, kind="stable")
    ious = iou_matrix(pred_boxes, gt_boxes)
    same_cls = pred_cls[:, None] == gt_cls[None, :]
    ious = np.where(same_cls, ious, 0.0)
    for t_idx, thr in enumerate(thresholds):
        used = np.zeros(len(gt_boxes), dtype=bool)
        for i in order:
            cand = np.where(used, 0.0, ious[i])
            j = int(cand.argmax())
            if cand[j] >= thr:
                used[j] = True
                tp[i, t_idx] = True
    return tp


@dataclass
class DetectionAccumulator:
    """Collects per-image match results and computes dataset-level metrics."""

    thresholds: np.ndarray = field(default_factory=lambda: IOU_THRESHOLDS.copy())

    def __post_init__(self) -> None:
        self._tp: list[np.ndarray] = []
        self._scores: list[np.ndarray] = []
        self._pred_cls: list[np.ndarray] = []
        self._gt_cls: list[np.ndarray] = []

    def add(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_cls: np.ndarray,
        gt_boxes: np.ndarray,
        gt_cls: np.ndarray,
    ) -> None:
        pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
        gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
        pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
        pred_cls = np.asarray(pred_cls, dtype=np.int64).reshape(-1)
        gt_cls = np.asarray(gt_cls, dtype=np.int64).reshape(-1)
        self._tp.append(
            match_image(pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls, self.thresholds)
        )
        self._scores.append(pred_scores)
        self._pred_cls.append(pred_cls)
        self._gt_cls.append(gt_cls)

    def compute(self, pr_conf: float = 0.25) -> dict[str, object]:
        if not self._tp:
            raise ValueError("no images were added")
        tp = np.concatenate(self._tp)
        scores = np.concatenate(self._scores)
        pcls = np.concatenate(self._pred_cls)
        gcls = np.concatenate(self._gt_cls)
        n_gt = int(len(gcls))
        classes = np.unique(gcls)
        ap = np.zeros((len(classes), len(self.thresholds)))
        for ci, c in enumerate(classes):
            sel = pcls == c
            n_c = int((gcls == c).sum())
            if not sel.any() or n_c == 0:
                continue
            order = np.argsort(-scores[sel], kind="stable")
            tpc_all = tp[sel][order]
            for t in range(len(self.thresholds)):
                tpc = np.cumsum(tpc_all[:, t])
                fpc = np.cumsum(~tpc_all[:, t])
                recall = tpc / n_c
                precision = tpc / np.maximum(tpc + fpc, 1e-12)
                ap[ci, t] = compute_ap(recall, precision)
        keep = scores >= pr_conf
        n_pred = int(keep.sum())
        n_tp = int(tp[keep, 0].sum())
        return {
            "n_gt": n_gt,
            "n_predictions_at_pr_conf": n_pred,
            "precision": n_tp / n_pred if n_pred else 0.0,
            "recall": n_tp / n_gt if n_gt else 0.0,
            "map50": float(ap[:, 0].mean()) if len(classes) else 0.0,
            "map50_95": float(ap.mean()) if len(classes) else 0.0,
            "per_class_ap50": {int(c): float(ap[i, 0]) for i, c in enumerate(classes)},
            "pr_conf": pr_conf,
        }
