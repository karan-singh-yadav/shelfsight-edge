"""Backend-independent pre/post-processing for YOLOv8-style detectors.

Decode expects the raw head output of an Ultralytics YOLOv8 detection model exported to ONNX:
shape (1, 4+nc, num_anchors) with rows [cx, cy, w, h, class scores...] in letterboxed pixels.
Channels-last (1, num_anchors, 4+nc) is also accepted (detected by shape).
"""

from __future__ import annotations

import cv2
import numpy as np

from shelfsight.inference.base import Detections


def letterbox(
    image: np.ndarray, size: int, color: int = 114
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio and pad to ``size`` x ``size``.

    Returns (padded image, scale ratio, (pad_x, pad_y)) where original = (padded - pad) / ratio.
    """
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    new_w, new_h = round(w * ratio), round(h * ratio)
    resized = (
        cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if (new_w, new_h) != (w, h)
        else image
    )
    pad_x, pad_y = (size - new_w) / 2.0, (size - new_h) / 2.0
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color)
    )
    return out, ratio, (float(left), float(top))


def preprocess_bgr(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """BGR uint8 -> float32 NCHW RGB in [0,1] plus letterbox parameters."""
    padded, ratio, pad = letterbox(image, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, ratio, pad


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against many."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a + b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    return np.stack([box_iou(box, b) for box in a])


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_det: int) -> np.ndarray:
    """Greedy NMS; returns kept indices sorted by descending score."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=np.int64)


def decode_yolov8(
    output: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    ratio: float,
    pad: tuple[float, float],
    orig_shape: tuple[int, int],
    class_names: tuple[str, ...] = (),
    pre_nms_cap: int = 30000,
    channels_first: bool | None = None,
) -> Detections:
    """Decode raw YOLOv8 output into :class:`Detections` in original-image pixels.

    ``channels_first`` states the layout explicitly ((4+nc, A) if True, (A, 4+nc) if False).
    When ``None`` it is inferred (fewer rows than columns => channels-first), which is only
    ambiguous when the anchor count is smaller than the channel count (never for real models).
    """
    pred = np.asarray(output)
    if pred.ndim == 3:
        if pred.shape[0] != 1:
            raise ValueError(f"batch size 1 expected, got output shape {pred.shape}")
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"unexpected output shape {output.shape}")
    if channels_first is None:
        channels_first = pred.shape[0] < pred.shape[1]
    if channels_first:  # (4+nc, A) -> (A, 4+nc)
        pred = pred.T
    if pred.shape[1] < 5:
        raise ValueError(f"output has too few channels: {output.shape}")
    class_scores = pred[:, 4:]
    scores = class_scores.max(axis=1)
    mask = scores >= conf_threshold
    if not mask.any():
        return Detections.empty(class_names)
    pred, scores = pred[mask], scores[mask]
    class_ids = class_scores[mask].argmax(axis=1)
    if len(scores) > pre_nms_cap:
        top = np.argsort(-scores)[:pre_nms_cap]
        pred, scores, class_ids = pred[top], scores[top], class_ids[top]

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    # Class-aware NMS via coordinate offset.
    offset = class_ids[:, None].astype(np.float32) * (max(orig_shape) * 2 + 4096)
    keep = nms_numpy(boxes + offset, scores, iou_threshold, max_det)
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio
    h0, w0 = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return Detections(boxes[valid], scores[valid], class_ids[valid], class_names)
