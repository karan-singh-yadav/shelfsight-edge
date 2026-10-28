import numpy as np
import pytest

from shelfsight.inference.postprocess import (
    box_iou,
    decode_yolov8,
    iou_matrix,
    letterbox,
    nms_numpy,
    preprocess_bgr,
)


def dec(rows, conf, iou, max_det, ratio, pad, shape, names=(), cf=True):
    return decode_yolov8(
        raw(rows, cf), conf, iou, max_det, ratio, pad, shape, names, channels_first=cf
    )


def raw(rows, channels_first=True):
    """rows: list of (cx,cy,w,h,*class_scores) -> raw output tensor with batch dim."""
    arr = np.array(rows, dtype=np.float32)
    return arr.T[None] if channels_first else arr[None]


def test_letterbox_geometry_roundtrip():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out, ratio, (px, py) = letterbox(img, 64)
    assert out.shape == (64, 64, 3)
    assert ratio == pytest.approx(0.32)
    assert px == 0 and py == pytest.approx(16)
    # a point in the original maps back exactly
    x_l, y_l = 100 * ratio + px, 50 * ratio + py
    assert ((x_l - px) / ratio, (y_l - py) / ratio) == pytest.approx((100, 50))


def test_preprocess_layout_and_range():
    img = np.full((10, 20, 3), 255, dtype=np.uint8)
    t, _, _ = preprocess_bgr(img, 32)
    assert t.shape == (1, 3, 32, 32) and t.dtype == np.float32
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_iou_and_nms():
    a = np.array([0, 0, 10, 10.0])
    assert box_iou(a, np.array([[0, 0, 10, 10.0], [20, 20, 30, 30.0]])).tolist() == [1.0, 0.0]
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60.0]])
    keep = nms_numpy(boxes, np.array([0.9, 0.8, 0.7]), 0.5, 10)
    assert keep.tolist() == [0, 2]
    assert nms_numpy(boxes, np.array([0.9, 0.8, 0.7]), 0.5, 1).tolist() == [0]


def test_decode_basic_and_unscale():
    det = dec([(32, 32, 20, 20, 0.9)], 0.25, 0.45, 100, 0.5, (0, 0), (100, 100))
    assert len(det) == 1
    assert det.boxes[0].tolist() == pytest.approx([44, 44, 84, 84])  # (22..42)/0.5
    assert det.scores[0] == pytest.approx(0.9)


def test_decode_accepts_channels_last_and_matches_channels_first():
    rows = [(30, 30, 10, 10, 0.8), (80, 80, 10, 10, 0.6)]
    a = dec(rows, 0.25, 0.45, 10, 1.0, (0, 0), (100, 100), cf=True)
    b = dec(rows, 0.25, 0.45, 10, 1.0, (0, 0), (100, 100), cf=False)
    assert np.allclose(a.boxes, b.boxes) and len(a) == 2


def test_decode_threshold_empty_and_max_det():
    rows = [(10 + 12 * i, 10, 8, 8, 0.5) for i in range(5)]
    assert len(dec(rows, 0.9, 0.45, 10, 1.0, (0, 0), (100, 100))) == 0
    assert len(dec(rows, 0.25, 0.45, 3, 1.0, (0, 0), (100, 100))) == 3


def test_decode_class_aware_nms_keeps_overlapping_different_classes():
    rows = [(30, 30, 20, 20, 0.9, 0.0), (30, 30, 20, 20, 0.0, 0.8)]
    det = dec(rows, 0.25, 0.45, 10, 1.0, (0, 0), (100, 100), ("a", "b"))
    assert sorted(det.class_ids.tolist()) == [0, 1]
    same = [(30, 30, 20, 20, 0.9, 0.0), (31, 31, 20, 20, 0.8, 0.0)]
    assert len(dec(same, 0.25, 0.45, 10, 1.0, (0, 0), (100, 100))) == 1


def test_decode_clips_to_image_and_drops_degenerate():
    rows = [(5, 5, 40, 40, 0.9), (500, 500, 4, 4, 0.9)]  # second is fully outside
    det = dec(rows, 0.25, 0.45, 10, 1.0, (0, 0), (50, 50))
    assert len(det) == 1
    assert det.boxes[0].min() >= 0 and det.boxes[0].max() <= 50


def test_decode_rejects_bad_shapes():
    with pytest.raises(ValueError):
        decode_yolov8(np.zeros((2, 5, 8)), 0.25, 0.45, 10, 1.0, (0, 0), (10, 10))
    with pytest.raises(ValueError):
        decode_yolov8(np.zeros((1, 3, 8)), 0.25, 0.45, 10, 1.0, (0, 0), (10, 10))
