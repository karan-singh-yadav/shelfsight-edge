from pathlib import Path

import cv2
import numpy as np
import pytest

from shelfsight.errors import InvalidImageError, ModelNotFoundError
from shelfsight.inference.base import Detections, load_image, validate_image
from shelfsight.inference.factory import create_detector
from shelfsight.inference.onnx_backend import OnnxDetector


def test_detections_schema_and_helpers():
    d = Detections([[0, 0, 5, 5], [1, 1, 6, 6]], [0.9, 0.2], [0, 1], ("a", "b"))
    assert len(d) == 2 and d.boxes.dtype == np.float32 and d.class_ids.dtype == np.int64
    assert d.counts() == {"a": 1, "b": 1}
    assert len(d.filter(0.5)) == 1
    assert list(d.to_dataframe().columns) == [
        "x1", "y1", "x2", "y2", "score", "class_id", "class_name",
    ]  # fmt: skip
    assert d.name_of(7) == "class_7"
    with pytest.raises(ValueError):
        Detections([[0, 0, 1, 1]], [0.5, 0.5], [0])


def test_empty_detections():
    e = Detections.empty(("a",))
    assert len(e) == 0 and e.counts() == {} and len(e.to_dataframe()) == 0


@pytest.mark.parametrize(
    "bad",
    [None, "x", np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.float32),
     np.zeros((10, 10, 4), dtype=np.uint8), np.zeros((1, 1, 3), dtype=np.uint8)],
)  # fmt: skip
def test_validate_image_rejects(bad):
    with pytest.raises(InvalidImageError):
        validate_image(bad)


def test_load_image_errors(tmp_path):
    with pytest.raises(InvalidImageError, match="not found"):
        load_image(tmp_path / "missing.jpg")
    corrupt = tmp_path / "bad.jpg"
    corrupt.write_bytes(b"\x00\x01garbage")
    with pytest.raises(InvalidImageError, match="decodable"):
        load_image(corrupt)
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(InvalidImageError):
        load_image(empty)


def test_load_image_unicode_path(tmp_path):
    p = tmp_path / "ünï cödé.png"
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((8, 8, 3), np.uint8))
    (tmp_path / "a.png").rename(p)
    assert load_image(p).shape == (8, 8, 3)


def test_onnx_backend_end_to_end_on_tiny_model(tiny_onnx):
    det = OnnxDetector(tiny_onnx, conf_threshold=0.25, num_threads=1)
    assert det.imgsz == 64 and det.class_names == ("object",)
    assert det.session.get_providers() == ["CPUExecutionProvider"]
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = det.predict(img)
    assert len(out) == 1  # 64 identical anchors collapse to one box via NMS
    assert out.boxes[0].tolist() == pytest.approx([22, 22, 42, 42])
    assert out.counts() == {"object": 1}
    t = det.last_timings
    assert t.inference_ms > 0 and t.total_ms >= t.inference_ms
    assert det.warmup(2) > 0


def test_onnx_backend_rescales_to_original_resolution(tiny_onnx):
    det = OnnxDetector(tiny_onnx)
    out = det.predict(np.zeros((128, 128, 3), dtype=np.uint8))  # ratio 0.5 -> boxes x2
    assert out.boxes[0].tolist() == pytest.approx([44, 44, 84, 84])


def test_onnx_backend_conf_threshold_filters(tiny_onnx):
    assert (
        len(OnnxDetector(tiny_onnx, conf_threshold=0.95).predict(np.zeros((64, 64, 3), np.uint8)))
        == 0
    )


def test_onnx_backend_rejects_bad_input_and_missing_model(tiny_onnx, tmp_path):
    det = OnnxDetector(tiny_onnx)
    with pytest.raises(InvalidImageError):
        det.predict(np.zeros((64, 64), dtype=np.uint8))
    with pytest.raises(ModelNotFoundError):
        OnnxDetector(tmp_path / "nope.onnx")


def test_factory(tiny_onnx, tmp_path):
    assert isinstance(create_detector("onnx", tiny_onnx), OnnxDetector)
    with pytest.raises(ValueError, match="Unknown backend"):
        create_detector("tensorrt", tiny_onnx)


def test_pytorch_backend_errors_without_loading_weights(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    with pytest.raises(ModelNotFoundError, match="never downloaded"):
        create_detector("pytorch", tmp_path / "nope.pt")


WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "yolov8n.pt"


@pytest.mark.integration
@pytest.mark.skipif(not WEIGHTS.is_file(), reason="models/yolov8n.pt not downloaded")
def test_pytorch_vs_onnx_consistency_real_model(tmp_path):
    """Real-model check: exports yolov8n to ONNX and compares detections on a sample image."""
    ultralytics = pytest.importorskip("ultralytics")
    pytest.importorskip("onnx")
    from shelfsight.evaluation.benchmark import compare_detections

    sample = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    if not sample.is_file():
        pytest.skip("ultralytics sample image unavailable")
    from ultralytics import YOLO

    onnx_path = Path(
        YOLO(str(WEIGHTS)).export(format="onnx", imgsz=640, dynamic=False, device="cpu")
    )
    img = load_image(sample)
    pt = create_detector("pytorch", WEIGHTS, conf_threshold=0.25)
    ox = create_detector("onnx", onnx_path, conf_threshold=0.25)
    a, b = pt.predict(img), ox.predict(img)
    cmp = compare_detections(a, b)
    assert cmp["match_rate"] >= 0.95 and cmp["count_diff"] == 0
    assert a.counts() == b.counts()
