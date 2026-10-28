import json

import numpy as np
import pytest

from helpers import ScriptedDetector, dets
from shelfsight.data.convert import convert_split
from shelfsight.data.formats import load_annotations
from shelfsight.errors import ShelfSightError
from shelfsight.evaluation.benchmark import (
    NOT_MEASURED,
    benchmark_detector,
    compare_detections,
    comparison_table,
    save_benchmark,
)
from shelfsight.evaluation.evaluate import evaluate_detector, latency_stats, load_gt
from shelfsight.evaluation.failure_analysis import analyze_image, failure_report
from shelfsight.evaluation.metrics import DetectionAccumulator, compute_ap
from shelfsight.inference.base import Detections


def test_compute_ap_known_cases():
    assert compute_ap(np.array([1.0]), np.array([1.0])) == pytest.approx(1.0)
    assert compute_ap(np.array([]), np.array([])) == 0.0
    # FP ranked first, then TP: precision 0.5 whenever recall reaches 1 -> AP 0.5
    assert compute_ap(np.array([0.0, 1.0]), np.array([0.0, 0.5])) == pytest.approx(0.5)
    # recall only reaches 0.5 with precision 1 -> AP ~ 0.5 (r<=0.5 satisfied)
    assert compute_ap(np.array([0.5]), np.array([1.0])) == pytest.approx(51 / 101)


def acc_one(pred, scores, gt):
    a = DetectionAccumulator()
    pred = np.array(pred, dtype=float).reshape(-1, 4)
    a.add(
        pred,
        scores,
        np.zeros(len(pred)),
        np.array(gt, dtype=float).reshape(-1, 4),
        np.zeros(len(gt)),
    )
    return a.compute(pr_conf=0.25)


def test_perfect_prediction():
    m = acc_one([[0, 0, 10, 10]], [0.9], [[0, 0, 10, 10]])
    assert m["map50"] == pytest.approx(1.0) and m["map50_95"] == pytest.approx(1.0)
    assert m["precision"] == 1.0 and m["recall"] == 1.0


def test_localisation_quality_affects_map5095_only():
    # IoU = 0.62 -> true positive at thresholds 0.50, 0.55, 0.60 only
    m = acc_one([[0, 0, 100, 62]], [0.9], [[0, 0, 100, 100]])
    assert m["map50"] == pytest.approx(1.0)
    assert m["map50_95"] == pytest.approx(0.3)


def test_false_positive_and_miss():
    m = acc_one([[50, 50, 60, 60], [0, 0, 10, 10]], [0.9, 0.8], [[0, 0, 10, 10]])
    assert m["map50"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(1.0)
    none = acc_one([], [], [[0, 0, 10, 10]])
    assert none["map50"] == 0.0 and none["recall"] == 0.0 and none["precision"] == 0.0


def test_duplicate_detections_count_as_false_positives():
    m = acc_one([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8], [[0, 0, 10, 10]])
    assert m["precision"] == pytest.approx(0.5)


def test_class_mismatch_is_not_a_match():
    a = DetectionAccumulator()
    a.add(
        np.array([[0, 0, 10, 10.0]]),
        np.array([0.9]),
        np.array([1]),
        np.array([[0, 0, 10, 10.0]]),
        np.array([0]),
    )
    assert a.compute()["map50"] == 0.0


def test_accumulator_requires_images():
    with pytest.raises(ValueError):
        DetectionAccumulator().compute()


@pytest.fixture
def yolo_split(synth, tmp_path):
    root, csvs = synth
    out = tmp_path / "yolo"
    convert_split(load_annotations(csvs["train"]), root / "images", out, "train", ["object"])
    return out / "images" / "train", out / "labels" / "train"


def oracle(images, labels, keep=1.0):
    from shelfsight.inference.base import load_image

    script = []
    for p in sorted(images.iterdir()):
        boxes, cls = load_gt(labels / f"{p.stem}.txt", load_image(p).shape[:2])
        script.append(dets(boxes[: int(len(boxes) * keep)]))
    return ScriptedDetector(script)


def test_evaluate_perfect_oracle(yolo_split):
    images, labels = yolo_split
    det = oracle(images, labels)
    res, records = evaluate_detector(det, images, labels)
    assert res["map50"] == pytest.approx(1.0) and res["count_mae"] == 0.0
    assert res["precision"] == 1.0 and res["recall"] == 1.0
    assert res["n_images"] == len(records) == 4
    assert det.conf_threshold == 0.25  # restored after evaluation
    assert res["latency_wall_ms"]["median_ms"] >= 0


def test_evaluate_half_recall_and_count_error(yolo_split):
    images, labels = yolo_split
    res, _ = evaluate_detector(oracle(images, labels, keep=0.5), images, labels)
    assert 0.4 < res["recall"] < 0.6 and res["precision"] == 1.0
    assert res["count_mae"] > 0


def test_evaluate_empty_detector_and_agnostic(yolo_split):
    images, labels = yolo_split
    res, _ = evaluate_detector(
        ScriptedDetector([Detections.empty()]), images, labels, class_agnostic=True
    )
    assert res["map50"] == 0.0 and res["class_agnostic"] is True
    assert res["count_mae"] == pytest.approx(res["mean_gt_count"])


def test_evaluate_errors(tmp_path):
    from shelfsight.data.formats import DatasetError

    (tmp_path / "empty").mkdir()
    with pytest.raises(DatasetError):
        evaluate_detector(ScriptedDetector([Detections.empty()]), tmp_path / "empty", tmp_path)


def test_failure_analysis(yolo_split, tmp_path):
    images, labels = yolo_split
    _, records = evaluate_detector(oracle(images, labels, keep=0.5), images, labels)
    row = analyze_image(records[0], conf=0.25, dense_gt=5)
    assert row["false_negatives"] == row["gt"] - row["tp"] > 0
    assert row["false_positives"] == 0 and row["dense_scene"] is True
    df = failure_report(records, tmp_path / "fa", images_dir=images, top_k=2)
    assert (tmp_path / "fa" / "failures.csv").is_file() and len(df) == 4
    assert any(p.name.startswith("most_missed_") for p in (tmp_path / "fa").iterdir())


def test_latency_stats():
    s = latency_stats([1, 2, 3, 4, 100])
    assert s["median_ms"] == 3 and s["p95_ms"] > 4 and s["max_ms"] == 100
    assert latency_stats([]) == {}


def test_benchmark_detector_and_table(tmp_path):
    det = ScriptedDetector([dets([[0, 0, 5, 5]])])
    imgs = [np.zeros((32, 32, 3), np.uint8)]
    r = benchmark_detector(det, imgs, warmup_runs=2, timed_runs=5)
    assert det.calls == 7 and r["settings"]["timed_runs"] == 5
    assert r["model_path"] == NOT_MEASURED and r["load_time_ms"] == NOT_MEASURED
    assert r["latency_wall_ms"]["p95_ms"] >= r["latency_wall_ms"]["median_ms"]
    assert r["input"]["batch_size"] == 1 and "python" in r["system"] and "cpu_model" in r["system"]
    p = save_benchmark(r, tmp_path / "b" / "m.json")
    table = comparison_table([p])
    assert NOT_MEASURED in table  # accuracy was never measured for this model
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"map50": 0.5, "map50_95": 0.25, "precision": 0.9, "recall": 0.8}))
    assert "0.500" in comparison_table([p], {"m": ev})
    with pytest.raises(ValueError):
        benchmark_detector(det, [], 1, 1)
    with pytest.raises(ValueError):
        benchmark_detector(det, imgs, 0, 0)


def test_benchmark_records_model_hash(tiny_onnx):
    from shelfsight.inference.onnx_backend import OnnxDetector

    det = OnnxDetector(tiny_onnx, num_threads=1)
    r = benchmark_detector(det, [np.zeros((64, 64, 3), np.uint8)], 1, 3, tiny_onnx, 12.3)
    assert len(r["model_sha256"]) == 64 and r["model_size_mb"] > 0
    assert r["settings"]["requested_num_threads"] == 1 and r["load_time_ms"] == 12.3


def test_compare_detections():
    a = dets([[0, 0, 10, 10], [20, 20, 30, 30]])
    assert compare_detections(a, a)["match_rate"] == 1.0
    shifted = dets([[0, 0, 10, 10], [100, 100, 110, 110]])
    assert compare_detections(a, shifted)["match_rate"] == 0.5
    assert compare_detections(a, Detections.empty())["match_rate"] == 0.0
    assert compare_detections(Detections.empty(), Detections.empty())["match_rate"] == 1.0
    assert issubclass(ShelfSightError, Exception)


@pytest.mark.integration
def test_map_matches_ultralytics_ap_on_continuous_scores():
    """Cross-validation: same TP matrix + scores through Ultralytics' ap_per_class.

    Uses continuous random scores (no ties); with saturated/tied scores any two
    implementations may order equal scores differently and legitimately disagree.
    """
    pytest.importorskip("ultralytics")
    from ultralytics.utils.metrics import ap_per_class

    from shelfsight.evaluation.metrics import match_image

    rng = np.random.default_rng(0)
    tps, confs, pcls, tcls = [], [], [], []
    acc = DetectionAccumulator()
    for _ in range(40):
        n_gt = 10
        xy = rng.uniform(0, 400, (n_gt, 2))
        gt = np.hstack([xy, xy + rng.uniform(20, 60, (n_gt, 2))])
        keep = rng.random(n_gt) < 0.75
        good = gt[keep] + rng.normal(0, 3, (int(keep.sum()), 4))
        fxy = rng.uniform(0, 400, (8, 2))
        fp = np.hstack([fxy, fxy + rng.uniform(20, 60, (8, 2))])
        pred = np.vstack([good, fp])
        scores = np.concatenate([rng.uniform(0.3, 1.0, len(good)), rng.uniform(0.0, 0.7, 8)])
        cls = np.zeros(len(pred), dtype=np.int64)
        gcls = np.zeros(n_gt, dtype=np.int64)
        acc.add(pred, scores, cls, gt, gcls)
        tps.append(match_image(pred, scores, cls, gt, gcls))
        confs.append(scores)
        pcls.append(cls)
        tcls.append(gcls)
    ap = ap_per_class(
        np.concatenate(tps), np.concatenate(confs), np.concatenate(pcls), np.concatenate(tcls)
    )[5]
    mine = acc.compute()
    assert mine["map50"] == pytest.approx(ap[:, 0].mean(), abs=0.01)
    assert mine["map50_95"] == pytest.approx(ap.mean(), abs=0.01)
