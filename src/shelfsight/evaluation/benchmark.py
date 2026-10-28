"""Reproducible CPU latency/throughput benchmark with full environment capture.

Latency definition: wall-clock time of ``detector.predict(image)`` on an already-decoded
in-memory image, batch size 1. It INCLUDES preprocessing, model inference and postprocessing
and EXCLUDES file decoding, tracking and rendering. Run ONE model per process so the
process-wide peak-memory figure is attributable to that model.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from shelfsight.evaluation.evaluate import latency_stats
from shelfsight.inference.base import Detections, Detector
from shelfsight.inference.postprocess import iou_matrix
from shelfsight.system_info import collect_system_info

NOT_MEASURED = "Not measured"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_rss_mb() -> float | None:
    """Process-wide peak resident memory in MB (None if unavailable on this platform)."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        if hasattr(info, "peak_wset"):  # Windows
            return round(info.peak_wset / 1024**2, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / 1024 if sys.platform != "darwin" else peak / 1024**2, 1)
    except Exception:  # noqa: BLE001
        return None


def thread_settings(detector: Detector) -> dict[str, Any]:
    out: dict[str, Any] = {"requested_num_threads": getattr(detector, "num_threads", None)}
    if sys.modules.get("torch"):
        out["torch_num_threads"] = sys.modules["torch"].get_num_threads()
    return out


def benchmark_detector(
    detector: Detector,
    images: list[np.ndarray],
    warmup_runs: int = 10,
    timed_runs: int = 100,
    model_path: str | Path | None = None,
    load_time_ms: float | None = None,
) -> dict[str, Any]:
    """Time ``timed_runs`` predictions cycling through ``images`` after ``warmup_runs``."""
    if not images:
        raise ValueError("need at least one image")
    if timed_runs < 1 or warmup_runs < 0:
        raise ValueError("timed_runs must be >= 1 and warmup_runs >= 0")
    t0 = time.perf_counter()
    for i in range(warmup_runs):
        detector.predict(images[i % len(images)])
    warmup_ms = (time.perf_counter() - t0) * 1000.0

    wall: list[float] = []
    stages: dict[str, list[float]] = {"preprocess_ms": [], "inference_ms": [], "postprocess_ms": []}
    n_det: list[int] = []
    for i in range(timed_runs):
        img = images[i % len(images)]
        t = time.perf_counter()
        det = detector.predict(img)
        wall.append((time.perf_counter() - t) * 1000.0)
        n_det.append(len(det))
        for k in stages:
            stages[k].append(getattr(detector.last_timings, k))
    total_s = sum(wall) / 1000.0
    model_path = Path(model_path) if model_path else getattr(detector, "model_path", None)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "detector": detector.name,
        "model_path": str(model_path) if model_path else NOT_MEASURED,
        "model_sha256": file_sha256(model_path) if model_path else NOT_MEASURED,
        "model_size_mb": round(Path(model_path).stat().st_size / 1024**2, 3)
        if model_path
        else NOT_MEASURED,
        "input": {
            "imgsz": detector.imgsz,
            "batch_size": 1,
            "n_distinct_images": len(images),
            "image_shapes": sorted({str(im.shape) for im in images}),
        },
        "settings": {
            "conf_threshold": detector.conf_threshold,
            "iou_threshold": detector.iou_threshold,
            "max_det": detector.max_det,
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
            **thread_settings(detector),
        },
        "load_time_ms": round(load_time_ms, 1) if load_time_ms is not None else NOT_MEASURED,
        "warmup_total_ms": round(warmup_ms, 1),
        "latency_wall_ms": latency_stats(wall),
        "latency_stage_median_ms": {k: float(np.median(v)) for k, v in stages.items()},
        "throughput_fps": round(len(wall) / total_s, 3) if total_s > 0 else NOT_MEASURED,
        "throughput_definition": (
            "timed_runs / sum(predict wall time); batch 1; includes pre/inference/post; "
            "excludes image decode, tracking, rendering"
        ),
        "mean_detections_per_image": float(np.mean(n_det)),
        "peak_rss_mb_process": peak_rss_mb() or NOT_MEASURED,
        "peak_rss_note": "process-wide peak since start; includes interpreter, libraries, model",
        "system": collect_system_info(),
        "python_implementation": platform.python_implementation(),
    }


def save_benchmark(result: dict[str, Any], out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return p


def compare_detections(a: Detections, b: Detections, iou_thr: float = 0.5) -> dict[str, float]:
    """Prediction consistency between two detectors on the same image (greedy IoU matching).

    ``match_rate`` = matched pairs / max(len(a), len(b)); 1.0 means identical box sets at
    the IoU threshold. Class labels are ignored here.
    """
    if len(a) == 0 and len(b) == 0:
        return {"match_rate": 1.0, "count_diff": 0.0, "mean_matched_iou": 1.0}
    if len(a) == 0 or len(b) == 0:
        return {
            "match_rate": 0.0,
            "count_diff": float(abs(len(a) - len(b))),
            "mean_matched_iou": 0.0,
        }
    ious = iou_matrix(a.boxes, b.boxes)
    used_b: set[int] = set()
    matched: list[float] = []
    for i in np.argsort(-a.scores):
        row = ious[i].copy()
        row[list(used_b)] = 0.0
        j = int(row.argmax())
        if row[j] >= iou_thr:
            used_b.add(j)
            matched.append(float(row[j]))
    return {
        "match_rate": len(matched) / max(len(a), len(b)),
        "count_diff": float(abs(len(a) - len(b))),
        "mean_matched_iou": float(np.mean(matched)) if matched else 0.0,
    }


def comparison_table(
    result_files: list[str | Path], eval_files: dict[str, str | Path] | None = None
) -> str:
    """Markdown table from benchmark JSONs, with accuracy merged from evaluation JSONs
    (``eval_files`` maps benchmark file stem -> evaluation JSON). Missing values are
    rendered as 'Not measured'."""
    header = (
        "| Model | Size MB | Median ms | p95 ms | FPS | Peak RSS MB | mAP50 | mAP50-95 | P | R |"
    )
    rows = [header, "|---|---|---|---|---|---|---|---|---|---|"]

    def fmt(v: Any, nd: int = 2) -> str:
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else NOT_MEASURED

    for f in result_files:
        b = json.loads(Path(f).read_text(encoding="utf-8"))
        ev: dict[str, Any] = {}
        ef = (eval_files or {}).get(Path(f).stem)
        if ef and Path(ef).is_file():
            ev = json.loads(Path(ef).read_text(encoding="utf-8"))
        lat = b.get("latency_wall_ms", {})
        rows.append(
            f"| {Path(f).stem} | {fmt(b.get('model_size_mb'))} | {fmt(lat.get('median_ms'))} | "
            f"{fmt(lat.get('p95_ms'))} | {fmt(b.get('throughput_fps'))} | "
            f"{fmt(b.get('peak_rss_mb_process'), 1)} | {fmt(ev.get('map50'), 3)} | "
            f"{fmt(ev.get('map50_95'), 3)} | {fmt(ev.get('precision'), 3)} | "
            f"{fmt(ev.get('recall'), 3)} |"
        )
    return "\n".join(rows)
