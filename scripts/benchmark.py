"""CPU benchmark for ONE model per process.

  python scripts/benchmark.py --model models/yolov8n.pt --images some_dir_or_file \
      --threads 4 --warmup 10 --runs 100 --out reports/benchmarks/pytorch_fp32.json
Settings default to configs/benchmark.yaml when flags are omitted.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from shelfsight.config import load_yaml
from shelfsight.data.formats import IMAGE_EXTENSIONS
from shelfsight.errors import ShelfSightError
from shelfsight.evaluation.benchmark import benchmark_detector, save_benchmark
from shelfsight.inference.base import load_image
from shelfsight.inference.factory import create_detector


def collect_images(path: Path, limit: int) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)[:limit]


def main() -> int:
    cfg = {}
    cfg_path = Path("configs/benchmark.yaml")
    if cfg_path.is_file():
        cfg = load_yaml(cfg_path)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--images", required=True, help="image file or directory")
    p.add_argument("--backend", choices=["pytorch", "onnx"])
    p.add_argument("--threads", type=int, default=cfg.get("num_threads", 4))
    p.add_argument("--warmup", type=int, default=cfg.get("warmup_runs", 10))
    p.add_argument("--runs", type=int, default=cfg.get("timed_runs", 100))
    p.add_argument("--imgsz", type=int, default=cfg.get("imgsz", 640))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-images", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    backend = a.backend or ("onnx" if a.model.endswith(".onnx") else "pytorch")
    try:
        t0 = time.perf_counter()
        det = create_detector(backend, a.model, a.imgsz, a.conf, num_threads=a.threads)
        load_ms = (time.perf_counter() - t0) * 1000.0
        files = collect_images(Path(a.images), a.max_images)
        if not files:
            print(f"ERROR: no images at {a.images}", file=sys.stderr)
            return 2
        images = [load_image(f) for f in files]
        result = benchmark_detector(det, images, a.warmup, a.runs, a.model, load_ms)
    except ShelfSightError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    save_benchmark(result, a.out)
    lat = result["latency_wall_ms"]
    print(
        f"{result['detector']}: median {lat['median_ms']:.1f} ms | p95 {lat['p95_ms']:.1f} ms | "
        f"{result['throughput_fps']} FPS (batch 1, pre+infer+post) | threads={a.threads}"
    )
    print(f"Saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
