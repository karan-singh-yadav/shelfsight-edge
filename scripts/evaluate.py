"""Evaluate one model on a YOLO-format split (same settings for every model).

  python scripts/evaluate.py --model models/best.onnx --images data/yolo/images/test \
      --labels data/yolo/labels/test --out reports/benchmarks/onnx_fp32_eval.json [--agnostic]
Add --failures-dir to also write the failure-analysis CSV and overlay images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from shelfsight.errors import ShelfSightError
from shelfsight.evaluation.evaluate import evaluate_detector
from shelfsight.evaluation.failure_analysis import failure_report
from shelfsight.inference.factory import create_detector
from shelfsight.system_info import collect_system_info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--backend", choices=["pytorch", "onnx"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=1000)
    p.add_argument("--pr-conf", type=float, default=0.25)
    p.add_argument("--eval-conf", type=float, default=0.001)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--agnostic", action="store_true", help="collapse all classes to one")
    p.add_argument("--max-images", type=int)
    p.add_argument("--split-name", default="unspecified", help="recorded in the output")
    p.add_argument("--failures-dir")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    backend = a.backend or ("onnx" if a.model.endswith(".onnx") else "pytorch")
    try:
        det = create_detector(backend, a.model, a.imgsz, a.pr_conf, a.iou, a.max_det, a.threads)
        result, records = evaluate_detector(
            det, a.images, a.labels, a.agnostic, a.pr_conf, a.eval_conf, a.max_images
        )
    except ShelfSightError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    h = hashlib.sha256(Path(a.model).read_bytes()).hexdigest()
    result.update(model=a.model, model_sha256=h, backend=backend, split=a.split_name,
                  images_dir=a.images, system=collect_system_info())  # fmt: skip
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{det.name}: mAP50={result['map50']:.3f} mAP50-95={result['map50_95']:.3f} "
        f"P={result['precision']:.3f} R={result['recall']:.3f} count_MAE={result['count_mae']:.2f} "
        f"({result['n_images']} images)"
    )
    if a.failures_dir:
        failure_report(records, a.failures_dir, a.pr_conf, images_dir=a.images)
        print(f"Failure analysis written to {a.failures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
