"""Single-image detection: python scripts/detect.py --image x.jpg --model models/yolov8n.pt"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from shelfsight.drawing import draw_detections
from shelfsight.errors import ShelfSightError
from shelfsight.inference.base import load_image
from shelfsight.inference.factory import create_detector
from shelfsight.logging_config import setup_logging


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--backend", choices=["pytorch", "onnx"], default=None, help="default: by suffix"
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-det", type=int, default=1000)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default="reports/figures/detect_out.jpg")
    p.add_argument("--csv", help="optional CSV of detections")
    a = p.parse_args()
    setup_logging("INFO")
    backend = a.backend or ("onnx" if a.model.endswith(".onnx") else "pytorch")
    try:
        det = create_detector(backend, a.model, a.imgsz, a.conf, a.iou, a.max_det, a.threads)
        img = load_image(a.image)
        res = det.predict(img)
    except ShelfSightError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, draw_detections(img, res))
    t = det.last_timings
    print(f"{det.name}: {len(res)} detections {res.counts()}")
    print(
        f"stage ms (single run, includes first-call overhead): pre={t.preprocess_ms:.1f} "
        f"infer={t.inference_ms:.1f} post={t.postprocess_ms:.1f}"
    )
    if a.csv:
        res.to_dataframe().to_csv(a.csv, index=False)
    print(f"Annotated image: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
