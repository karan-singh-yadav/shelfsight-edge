"""Export a YOLO .pt checkpoint to ONNX (FP32), validate it, and check consistency vs PyTorch.

python scripts/export_onnx.py --weights models/yolov8n.pt --imgsz 640 --check-image path.jpg
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--out", help="output .onnx path (default: next to the weights)")
    p.add_argument("--no-simplify", action="store_true")
    p.add_argument("--check-image", help="image used to compare PyTorch vs ONNX detections")
    a = p.parse_args()

    w = Path(a.weights)
    if not w.is_file():
        print(f"ERROR: weights not found: {w}", file=sys.stderr)
        return 2
    from ultralytics import YOLO

    exported = Path(
        YOLO(str(w)).export(
            format="onnx",
            imgsz=a.imgsz,
            opset=a.opset,
            simplify=not a.no_simplify,
            dynamic=False,
            half=False,
            device="cpu",
        )  # fmt: skip
    )
    target = Path(a.out) if a.out else exported
    if target != exported:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), str(target))
    import onnx

    onnx.checker.check_model(onnx.load(str(target)))
    print(f"ONNX model valid (onnx.checker): {target} ({target.stat().st_size / 1024**2:.2f} MB)")

    if a.check_image:
        from shelfsight.evaluation.benchmark import compare_detections
        from shelfsight.inference.base import load_image
        from shelfsight.inference.factory import create_detector

        img = load_image(a.check_image)
        d_pt = create_detector("pytorch", w, a.imgsz)
        d_ox = create_detector("onnx", target, a.imgsz)
        r_pt, r_ox = d_pt.predict(img), d_ox.predict(img)
        cmp = compare_detections(r_pt, r_ox)
        print(f"PyTorch: {len(r_pt)} dets | ONNX: {len(r_ox)} dets | consistency: {cmp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
