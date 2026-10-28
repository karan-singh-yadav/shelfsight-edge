"""Static INT8 quantization of an FP32 ONNX model with ONNX Runtime (QDQ format).

Calibration uses images from the TRAIN split only (never val/test). The preprocessing is
identical to inference. INT8 is not assumed faster or accurate: always benchmark and
evaluate the result with scripts/benchmark.py and scripts/evaluate.py.

  python scripts/quantize.py --model models/best.onnx --calib-dir data/yolo/images/train \
      --out models/best_int8.onnx --num-calib 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from shelfsight.data.formats import IMAGE_EXTENSIONS
from shelfsight.inference.base import load_image
from shelfsight.inference.postprocess import preprocess_bgr


class ImageCalibrationReader:
    """onnxruntime CalibrationDataReader over letterboxed training images."""

    def __init__(self, files: list[Path], input_name: str, imgsz: int) -> None:
        self.files, self.input_name, self.imgsz = files, input_name, imgsz
        self._it = iter(self.files)

    def get_next(self) -> dict[str, np.ndarray] | None:
        for path in self._it:
            try:
                tensor, _, _ = preprocess_bgr(load_image(path), self.imgsz)
            except Exception as exc:  # noqa: BLE001
                print(f"skip calibration image {path.name}: {exc}", file=sys.stderr)
                continue
            return {self.input_name: tensor}
        return None

    def rewind(self) -> None:
        self._it = iter(self.files)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--calib-dir", required=True, help="TRAIN images only")
    p.add_argument("--out", required=True)
    p.add_argument("--num-calib", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", choices=["minmax", "entropy", "percentile"], default="minmax")
    p.add_argument("--per-channel", action="store_true")
    p.add_argument("--exclude-nodes", nargs="*", default=[], help="node names to keep in FP32")
    p.add_argument(
        "--exclude-prefix", nargs="*", default=[],
        help="keep every node whose name starts with this prefix in FP32, e.g. /model.22",
    )  # fmt: skip
    p.add_argument(
        "--exclude-head", action="store_true",
        help="auto-detect the last /model.N/ block (the Detect head of Ultralytics exports) "
        "and keep it in FP32; YOLO heads are quantization-sensitive",
    )  # fmt: skip
    a = p.parse_args()

    calib_dir = Path(a.calib_dir)
    if "test" in {part.lower() for part in calib_dir.parts}:
        print("ERROR: refusing to calibrate on a path containing 'test'.", file=sys.stderr)
        return 2
    files = (
        sorted(f for f in calib_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        if calib_dir.is_dir()
        else []
    )
    if not files:
        print(f"ERROR: no calibration images in {calib_dir}", file=sys.stderr)
        return 2
    rng = np.random.default_rng(a.seed)
    if len(files) > a.num_calib:
        files = [files[i] for i in sorted(rng.choice(len(files), a.num_calib, replace=False))]

    import re

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    node_names = [n.name for n in onnx.load(a.model).graph.node]
    prefixes = list(a.exclude_prefix)
    if a.exclude_head:
        blocks = {int(m.group(1)) for n in node_names if (m := re.match(r"^/model\.(\d+)/", n))}
        if not blocks:
            print(
                "ERROR: --exclude-head found no '/model.N/' nodes in this graph.", file=sys.stderr
            )
            return 2
        prefixes.append(f"/model.{max(blocks)}/")
    excluded = sorted(
        set(a.exclude_nodes) | {n for n in node_names if n.startswith(tuple(prefixes))}
    )
    if prefixes:
        print(f"Keeping {len(excluded)} node(s) in FP32 (prefixes {prefixes})")

    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    imgsz = inp.shape[2] if isinstance(inp.shape[2], int) else 640
    reader = ImageCalibrationReader(files, inp.name, imgsz)
    method = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }[a.method]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        a.model, a.out, reader, quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        per_channel=a.per_channel, calibrate_method=method, nodes_to_exclude=excluded,
    )  # fmt: skip
    src, dst = Path(a.model).stat().st_size, Path(a.out).stat().st_size
    print(f"Quantized with {len(files)} calibration images ({a.method}): {a.out}")
    print(f"size FP32 {src / 1024**2:.2f} MB -> INT8 {dst / 1024**2:.2f} MB")
    print("Next: benchmark and evaluate it; if accuracy drops or it is slower, keep FP32.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
