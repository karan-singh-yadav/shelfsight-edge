"""ONNX Runtime CPU backend for YOLOv8-style detection models."""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np

from shelfsight.errors import ModelNotFoundError
from shelfsight.inference.base import Detections, Detector, validate_image
from shelfsight.inference.postprocess import decode_yolov8, preprocess_bgr


def _parse_names(meta: dict[str, str]) -> tuple[str, ...]:
    raw = meta.get("names")
    if not raw:
        return ()
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return ()
    if isinstance(parsed, dict):
        return tuple(str(parsed[k]) for k in sorted(parsed))
    if isinstance(parsed, (list, tuple)):
        return tuple(str(v) for v in parsed)
    return ()


class OnnxDetector(Detector):
    """Runs an exported ONNX model with ``CPUExecutionProvider`` only."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: tuple[str, ...] | None = None,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 1000,
        num_threads: int = 0,
    ) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.is_file():
            raise ModelNotFoundError(f"ONNX model not found: {path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads > 0:
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = 1
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.load_time_ms = (time.perf_counter() - t0) * 1000.0
        self.model_path = path
        self.num_threads = num_threads
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape  # [N, 3, H, W], may be symbolic
        static_hw = shape[2] if isinstance(shape[2], int) else None
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        names = tuple(class_names) if class_names else _parse_names(meta)
        super().__init__(
            imgsz=static_hw or imgsz,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            class_names=names,
        )
        self.name = f"onnx:{path.name}"

    def predict(self, image_bgr: np.ndarray) -> Detections:
        validate_image(image_bgr)
        t0 = time.perf_counter()
        tensor, ratio, pad = preprocess_bgr(image_bgr, self.imgsz)
        t1 = time.perf_counter()
        output = self.session.run(None, {self.input_name: tensor})[0]
        t2 = time.perf_counter()
        det = decode_yolov8(
            output,
            self.conf_threshold,
            self.iou_threshold,
            self.max_det,
            ratio,
            pad,
            image_bgr.shape[:2],
            self.class_names,
        )
        t3 = time.perf_counter()
        self.last_timings.preprocess_ms = (t1 - t0) * 1000.0
        self.last_timings.inference_ms = (t2 - t1) * 1000.0
        self.last_timings.postprocess_ms = (t3 - t2) * 1000.0
        return det
