"""PyTorch (Ultralytics) baseline backend. Never downloads weights implicitly."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shelfsight.errors import DeviceUnavailableError, ModelNotFoundError
from shelfsight.inference.base import Detections, Detector, validate_image


class PyTorchDetector(Detector):
    """Wraps an Ultralytics YOLO ``.pt`` checkpoint.

    ``rect=False`` (default) letterboxes to a full square ``imgsz`` x ``imgsz`` input, exactly
    like the static-shape ONNX export. Ultralytics' default (rect=True) pads only to a minimal
    rectangle, which changes the network input and therefore the scores; it was measured to
    drop a borderline detection versus ONNX on a test image. Keep ``rect=False`` whenever
    comparing backends.
    """

    def __init__(
        self,
        weights: str | Path,
        device: str = "cpu",
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 1000,
        num_threads: int = 0,
        rect: bool = False,
    ) -> None:
        path = Path(weights)
        if not path.is_file():
            raise ModelNotFoundError(
                f"Weights not found: {path}. Weights are never downloaded implicitly; run "
                "`python scripts/download_weights.py` or train/export your own."
            )
        import torch
        from ultralytics import YOLO

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise DeviceUnavailableError(
                f"Device '{device}' requested but CUDA is not available. Use device='cpu'."
            )
        if num_threads > 0:
            torch.set_num_threads(num_threads)
        self.model = YOLO(str(path))
        self.device = device
        self.rect = rect
        self.model_path = path
        names = self.model.names
        class_names = tuple(str(names[k]) for k in sorted(names))
        super().__init__(
            imgsz=imgsz,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            class_names=class_names,
        )
        self.name = f"pytorch:{path.name}"

    def predict(self, image_bgr: np.ndarray) -> Detections:
        validate_image(image_bgr)
        result = self.model.predict(
            image_bgr,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.device,
            rect=self.rect,
            verbose=False,
        )[0]
        speed = result.speed  # ms: preprocess / inference / postprocess (Ultralytics' own timers)
        self.last_timings.preprocess_ms = float(speed.get("preprocess", 0.0))
        self.last_timings.inference_ms = float(speed.get("inference", 0.0))
        self.last_timings.postprocess_ms = float(speed.get("postprocess", 0.0))
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty(self.class_names)
        return Detections(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(np.int64),
            self.class_names,
        )
