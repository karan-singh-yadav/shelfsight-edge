"""Create a detector from a backend name so callers never branch on backend type."""

from __future__ import annotations

from pathlib import Path

from shelfsight.inference.base import Detector

BACKENDS = ("pytorch", "onnx")


def create_detector(
    backend: str,
    model_path: str | Path,
    imgsz: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 1000,
    num_threads: int = 0,
    device: str = "cpu",
) -> Detector:
    """Instantiate ``pytorch`` or ``onnx`` backend. Heavy imports happen lazily here."""
    common = dict(
        imgsz=imgsz,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_det=max_det,
        num_threads=num_threads,
    )
    if backend == "onnx":
        from shelfsight.inference.onnx_backend import OnnxDetector

        return OnnxDetector(model_path, **common)
    if backend == "pytorch":
        from shelfsight.inference.pytorch_backend import PyTorchDetector

        return PyTorchDetector(model_path, device=device, **common)
    raise ValueError(f"Unknown backend '{backend}'; choose from {BACKENDS}")
