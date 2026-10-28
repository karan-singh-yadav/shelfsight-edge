"""Shared test doubles."""

from __future__ import annotations

import numpy as np

from shelfsight.inference.base import Detections, Detector


class ScriptedDetector(Detector):
    """Returns a pre-scripted Detections per call (cycling) so no model is needed."""

    def __init__(self, script: list[Detections], imgsz: int = 64) -> None:
        super().__init__(imgsz=imgsz, class_names=("object",))
        self.script = script
        self.calls = 0
        self.name = "scripted"

    def predict(self, image_bgr: np.ndarray) -> Detections:
        det = self.script[self.calls % len(self.script)]
        self.calls += 1
        self.last_timings.inference_ms = 1.0
        return det


def dets(boxes, scores=None, cls=None) -> Detections:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    n = len(boxes)
    return Detections(
        boxes,
        np.full(n, 0.9) if scores is None else scores,
        np.zeros(n, dtype=np.int64) if cls is None else cls,
        ("object",),
    )
