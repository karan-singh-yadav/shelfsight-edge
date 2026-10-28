"""Common detector interface and the normalized detection schema.

All backends return :class:`Detections` (pixel-space xyxy in the ORIGINAL image frame), so
tracking, analytics, evaluation and the UI never depend on a specific backend.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from shelfsight.errors import InvalidImageError


@dataclass
class StageTimings:
    """Milliseconds spent in each stage of the most recent ``predict`` call."""

    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass
class Detections:
    """Detections for one image.

    ``boxes`` (N,4) xyxy pixels, ``scores`` (N,), ``class_ids`` (N,).
    """

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    class_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        self.class_ids = np.asarray(self.class_ids, dtype=np.int64).reshape(-1)
        n = len(self.boxes)
        if len(self.scores) != n or len(self.class_ids) != n:
            raise ValueError("boxes, scores and class_ids must have the same length")

    def __len__(self) -> int:
        return len(self.boxes)

    @classmethod
    def empty(cls, class_names: tuple[str, ...] = ()) -> Detections:
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64), class_names)

    def name_of(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def filter(self, min_score: float) -> Detections:
        keep = self.scores >= min_score
        return Detections(
            self.boxes[keep], self.scores[keep], self.class_ids[keep], self.class_names
        )

    def counts(self) -> dict[str, int]:
        """Number of detections per class name (detections, NOT unique products)."""
        out: dict[str, int] = {}
        for cid in self.class_ids:
            name = self.name_of(int(cid))
            out[name] = out.get(name, 0) + 1
        return out

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x1": self.boxes[:, 0],
                "y1": self.boxes[:, 1],
                "x2": self.boxes[:, 2],
                "y2": self.boxes[:, 3],
                "score": self.scores,
                "class_id": self.class_ids,
                "class_name": [self.name_of(int(c)) for c in self.class_ids],
            }
        )


def validate_image(image: np.ndarray) -> None:
    """Raise ``InvalidImageError`` unless ``image`` is a non-empty HxWx3 uint8 array (BGR)."""
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"image must be a numpy array, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise InvalidImageError(
            f"expected HxWx3 uint8 BGR image, got shape={image.shape} dtype={image.dtype}"
        )
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise InvalidImageError(f"image too small: {image.shape[:2]}")


def load_image(path: str | Path) -> np.ndarray:
    """Decode an image file to BGR. Handles non-ASCII Windows paths."""
    p = Path(path)
    if not p.is_file():
        raise InvalidImageError(f"Image file not found: {p}")
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        if data.size == 0:
            raise InvalidImageError(f"Image file is empty: {p}")
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as exc:
        raise InvalidImageError(f"Cannot read {p}: {exc}") from exc
    if img is None:
        raise InvalidImageError(f"File is not a decodable image: {p}")
    return img


@dataclass
class Detector(ABC):
    """Backend-agnostic detector. Subclasses implement :meth:`predict`."""

    imgsz: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 1000
    class_names: tuple[str, ...] = ()
    last_timings: StageTimings = field(default_factory=StageTimings)

    name: str = "detector"

    @abstractmethod
    def predict(self, image_bgr: np.ndarray) -> Detections:
        """Run detection on one BGR uint8 image and update ``last_timings``."""

    def predict_path(self, path: str | Path) -> Detections:
        return self.predict(load_image(path))

    def warmup(self, runs: int = 3) -> float:
        """Run ``runs`` dummy inferences; returns total warm-up time in ms."""
        dummy = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(runs):
            self.predict(dummy)
        return (time.perf_counter() - t0) * 1000.0

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (optional)."""
