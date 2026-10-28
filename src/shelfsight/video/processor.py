"""Streaming video processing: decode -> detect -> track -> analytics -> render -> write.

Frames are read one at a time (the video is never loaded into RAM). Reported FPS figures say
exactly which stages they include.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2

from shelfsight.analytics.counting import FrameCounter, LineCrossingCounter
from shelfsight.analytics.reporting import AnalyticsRecorder
from shelfsight.analytics.shelf_regions import ShelfRegion, analyze_regions
from shelfsight.drawing import draw_detections, draw_regions
from shelfsight.errors import VideoError
from shelfsight.inference.base import Detections, Detector
from shelfsight.tracking.tracker import ByteTracker

logger = logging.getLogger(__name__)


@dataclass
class VideoStats:
    input_path: str
    output_path: str | None
    width: int
    height: int
    source_fps: float
    frame_stride: int
    frames_read: int = 0
    frames_processed: int = 0
    wall_time_s: float = 0.0
    stage_time_s: dict[str, float] = field(default_factory=dict)
    detections_total: int = 0
    mean_detections_per_frame: float = 0.0
    unique_tracks: int | None = None
    line_crossings: int | None = None

    @property
    def end_to_end_fps(self) -> float:
        """Processed frames / wall time (decode, detect, track, analytics, render, write)."""
        return self.frames_processed / self.wall_time_s if self.wall_time_s > 0 else 0.0

    @property
    def detect_only_fps(self) -> float:
        """Processed frames / time in detector.predict only (pre+inference+post)."""
        t = self.stage_time_s.get("detect", 0.0)
        return self.frames_processed / t if t > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["end_to_end_fps"] = self.end_to_end_fps
        d["detect_only_fps"] = self.detect_only_fps
        return d


def process_video(
    input_path: str | Path,
    detector: Detector,
    output_path: str | Path | None = None,
    tracker: ByteTracker | None = None,
    regions: list[ShelfRegion] | None = None,
    low_occupancy_threshold: float = 0.2,
    frame_stride: int = 1,
    max_frames: int | None = None,
    recorder: AnalyticsRecorder | None = None,
    line: tuple[tuple[float, float], tuple[float, float]] | None = None,
    progress: Callable[[int, int], None] | None = None,
    fourcc: str = "mp4v",
) -> VideoStats:
    """Process a video file and optionally write an annotated copy.

    Only every ``frame_stride``-th frame is processed; the output video contains just those
    frames at ``source_fps / frame_stride``. The tracker only sees processed frames, so a large
    stride lowers tracking quality. ``line`` (two normalized points) enables crossing counts
    and requires a tracker.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if line is not None and tracker is None:
        raise ValueError("line counting requires a tracker")
    src = Path(input_path)
    if not src.is_file():
        raise VideoError(f"Video file not found: {src}")
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise VideoError(f"Cannot open video (unsupported codec or corrupt file): {src}")
    writer: cv2.VideoWriter | None = None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and math.isfinite(fps) and fps > 0 else 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if width <= 0 or height <= 0:
            raise VideoError(f"Video reports invalid frame size: {src}")
        stats = VideoStats(
            str(src), str(output_path) if output_path else None, width, height, fps, frame_stride
        )
        stage = dict.fromkeys(("decode", "detect", "track", "analytics", "render", "write"), 0.0)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*fourcc),
                fps / frame_stride,
                (width, height),
            )
            if not writer.isOpened():
                raise VideoError(f"Cannot open video writer for {output_path} (fourcc={fourcc})")
        counter = FrameCounter()
        crossing = LineCrossingCounter(*line) if line else None
        t_start = time.perf_counter()
        idx = -1
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            stage["decode"] += time.perf_counter() - t0
            if not ok:
                break
            idx += 1
            stats.frames_read += 1
            if idx % frame_stride:
                continue
            if max_frames is not None and stats.frames_processed >= max_frames:
                break

            t0 = time.perf_counter()
            det = detector.predict(frame)
            stage["detect"] += time.perf_counter() - t0

            track_ids = None
            shown: Detections = det
            if tracker is not None:
                t0 = time.perf_counter()
                tracked = tracker.update(det)
                stage["track"] += time.perf_counter() - t0
                shown, track_ids = tracked.detections, tracked.track_ids
                if crossing is not None:
                    crossing.update(shown.boxes, track_ids, frame.shape[:2])

            t0 = time.perf_counter()
            region_stats = (
                analyze_regions(shown, regions, frame.shape[:2], low_occupancy_threshold)
                if regions
                else None
            )
            counter.update(len(det), track_ids)
            if recorder is not None:
                recorder.add_frame(
                    idx,
                    idx / fps,
                    len(det),
                    len(shown) if tracker is not None else None,
                    region_stats,
                )
            stage["analytics"] += time.perf_counter() - t0

            if writer is not None:
                t0 = time.perf_counter()
                canvas = draw_detections(frame, shown, track_ids, show_conf=False, thickness=1)
                if regions and region_stats:
                    canvas = draw_regions(
                        canvas,
                        [(r.name, r.to_pixels(frame.shape[:2])) for r in regions],
                        {s.name: s.low_occupancy for s in region_stats},
                        {
                            s.name: f"{s.name}: {s.count} ({s.occupancy_ratio:.0%})"
                            for s in region_stats
                        },
                    )
                stage["render"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                writer.write(canvas)
                stage["write"] += time.perf_counter() - t0
            stats.frames_processed += 1
            if progress is not None:
                progress(idx + 1, total)
        stats.wall_time_s = time.perf_counter() - t_start
        stats.stage_time_s = stage
        summary = counter.summary(len(crossing.crossed) if crossing else None)
        stats.detections_total = summary.detections_total
        stats.mean_detections_per_frame = summary.mean_detections_per_frame
        stats.unique_tracks = tracker.unique_tracks if tracker is not None else None
        stats.line_crossings = summary.line_crossings
        if stats.frames_read == 0:
            raise VideoError(f"No frames could be decoded from {src}")
        return stats
    finally:
        cap.release()
        if writer is not None:
            writer.release()
