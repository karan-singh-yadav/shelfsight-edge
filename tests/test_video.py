import cv2
import numpy as np
import pytest

from helpers import ScriptedDetector, dets
from shelfsight.analytics.reporting import AnalyticsRecorder
from shelfsight.analytics.shelf_regions import ShelfRegion
from shelfsight.errors import VideoError
from shelfsight.tracking.tracker import ByteTracker
from shelfsight.video.processor import process_video

W, H, N = 160, 120, 12


def square_x(i):
    return 10 + 8 * i


def make_video(path, n=N, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    assert writer.isOpened(), "MJPG AVI writer unavailable in this environment"
    for i in range(n):
        frame = np.zeros((H, W, 3), np.uint8)
        x = square_x(i)
        cv2.rectangle(frame, (x, 40), (x + 20, 60), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def moving_script(n=N, stride=1):
    return [dets([[square_x(i), 40, square_x(i) + 20, 60]]) for i in range(0, n, stride)]


def frame_count(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / "in.avi")


def test_process_video_tracking_and_output(video, tmp_path):
    out = tmp_path / "out" / "o.avi"
    det = ScriptedDetector(moving_script())
    stats = process_video(video, det, out, ByteTracker(), fourcc="MJPG")
    assert stats.frames_read == N and stats.frames_processed == N
    assert stats.unique_tracks == 1  # one object seen in 12 frames
    assert stats.detections_total == N  # but 12 per-frame detections: not interchangeable
    assert frame_count(out) == N
    assert stats.end_to_end_fps > 0 and stats.detect_only_fps >= stats.end_to_end_fps
    assert set(stats.stage_time_s) == {"decode", "detect", "track", "analytics", "render", "write"}
    assert stats.to_dict()["frame_stride"] == 1


def test_frame_stride_and_max_frames(video, tmp_path):
    det = ScriptedDetector(moving_script(stride=3))
    stats = process_video(
        video, det, tmp_path / "s.avi", ByteTracker(), frame_stride=3, fourcc="MJPG"
    )
    assert stats.frames_read == N and stats.frames_processed == 4 and det.calls == 4
    assert frame_count(tmp_path / "s.avi") == 4
    capped = process_video(video, ScriptedDetector(moving_script()), max_frames=5)
    assert capped.frames_processed == 5


def test_without_tracker_or_output(video):
    stats = process_video(video, ScriptedDetector(moving_script()))
    assert stats.unique_tracks is None and stats.output_path is None
    assert stats.stage_time_s["track"] == 0.0


def test_regions_and_recorder(video):
    regions = [ShelfRegion("has_item", 0, 0.25, 1, 0.75), ShelfRegion("empty", 0, 0.8, 1, 1)]
    rec = AnalyticsRecorder()
    process_video(video, ScriptedDetector(moving_script()), regions=regions, recorder=rec,
                  low_occupancy_threshold=0.05)  # fmt: skip
    df = rec.region_dataframe()
    assert len(df) == 2 * N and len(rec.frame_dataframe()) == N
    assert df[df.region == "empty"]["low_occupancy"].all()
    assert (df[df.region == "has_item"]["count"] == 1).all()
    assert rec.frame_dataframe()["timestamp_s"].iloc[3] == pytest.approx(0.3)


def test_line_crossing(video):
    stats = process_video(video, ScriptedDetector(moving_script()), tracker=ByteTracker(),
                          line=((0.5, 0.0), (0.5, 1.0)))  # fmt: skip
    assert stats.line_crossings == 1  # centre x goes 20 -> 108, crossing x=80 once


def test_errors(tmp_path, video):
    det = ScriptedDetector(moving_script())
    with pytest.raises(VideoError, match="not found"):
        process_video(tmp_path / "missing.avi", det)
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(VideoError):
        process_video(bad, det)
    with pytest.raises(ValueError):
        process_video(video, det, frame_stride=0)
    with pytest.raises(ValueError, match="requires a tracker"):
        process_video(video, det, line=((0, 0), (1, 1)))
    with pytest.raises(VideoError, match="writer"):
        process_video(video, det, tmp_path / "x.avi", fourcc="ZZZZ")


def test_zero_frame_video(tmp_path):
    p = tmp_path / "zero.avi"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10, (W, H))
    w.release()
    with pytest.raises(VideoError):
        process_video(p, ScriptedDetector([dets([])]))
