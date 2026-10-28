import numpy as np
import pytest

from shelfsight.analytics.counting import FrameCounter, LineCrossingCounter, count_mae


def test_frame_counter_separates_detections_from_unique_tracks():
    fc = FrameCounter()
    for _ in range(10):  # same 2 objects visible for 10 frames
        fc.update(2, np.array([1, 2]))
    s = fc.summary()
    assert s.detections_total == 20  # detections accumulate every frame
    assert s.unique_tracks == 2  # but only 2 distinct tracks
    assert s.mean_detections_per_frame == 2 and s.max_detections_in_frame == 2


def test_frame_counter_empty():
    s = FrameCounter().summary()
    assert s.frames == 0 and s.mean_detections_per_frame == 0.0


def test_line_crossing_counts_each_id_once():
    c = LineCrossingCounter((0.5, 0.0), (0.5, 1.0))  # vertical line x=0.5
    shape = (100, 100)
    for x in (10, 30, 45, 55, 70, 55, 45, 60):  # crosses back and forth
        n = c.update(np.array([[x - 5, 40, x + 5, 60.0]]), np.array([7]), shape)
    assert n == 1
    n = c.update(np.array([[0, 40, 10, 60.0]]), np.array([8]), shape)  # id 8 never crosses
    assert n == 1


def test_line_crossing_validation():
    with pytest.raises(ValueError):
        LineCrossingCounter((0.1, 0.1), (0.1, 0.1))


def test_count_mae():
    assert count_mae([10, 20], [12, 17]) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        count_mae([1], [1, 2])
    with pytest.raises(ValueError):
        count_mae([], [])
