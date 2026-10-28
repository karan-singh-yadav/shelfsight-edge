import numpy as np

from helpers import dets
from shelfsight.inference.base import Detections
from shelfsight.tracking.tracker import ByteTracker, KalmanBoxFilter


def box_at(x, y=20, size=40):
    return [x, y, x + size, y + size]


def run(tracker, frames):
    out = []
    for f in frames:
        out.append(tracker.update(f))
    return out


def test_kalman_update_moves_toward_measurement():
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate(kf.to_measurement(np.array([0, 0, 40, 40.0])))
    mean, cov = kf.predict(mean, cov)
    new, _ = kf.update(mean, cov, kf.to_measurement(np.array([10, 0, 50, 40.0])))
    assert mean[0] < new[0] <= 30.0


def test_single_moving_object_keeps_one_id():
    t = ByteTracker()
    res = run(t, [dets([box_at(10 + 5 * i)]) for i in range(20)])
    ids = {int(r.track_ids[0]) for r in res if len(r.track_ids)}
    assert len(ids) == 1 and t.unique_tracks == 1
    assert all(len(r.detections) == 1 for r in res)  # first frame confirmed immediately


def test_two_separated_objects_get_two_stable_ids():
    t = ByteTracker()
    res = run(t, [dets([box_at(10 + 3 * i, 10), box_at(200 - 3 * i, 150)]) for i in range(15)])
    assert t.unique_tracks == 2
    first = dict(zip(res[0].track_ids.tolist(), res[0].detections.boxes[:, 0], strict=True))
    last = dict(zip(res[-1].track_ids.tolist(), res[-1].detections.boxes[:, 0], strict=True))
    assert set(first) == set(last)


def test_short_occlusion_recovers_same_id():
    t = ByteTracker(track_buffer=30)
    frames = [dets([box_at(10 + 4 * i)]) for i in range(8)]
    frames += [Detections.empty(("object",))] * 4
    frames += [dets([box_at(10 + 4 * i)]) for i in range(12, 20)]
    run(t, frames)
    assert t.unique_tracks == 1


def test_long_gap_creates_new_id_fragmentation():
    t = ByteTracker(track_buffer=5)
    frames = (
        [dets([box_at(10)])] * 5 + [Detections.empty(("object",))] * 12 + [dets([box_at(10)])] * 5
    )
    run(t, frames)
    assert t.unique_tracks == 2  # documents ID fragmentation after the buffer expires


def test_low_score_detection_extends_existing_track_only():
    t = ByteTracker(high_thresh=0.5, low_thresh=0.1)
    frames = [dets([box_at(10 + 2 * i)]) for i in range(5)]
    frames += [dets([box_at(10 + 2 * i)], scores=np.array([0.3])) for i in range(5, 9)]
    res = run(t, frames)
    assert t.unique_tracks == 1 and len(res[-1].track_ids) == 1
    t2 = ByteTracker()
    run(t2, [Detections.empty(("object",))] + [dets([box_at(10)], scores=np.array([0.3]))] * 5)
    assert t2.unique_tracks == 0  # low-score alone never starts a track


def test_single_frame_false_positive_is_never_confirmed():
    t = ByteTracker()
    frames = [dets([box_at(10)])] * 3
    frames += [dets([box_at(10), box_at(200, 120)])] + [dets([box_at(10)])] * 3
    run(t, frames)
    assert t.unique_tracks == 1


def test_empty_stream_and_reset():
    t = ByteTracker()
    r = t.update(Detections.empty(("object",)))
    assert len(r.detections) == 0 and len(r.track_ids) == 0
    run(t, [dets([box_at(10)])] * 3)
    assert t.unique_tracks == 1
    t.reset()
    assert t.unique_tracks == 0 and t.frame_id == 0
