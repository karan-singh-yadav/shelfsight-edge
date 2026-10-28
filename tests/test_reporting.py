import pandas as pd

from shelfsight.analytics.reporting import (
    OCCUPANCY_DISCLAIMER,
    AnalyticsRecorder,
    append_history,
    read_history,
)
from shelfsight.analytics.shelf_regions import RegionStats


def test_recorder_csv_roundtrip(tmp_path):
    rec = AnalyticsRecorder()
    rs = [RegionStats("top", 3, 0.4567891, False), RegionStats("bottom", 0, 0.0, True)]
    rec.add_frame(0, 0.0, 3, 3, rs)
    rec.add_frame(5, 0.2, 2, 2, rs)
    paths = rec.save_csv(tmp_path / "out", "run")
    frames = pd.read_csv(paths["frames"])
    regions = pd.read_csv(paths["regions"])
    assert list(frames.columns) == ["frame_index", "timestamp_s", "detections", "active_tracks"]
    assert len(frames) == 2 and len(regions) == 4
    assert regions["occupancy_ratio"].iloc[0] == 0.4568
    assert bool(regions[regions.region == "bottom"]["low_occupancy"].iloc[0])


def test_recorder_without_regions_or_tracks(tmp_path):
    rec = AnalyticsRecorder()
    rec.add_frame(0, 0.0, 1)
    paths = rec.save_csv(tmp_path)
    assert "regions" not in paths and paths["frames"].is_file()
    assert rec.region_dataframe().empty


def test_empty_recorder_writes_header_only(tmp_path):
    p = AnalyticsRecorder().save_csv(tmp_path)["frames"]
    assert p.read_text().strip().startswith("frame_index")


def test_history_append_read_and_tolerates_partial_line(tmp_path):
    path = tmp_path / "h.jsonl"
    assert read_history(path).empty
    append_history(path, {"kind": "image", "n": 3})
    append_history(path, {"kind": "video", "n": 9})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"kind": "tru')  # simulated crash mid-write
    df = read_history(path)
    assert list(df["kind"]) == ["image", "video"] and "time_utc" in df.columns


def test_disclaimer_present():
    assert "ESTIMATE" in OCCUPANCY_DISCLAIMER and "not an audited" in OCCUPANCY_DISCLAIMER
