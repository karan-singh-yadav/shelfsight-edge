import json

import pytest

from helpers import dets
from shelfsight.analytics.shelf_regions import (
    RegionError,
    ShelfRegion,
    analyze_regions,
    load_regions,
)

SHAPE = (100, 200)  # h, w


def test_region_validation():
    with pytest.raises(RegionError):
        ShelfRegion("a", 0.5, 0, 0.5, 1)
    with pytest.raises(RegionError):
        ShelfRegion("a", 0, 0, 1.2, 1)
    with pytest.raises(RegionError):
        ShelfRegion("", 0, 0, 1, 1)
    with pytest.raises(RegionError):
        ShelfRegion("a", 0, 0, 1, 1, low_threshold=2)
    with pytest.raises(RegionError, match="unique"):
        analyze_regions(dets([]), [ShelfRegion("a", 0, 0, 1, 1)] * 2, SHAPE)


def test_assignment_by_center_and_counts():
    regions = [ShelfRegion("left", 0, 0, 0.5, 1), ShelfRegion("right", 0.5, 0, 1, 1)]
    d = dets([[10, 10, 30, 30], [20, 40, 40, 60], [150, 10, 170, 30]])
    left, right = analyze_regions(d, regions, SHAPE)
    assert (left.count, right.count) == (2, 1)
    assert left.class_counts == {"object": 2}


def test_overlapping_boxes_use_union_not_sum():
    region = [ShelfRegion("all", 0, 0, 1, 1)]
    one = analyze_regions(dets([[0, 0, 100, 100]]), region, SHAPE)[0]
    dup = analyze_regions(dets([[0, 0, 100, 100]] * 3), region, SHAPE)[0]
    assert one.occupancy_ratio == pytest.approx(0.5, abs=0.02)  # 100x100 of 200x100
    assert dup.occupancy_ratio == pytest.approx(one.occupancy_ratio)
    assert dup.count == 3


def test_full_coverage_and_low_occupancy_flag():
    region = [ShelfRegion("all", 0, 0, 1, 1)]
    full = analyze_regions(dets([[0, 0, 200, 100]]), region, SHAPE)[0]
    assert full.occupancy_ratio == pytest.approx(1.0) and not full.low_occupancy
    sparse = analyze_regions(dets([[0, 0, 10, 10]]), region, SHAPE, default_low_threshold=0.2)[0]
    assert sparse.low_occupancy
    custom = ShelfRegion("x", 0, 0, 1, 1, low_threshold=0.0)
    assert not analyze_regions(dets([]), [custom], SHAPE)[0].low_occupancy


def test_empty_detections_and_outside_detections_ignored():
    r = [ShelfRegion("top", 0, 0, 1, 0.5)]
    empty = analyze_regions(dets([]), r, SHAPE)[0]
    assert empty.count == 0 and empty.occupancy_ratio == 0.0 and empty.low_occupancy
    outside = analyze_regions(dets([[10, 80, 30, 95]]), r, SHAPE)[0]
    assert outside.count == 0


def test_first_region_wins_when_overlapping():
    regions = [ShelfRegion("a", 0, 0, 1, 1), ShelfRegion("b", 0, 0, 1, 1)]
    a, b = analyze_regions(dets([[10, 10, 30, 30]]), regions, SHAPE)
    assert (a.count, b.count) == (1, 0)


def test_load_regions_yaml_json_and_errors(tmp_path):
    y = tmp_path / "r.yaml"
    y.write_text("regions:\n  - {name: s1, x1: 0, y1: 0, x2: 1, y2: 0.5}\n", encoding="utf-8")
    assert load_regions(y)[0].name == "s1"
    j = tmp_path / "r.json"
    j.write_text(json.dumps([{"name": "s", "x1": 0, "y1": 0, "x2": 1, "y2": 1}]), encoding="utf-8")
    assert len(load_regions(j)) == 1
    bad = tmp_path / "bad.yaml"
    bad.write_text("regions: []\n", encoding="utf-8")
    with pytest.raises(RegionError):
        load_regions(bad)
    typo = tmp_path / "t.yaml"
    typo.write_text("- {name: a, x1: 0, y1: 0, x2: 1, y2: 1, bogus: 1}\n", encoding="utf-8")
    with pytest.raises(RegionError, match="invalid region"):
        load_regions(typo)
    with pytest.raises(RegionError, match="not found"):
        load_regions(tmp_path / "none.yaml")
    shipped = load_regions("configs/regions.example.yaml")
    assert [r.name for r in shipped] == ["top_shelf", "middle_shelf", "bottom_shelf"]
