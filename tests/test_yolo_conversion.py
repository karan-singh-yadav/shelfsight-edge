import pytest

from shelfsight.data.convert import convert_split, read_yolo_labels, write_dataset_yaml
from shelfsight.data.formats import (
    DatasetError,
    clip_box,
    load_annotations,
    select_subset,
    xyxy_to_yolo,
    yolo_to_xyxy,
)


def test_known_example():
    # 400x200 image, box x 100..300, y 50..150 -> centered, half width, half height
    assert xyxy_to_yolo(100, 50, 300, 150, 400, 200) == (0.5, 0.5, 0.5, 0.5)
    assert xyxy_to_yolo(0, 0, 40, 20, 400, 200) == pytest.approx((0.05, 0.05, 0.1, 0.1))


def test_roundtrip():
    box = (12.0, 30.0, 150.0, 90.0)
    back = yolo_to_xyxy(*xyxy_to_yolo(*box, 640, 480), 640, 480)
    assert back == pytest.approx(box)


def test_clip_box():
    assert clip_box(-5, -5, 700, 500, 640, 480) == (0, 0, 640, 480)


def test_xyxy_to_yolo_rejects_bad_dims():
    with pytest.raises(ValueError):
        xyxy_to_yolo(0, 0, 1, 1, 0, 10)


def test_convert_end_to_end(synth, tmp_path):
    root, csvs = synth
    df = load_annotations(csvs["train"])
    out = tmp_path / "yolo"
    stats = convert_split(df, root / "images", out, "train", ["object"])
    assert stats.images == 4 and stats.boxes_written == len(df)
    assert stats.boxes_dropped_invalid == 0 and stats.boxes_clipped == 0
    labels = read_yolo_labels(out / "labels" / "train" / "train_0.txt")
    first = df[df["image_name"] == "train_0.jpg"].iloc[0]
    xc, yc, bw, bh = labels[0][1:]
    assert yolo_to_xyxy(xc, yc, bw, bh, first.image_width, first.image_height) == pytest.approx(
        (first.x1, first.y1, first.x2, first.y2), abs=1e-3
    )
    assert (out / "images" / "train" / "train_0.jpg").is_file()
    yaml_path = write_dataset_yaml(out, ["object"], ["train"])
    text = yaml_path.read_text()
    assert "names" in text
    assert "path:" not in text and str(tmp_path) not in text  # relocatable


def test_convert_reports_dropped_and_clipped(synth, tmp_path):
    root, csvs = synth
    df = load_annotations(csvs["train"])
    df.loc[0, "x2"] = df.loc[0, "image_width"] + 20  # clipped, kept
    df.loc[1, "x2"] = df.loc[1, "x1"]  # degenerate -> dropped
    df.loc[2, "y1"] = float("nan")  # invalid -> dropped
    stats = convert_split(df, root / "images", tmp_path / "o", "train", ["object"])
    assert stats.boxes_clipped == 1
    assert stats.boxes_dropped_invalid == 2
    assert stats.boxes_written == len(df) - 2


def test_convert_errors(synth, tmp_path):
    root, csvs = synth
    df = load_annotations(csvs["train"])
    bad_cls = df.copy()
    bad_cls.loc[0, "class"] = "mystery"
    with pytest.raises(DatasetError, match="mapping"):
        convert_split(bad_cls, root / "images", tmp_path / "a", "train", ["object"])
    (root / "images" / "train_0.jpg").unlink()
    with pytest.raises(DatasetError, match="Missing image"):
        convert_split(df, root / "images", tmp_path / "b", "train", ["object"])
    stats = convert_split(df, root / "images", tmp_path / "c", "train", ["object"], strict=False)
    assert stats.images_missing == 1 and stats.images == 3


def test_subset_is_deterministic(synth):
    _, csvs = synth
    df = load_annotations(csvs["train"])
    a, b = select_subset(df, 2, seed=7), select_subset(df, 2, seed=7)
    assert a["image_name"].nunique() == 2
    assert list(a["image_name"]) == list(b["image_name"])
