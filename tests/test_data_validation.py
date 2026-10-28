import pandas as pd
import pytest

from shelfsight.data.formats import DatasetError, load_annotations
from shelfsight.data.validate_dataset import validate_dataset


def rewrite(csv, mutate):
    df = load_annotations(csv)
    mutate(df)
    df.to_csv(csv, index=False, header=False)


def test_clean_synthetic_passes(synth):
    root, csvs = synth
    report = validate_dataset(csvs["train"], root / "images", deep=True)
    assert report.ok and report.n_warnings == 0
    assert report.n_images == 4 and report.n_boxes > 0


def test_loader_accepts_header_and_headerless(synth, tmp_path):
    _, csvs = synth
    df = load_annotations(csvs["train"])
    with_header = tmp_path / "h.csv"
    df.to_csv(with_header, index=False)
    assert len(load_annotations(with_header)) == len(df)


def test_loader_rejects_wrong_columns_and_missing_file(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,1,2\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="columns"):
        load_annotations(bad)
    with pytest.raises(DatasetError, match="not found"):
        load_annotations(tmp_path / "nope.csv")
    empty = tmp_path / "e.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match="empty"):
        load_annotations(empty)


def test_detects_box_problems(synth):
    root, csvs = synth

    def mutate(df):
        df.loc[0, "x2"] = df.loc[0, "x1"]  # degenerate
        df.loc[1, "x1"] = float("nan")  # non-finite
        df.loc[2, "x2"] = df.loc[2, "image_width"] + 50  # out of bounds
        df.loc[3] = df.loc[4]  # duplicate row

    rewrite(csvs["train"], mutate)
    r = validate_dataset(csvs["train"], root / "images")
    assert r.counts["degenerate_box"] == 1
    assert r.counts["non_finite_value"] == 1
    assert r.counts["box_out_of_bounds"] == 1
    assert r.counts["duplicate_row"] >= 1
    assert not r.ok  # errors present
    assert r.severities["box_out_of_bounds"] == "warning"


def test_detects_image_problems(synth):
    root, csvs = synth
    (root / "images" / "train_0.jpg").unlink()  # missing
    (root / "images" / "train_1.jpg").write_bytes(b"not an image")  # corrupt
    from PIL import Image

    Image.new("RGB", (100, 100)).save(root / "images" / "train_2.jpg")  # wrong size
    r = validate_dataset(csvs["train"], root / "images")
    assert r.counts["missing_image"] == 1
    assert r.counts["corrupt_image"] == 1
    assert r.counts["dimension_mismatch"] == 1
    assert not r.ok


def test_inconsistent_dims_in_csv(synth):
    root, csvs = synth

    def mutate(df):
        df.loc[0, "image_width"] = 999

    rewrite(csvs["train"], mutate)
    r = validate_dataset(csvs["train"], root / "images")
    assert r.counts["inconsistent_dims_in_csv"] == 1


def test_stats_on_synthetic(synth):
    from shelfsight.data.stats import dataset_stats

    _, csvs = synth
    df = load_annotations(csvs["train"])
    s = dataset_stats(df)
    assert s["n_images"] == 4
    assert s["class_counts"] == {"object": len(df)}
    assert s["n_boxes_valid"] == len(df)
    assert 0 < s["box_area_rel"]["max"] < 1
    assert isinstance(pd.Series(s["boxes_per_image"]).mean(), float)


def test_corrupt_image_reported_even_when_pil_open_is_patched(synth, monkeypatch):
    """Regression: Ultralytics patches PIL.Image.open and raises ModuleNotFoundError on junk."""
    from PIL import Image

    root, csvs = synth
    (root / "images" / "train_1.jpg").write_bytes(b"junk")
    real_open = Image.open

    def patched(fp, *a, **k):
        try:
            return real_open(fp, *a, **k)
        except Exception as exc:
            raise ModuleNotFoundError("No module named 'pi_heif'") from exc

    monkeypatch.setattr(Image, "open", patched)
    r = validate_dataset(csvs["train"], root / "images")
    assert r.counts["corrupt_image"] == 1
