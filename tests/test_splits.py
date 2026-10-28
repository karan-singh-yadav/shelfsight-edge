import cv2
import numpy as np
import pytest

from shelfsight.data.formats import DatasetError
from shelfsight.data.splits import build_manifest, dhash_image, group_images, verify_splits


def make_images(folder, n=12, seed=0):
    """Distinct smooth random images + one near-duplicate (brightness shift, re-encoded)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        field = rng.random((6, 8, 3)).astype(np.float32)
        img = cv2.resize(field, (128, 96), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(folder / f"img_{i:02d}.png"), (np.clip(img, 0, 1) * 200).astype(np.uint8))
    base = cv2.imread(str(folder / "img_00.png"))
    cv2.imwrite(
        str(folder / "img_00_dup.jpg"), np.clip(base.astype(int) + 8, 0, 255).astype(np.uint8)
    )


def test_dhash_identical_and_near_dup(tmp_path):
    make_images(tmp_path)
    a, b = dhash_image(tmp_path / "img_00.png"), dhash_image(tmp_path / "img_00_dup.jpg")
    assert bin(a ^ b).count("1") <= 4
    c = dhash_image(tmp_path / "img_05.png")
    assert bin(a ^ c).count("1") > 4


def test_group_images():
    assert group_images([0, 1, 0xFFFFFFFF], threshold=1) == [0, 0, 1]
    assert group_images([], threshold=1) == []


def test_manifest_deterministic_and_leak_free(tmp_path):
    make_images(tmp_path)
    m1 = build_manifest(tmp_path, seed=3)
    m2 = build_manifest(tmp_path, seed=3)
    assert m1.equals(m2)
    assert set(m1["split"]) == {"train", "val", "test"}
    dup_splits = m1[m1["image"].isin(["img_00.png", "img_00_dup.jpg"])]["split"].nunique()
    assert dup_splits == 1  # near-duplicates stay together
    problems = [p for p in verify_splits(m1, tmp_path) if not p.startswith("WARNING")]
    assert problems == []
    assert any(p.startswith("WARNING") for p in verify_splits(m1))  # unknown provenance flagged


def test_verify_detects_injected_leakage_and_tampering(tmp_path):
    make_images(tmp_path)
    m = build_manifest(tmp_path, seed=3)
    leaked = m.copy()
    i = leaked.index[leaked["image"] == "img_00_dup.jpg"][0]
    other = next(s for s in ("train", "val", "test") if s != leaked.loc[i, "split"])
    leaked.loc[i, "split"] = other
    probs = verify_splits(leaked)
    assert any("near-duplicate across splits" in p for p in probs)

    (tmp_path / "img_03.png").write_bytes(b"tampered")
    assert any("file changed" in p for p in verify_splits(m, tmp_path))
    (tmp_path / "img_04.png").unlink()
    assert any("missing on disk" in p for p in verify_splits(m, tmp_path))


def test_manifest_errors(tmp_path):
    with pytest.raises(DatasetError):
        build_manifest(tmp_path)  # no images
    with pytest.raises(ValueError):
        build_manifest(tmp_path, ratios=(0.5, 0.5, 0.5))
