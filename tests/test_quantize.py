import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from shelfsight.inference.onnx_backend import OnnxDetector

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    env_path = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "quantize.py"), *map(str, args)],
        capture_output=True, text=True, env={"PYTHONPATH": env_path, "PATH": ""},
    )  # fmt: skip


def make_imgs(folder, n=4):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        cv2.imwrite(str(folder / f"i{i}.png"), rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))


def test_quantize_pipeline_on_tiny_model(tiny_onnx, tmp_path):
    train = tmp_path / "train"
    make_imgs(train)
    out = tmp_path / "q.onnx"
    r = run("--model", tiny_onnx, "--calib-dir", train, "--out", out, "--num-calib", 3)
    assert r.returncode == 0, r.stderr
    assert out.is_file()
    det = OnnxDetector(out, conf_threshold=0.25)
    assert len(det.predict(np.zeros((64, 64, 3), np.uint8))) == 1  # still decodes a box


def test_quantize_refuses_test_split_and_empty_dir(tiny_onnx, tmp_path):
    bad = tmp_path / "test"
    make_imgs(bad)
    r = run("--model", tiny_onnx, "--calib-dir", bad, "--out", tmp_path / "q.onnx")
    assert r.returncode == 2 and "refusing" in r.stderr
    (tmp_path / "empty").mkdir()
    r = run("--model", tiny_onnx, "--calib-dir", tmp_path / "empty", "--out", tmp_path / "q.onnx")
    assert r.returncode == 2 and "no calibration images" in r.stderr


def test_calibration_reader_skips_bad_images(tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("quantize_script", ROOT / "scripts" / "quantize.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    make_imgs(tmp_path / "t", 2)
    (tmp_path / "t" / "bad.png").write_bytes(b"junk")
    reader = mod.ImageCalibrationReader(sorted((tmp_path / "t").iterdir()), "images", 64)
    got = []
    while (item := reader.get_next()) is not None:
        got.append(item["images"].shape)
    assert got == [(1, 3, 64, 64)] * 2
    reader.rewind()
    assert reader.get_next() is not None
    pytest.importorskip("onnxruntime")


def test_exclude_head_errors_cleanly_when_graph_has_no_model_blocks(tiny_onnx, tmp_path):
    train = tmp_path / "train"
    make_imgs(train)
    r = run(
        "--model", tiny_onnx, "--calib-dir", train, "--out", tmp_path / "q.onnx", "--exclude-head"
    )
    assert r.returncode == 2 and "no '/model.N/'" in r.stderr
