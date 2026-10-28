from pathlib import Path

import pytest

from shelfsight.config import ConfigError, load_config


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_defaults_without_file(tmp_path):
    cfg = load_config(None, project_root=tmp_path)
    assert cfg.inference.backend == "pytorch"
    assert cfg.paths.data_dir == (tmp_path / "data").resolve()


def test_yaml_overrides_and_relative_paths(tmp_path):
    p = write(tmp_path, "inference:\n  backend: onnx\n  imgsz: 320\npaths:\n  data_dir: mydata\n")
    cfg = load_config(p, project_root=tmp_path)
    assert cfg.inference.backend == "onnx"
    assert cfg.inference.imgsz == 320
    assert cfg.paths.data_dir == (tmp_path / "mydata").resolve()


def test_repo_default_config_is_valid():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml", project_root=root)
    assert cfg.logging.log_file is None


@pytest.mark.parametrize(
    "text",
    [
        "inference:\n  backend: tensorrt\n",
        "inference:\n  conf_threshold: 1.5\n",
        "inference:\n  imgsz: 100\n",
        "inference:\n  num_threads: -1\n",
        "logging:\n  level: LOUD\n",
        "inference:\n  typo_key: 1\n",
        "unknown_section: {}\n",
        "inference: [1, 2]\n",
    ],
)
def test_invalid_config_raises(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text), project_root=tmp_path)


def test_missing_and_malformed_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", project_root=tmp_path)
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(write(tmp_path, "a: [unclosed"), project_root=tmp_path)
