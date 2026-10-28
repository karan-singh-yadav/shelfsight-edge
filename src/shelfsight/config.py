"""Typed, validated application configuration loaded from YAML.

Values come from built-in defaults, optionally overridden by a YAML file. Unknown keys and
out-of-range values raise ``ConfigError`` instead of being silently ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

VALID_BACKENDS = ("pytorch", "onnx")
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed, or invalid."""


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root (the directory holding ``pyproject.toml``).

    ``SHELFSIGHT_ROOT`` overrides the search. Falls back to the current directory.
    """
    env = os.environ.get("SHELFSIGHT_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")


@dataclass(frozen=True)
class InferenceConfig:
    backend: str = "pytorch"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 640
    num_threads: int = 0

    def __post_init__(self) -> None:
        if self.backend not in VALID_BACKENDS:
            raise ConfigError(
                f"inference.backend must be one of {VALID_BACKENDS}, got {self.backend!r}"
            )
        if not 0.0 <= self.conf_threshold <= 1.0:
            raise ConfigError("inference.conf_threshold must be in [0, 1]")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ConfigError("inference.iou_threshold must be in [0, 1]")
        if self.imgsz <= 0 or self.imgsz % 32 != 0:
            raise ConfigError("inference.imgsz must be a positive multiple of 32")
        if self.num_threads < 0:
            raise ConfigError("inference.num_threads must be >= 0")


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    json_format: bool = False
    log_file: Path | None = None

    def __post_init__(self) -> None:
        if self.level.upper() not in VALID_LOG_LEVELS:
            raise ConfigError(
                f"logging.level must be one of {VALID_LOG_LEVELS}, got {self.level!r}"
            )


@dataclass(frozen=True)
class AppConfig:
    project_root: Path = field(default_factory=find_project_root)
    paths: PathsConfig = field(default_factory=PathsConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _build(cls: type, data: Any, section: str) -> Any:
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"Section '{section}' must be a mapping, got {type(data).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{section}': {sorted(unknown)}; allowed: {sorted(allowed)}"
        )
    return cls(**data)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict, raising ``ConfigError`` with a clear message on failure."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {p}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {p} must be a mapping")
    return data


def load_config(path: str | Path | None = None, project_root: Path | None = None) -> AppConfig:
    """Load configuration, applying defaults for anything the file does not set.

    Relative paths in the ``paths`` section and ``logging.log_file`` are resolved against the
    project root so behaviour does not depend on the current working directory.
    """
    root = (project_root or find_project_root()).resolve()
    raw = load_yaml(path) if path is not None else {}
    unknown_top = set(raw) - {"paths", "inference", "logging"}
    if unknown_top:
        raise ConfigError(f"Unknown top-level key(s): {sorted(unknown_top)}")

    paths_raw = raw.get("paths") or {}
    if not isinstance(paths_raw, dict):
        raise ConfigError("Section 'paths' must be a mapping")
    paths_cfg = _build(PathsConfig, {k: Path(v) for k, v in paths_raw.items()}, "paths")
    paths_cfg = PathsConfig(
        **{f.name: _resolve(getattr(paths_cfg, f.name), root) for f in fields(paths_cfg)}
    )

    log_raw = dict(raw.get("logging") or {})
    if log_raw.get("log_file"):
        log_raw["log_file"] = _resolve(Path(log_raw["log_file"]), root)
    else:
        log_raw["log_file"] = None

    return AppConfig(
        project_root=root,
        paths=paths_cfg,
        inference=_build(InferenceConfig, raw.get("inference"), "inference"),
        logging=_build(LoggingConfig, log_raw, "logging"),
    )


def _resolve(p: Path, root: Path) -> Path:
    return p if p.is_absolute() else (root / p).resolve()
