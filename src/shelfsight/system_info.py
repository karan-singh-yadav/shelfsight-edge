"""Collect hardware/software details that every benchmark record must include."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from typing import Any

import psutil

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "opencv-python-headless",
    "pillow",
    "torch",
    "torchvision",
    "ultralytics",
    "onnx",
    "onnxruntime",
    "streamlit",
)


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Return installed versions, or ``"not installed"`` for missing packages."""
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def cpu_model() -> str:
    """Best-effort CPU model string; falls back to ``platform.processor()``."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        elif sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            if out:
                return out
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def collect_system_info() -> dict[str, Any]:
    """Snapshot of the environment for reproducible benchmark records."""
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_model": cpu_model(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "packages": package_versions(),
    }
