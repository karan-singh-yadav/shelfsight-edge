"""UI helper functions (no Streamlit state here, so they are unit-testable)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from shelfsight.analytics.shelf_regions import RegionError, ShelfRegion, validate_regions
from shelfsight.errors import InvalidImageError

logger = logging.getLogger(__name__)
MODEL_SUFFIXES = (".pt", ".onnx")
REGION_COLUMNS = ["name", "x1", "y1", "x2", "y2", "low_threshold"]


def list_model_files(models_dir: str | Path) -> list[Path]:
    """Model files present in ``models_dir`` (never downloads anything)."""
    d = Path(models_dir)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in MODEL_SUFFIXES)


def backend_for(path: str | Path) -> str:
    return "onnx" if str(path).lower().endswith(".onnx") else "pytorch"


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes to BGR, raising ``InvalidImageError`` with a clear message."""
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    try:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise InvalidImageError(f"Could not decode the image: {exc}") from exc
    if img is None:
        raise InvalidImageError(
            "The uploaded file is not a readable image (corrupt or unsupported)."
        )
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise InvalidImageError("Could not encode the result image.")
    return buf.tobytes()


def save_upload(data: bytes, suffix: str) -> Path:
    """Write uploaded bytes to a temp file and return its path (caller deletes if desired)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
        fh.write(data)
    return Path(fh.name)


def find_ffmpeg() -> str | None:
    """Locate ffmpeg without relying on PATH being refreshed.

    Order: PATH, Windows WinGet install folders, then the binary from ``imageio-ffmpeg``.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    local = os.environ.get("LOCALAPPDATA")
    if local:
        base = Path(local) / "Microsoft" / "WinGet"
        candidates = [base / "Links" / "ffmpeg.exe"]
        candidates += sorted((base / "Packages").glob("*FFmpeg*/**/bin/ffmpeg.exe"))
        for cand in candidates:
            if cand.is_file():
                return str(cand)
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - package missing or binary unavailable
        return None


def maybe_reencode_h264(path: Path) -> Path:
    """Browsers cannot play OpenCV's mp4v output. If ffmpeg exists, re-encode to H.264.

    Returns the new file, or the original path if ffmpeg is missing or fails.
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return path
    out = path.with_name(path.stem + "_h264.mp4")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-vcodec", "libx264",
             "-pix_fmt", "yuv420p", str(out)],
            check=True, timeout=600, capture_output=True, text=True,
        )  # fmt: skip
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg failed: %s", (exc.stderr or "")[-300:])
        return path
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ffmpeg could not run: %s", exc)
        return path
    return out if out.is_file() and out.stat().st_size > 0 else path


def regions_to_df(regions: list[ShelfRegion]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": r.name, "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
             "low_threshold": r.low_threshold}
            for r in regions
        ],
        columns=REGION_COLUMNS,
    )  # fmt: skip


def df_to_regions(df: pd.DataFrame) -> list[ShelfRegion]:
    """Validate an edited regions table; raises ``RegionError`` naming the bad row."""
    regions: list[ShelfRegion] = []
    for i, row in df.reset_index(drop=True).iterrows():
        if pd.isna(row["name"]) or str(row["name"]).strip() == "":
            raise RegionError(f"Row {i + 1}: name is required")
        try:
            thr = None if pd.isna(row["low_threshold"]) else float(row["low_threshold"])
            regions.append(
                ShelfRegion(
                    str(row["name"]).strip(),
                    float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]), thr,
                )
            )  # fmt: skip
        except (TypeError, ValueError) as exc:
            if isinstance(exc, RegionError):
                raise
            raise RegionError(f"Row {i + 1}: coordinates must be numbers ({exc})") from exc
    if not regions:
        raise RegionError("Define at least one region")
    validate_regions(regions)
    return regions