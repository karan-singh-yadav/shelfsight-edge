"""Timestamped analytics recording, CSV export and processing history."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from shelfsight.analytics.shelf_regions import RegionStats

OCCUPANCY_DISCLAIMER = (
    "Shelf occupancy is an ESTIMATE of visible product coverage from detections. It is not an "
    "audited inventory count and is affected by occlusion, perspective, overlap and camera angle."
)


class AnalyticsRecorder:
    """Collects frame-level and region-level rows during processing."""

    def __init__(self) -> None:
        self._frames: list[dict[str, Any]] = []
        self._regions: list[dict[str, Any]] = []

    def add_frame(
        self,
        frame_index: int,
        timestamp_s: float,
        n_detections: int,
        n_tracks: int | None = None,
        region_stats: list[RegionStats] | None = None,
    ) -> None:
        self._frames.append(
            {
                "frame_index": frame_index,
                "timestamp_s": round(timestamp_s, 4),
                "detections": n_detections,
                "active_tracks": n_tracks,
            }
        )
        for rs in region_stats or []:
            self._regions.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": round(timestamp_s, 4),
                    "region": rs.name,
                    "count": rs.count,
                    "occupancy_ratio": round(rs.occupancy_ratio, 4),
                    "low_occupancy": rs.low_occupancy,
                }
            )

    def frame_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._frames, columns=["frame_index", "timestamp_s", "detections", "active_tracks"]
        )

    def region_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._regions,
            columns=[
                "frame_index", "timestamp_s", "region", "count", "occupancy_ratio", "low_occupancy",
            ],
        )  # fmt: skip

    def save_csv(self, out_dir: str | Path, prefix: str = "analytics") -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}
        frames = self.frame_dataframe()
        paths["frames"] = out / f"{prefix}_frames.csv"
        frames.to_csv(paths["frames"], index=False)
        regions = self.region_dataframe()
        if len(regions):
            paths["regions"] = out / f"{prefix}_regions.csv"
            regions.to_csv(paths["regions"], index=False)
        return paths


def append_history(path: str | Path, record: dict[str, Any]) -> None:
    """Append one processing record (JSON line) with a UTC timestamp."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time_utc": datetime.now(UTC).isoformat(timespec="seconds"), **record}
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def read_history(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        return pd.DataFrame()
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a partially written line
    return pd.DataFrame(rows)
