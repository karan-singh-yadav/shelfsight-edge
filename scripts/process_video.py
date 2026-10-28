"""Run detection + tracking (+ shelf regions) on a video.

  python scripts/process_video.py --video in.mp4 --model models/best.onnx --out out.mp4 \
      --stride 2 --regions configs/regions.example.yaml --csv-dir reports/video_run
"""

from __future__ import annotations

import argparse
import json
import sys

from shelfsight.analytics.reporting import OCCUPANCY_DISCLAIMER, AnalyticsRecorder
from shelfsight.analytics.shelf_regions import RegionError, load_regions
from shelfsight.errors import ShelfSightError
from shelfsight.inference.factory import create_detector
from shelfsight.tracking.tracker import ByteTracker
from shelfsight.video.processor import process_video


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", help="annotated output video path")
    p.add_argument("--backend", choices=["pytorch", "onnx"])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--no-track", action="store_true")
    p.add_argument("--regions", help="YAML/JSON shelf regions")
    p.add_argument("--low-occupancy", type=float, default=0.2)
    p.add_argument("--csv-dir")
    a = p.parse_args()
    backend = a.backend or ("onnx" if a.model.endswith(".onnx") else "pytorch")
    try:
        det = create_detector(backend, a.model, conf_threshold=a.conf, num_threads=a.threads)
        regions = load_regions(a.regions) if a.regions else None
        rec = AnalyticsRecorder() if a.csv_dir else None
        stats = process_video(
            a.video, det, a.out, None if a.no_track else ByteTracker(), regions,
            a.low_occupancy, a.stride, a.max_frames, rec,
        )  # fmt: skip
    except (ShelfSightError, RegionError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(stats.to_dict(), indent=2))
    if rec is not None:
        print("CSV:", {k: str(v) for k, v in rec.save_csv(a.csv_dir).items()})
    if regions:
        print(OCCUPANCY_DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
