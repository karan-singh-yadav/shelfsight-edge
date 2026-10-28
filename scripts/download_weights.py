"""Explicitly download pretrained YOLOv8n weights into models/ (never done implicitly).

The weights are distributed by Ultralytics under AGPL-3.0; see docs/LICENSES.md before
redistributing them or anything derived from them.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="yolov8n.pt", help="Ultralytics asset name")
    p.add_argument("--out-dir", default="models")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / args.name
    if target.exists():
        print(f"Already present: {target}")
        return 0
    from ultralytics import YOLO

    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            import os

            os.chdir(tmp)  # Ultralytics downloads into the current directory
            YOLO(args.name)
            shutil.move(str(Path(tmp) / args.name), str(cwd / target))
        except Exception as exc:  # noqa: BLE001
            print(f"Download failed: {exc}", file=sys.stderr)
            return 1
        finally:
            os.chdir(cwd)
    print(f"Saved {target} ({target.stat().st_size / 1024**2:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
