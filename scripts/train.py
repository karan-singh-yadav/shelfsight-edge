"""Reproducible fine-tuning wrapper around Ultralytics.

CPU smoke test (verifies the pipeline only, not useful accuracy):
  python scripts/train.py --data data/yolo_synthetic/dataset.yaml --model models/yolov8n.pt --smoke
Cloud GPU run (Colab):
  python scripts/train.py --data <dataset.yaml> --model models/yolov8n.pt --device cuda:0 \
      --epochs 50 --imgsz 640 --batch 16 --seed 0 --name sku_v1
Resume an interrupted run:  add  --resume runs/<name>/weights/last.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shelfsight.errors import DeviceUnavailableError, ModelNotFoundError
from shelfsight.system_info import collect_system_info


def resolve_device(requested: str) -> str:
    """Return a usable device or raise ``DeviceUnavailableError`` (never silently fall back)."""
    import torch

    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise DeviceUnavailableError(
            f"Requested '{requested}' but CUDA is unavailable. Use --device cpu, or run on a GPU "
            "machine (e.g. Colab with a GPU runtime)."
        )
    return requested


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, help="Ultralytics dataset.yaml")
    p.add_argument("--model", default="models/yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="train")
    p.add_argument("--resume", help="path to last.pt of an interrupted run")
    p.add_argument(
        "--smoke", action="store_true", help="1 epoch, small imgsz/batch (pipeline test)"
    )
    a = p.parse_args()

    try:
        device = resolve_device(a.device)
        weights = Path(a.resume) if a.resume else Path(a.model)
        if not weights.is_file():
            raise ModelNotFoundError(f"Model/checkpoint not found: {weights}")
        if not Path(a.data).is_file():
            raise FileNotFoundError(f"Dataset YAML not found: {a.data}")
    except (DeviceUnavailableError, ModelNotFoundError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    epochs, imgsz, batch = (1, 320, 4) if a.smoke else (a.epochs, a.imgsz, a.batch)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    cfg = dict(
        data=str(Path(a.data).resolve()), epochs=epochs, imgsz=imgsz, batch=batch, seed=a.seed,
        device=device, workers=a.workers, project=a.project, name=a.name,
        deterministic=True, plots=not a.smoke,
    )  # fmt: skip
    model.train(resume=bool(a.resume), **({} if a.resume else cfg))
    # Ultralytics may auto-increment the run name (e.g. train-2); report the real folder.
    out_dir = Path(model.trainer.save_dir)
    (out_dir / "shelfsight_train_config.json").write_text(
        json.dumps(
            {"args": cfg, "system": collect_system_info(), "base_weights": str(weights)}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"Done. Best weights: {out_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
