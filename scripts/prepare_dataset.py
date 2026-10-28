"""Dataset preparation CLI (run with --help on any subcommand for options).

Subcommands: make-synthetic, validate, stats, visualize, convert, split, verify-splits.

Example workflow on the synthetic dataset:
  python scripts/prepare_dataset.py make-synthetic --out data/synthetic
  python scripts/prepare_dataset.py validate \\
      --csv data/synthetic/annotations_train.csv --images data/synthetic/images
  python scripts/prepare_dataset.py convert --images data/synthetic/images \\
      --out data/yolo_synthetic --train-csv data/synthetic/annotations_train.csv \\
      --val-csv data/synthetic/annotations_val.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from shelfsight.data.convert import convert_split, write_dataset_yaml
from shelfsight.data.formats import DatasetError, load_annotations, select_subset
from shelfsight.data.splits import build_manifest, verify_splits
from shelfsight.data.stats import dataset_stats, write_stats
from shelfsight.data.synthetic import make_synthetic_sku_dataset
from shelfsight.data.validate_dataset import validate_dataset
from shelfsight.data.visualize import render_samples
from shelfsight.logging_config import setup_logging


def cmd_make_synthetic(a: argparse.Namespace) -> int:
    csvs = make_synthetic_sku_dataset(a.out, seed=a.seed)
    print(f"Synthetic dataset written to {a.out} (for tooling tests only): {list(csvs.values())}")
    return 0


def cmd_validate(a: argparse.Namespace) -> int:
    report = validate_dataset(a.csv, a.images, deep=a.deep, check_images=not a.skip_images)
    print(report.summary())
    for issue in report.examples[:10]:
        print(f"   e.g. {issue.code}: {issue.image} - {issue.detail}")
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return 0 if report.ok else 1


def cmd_stats(a: argparse.Namespace) -> int:
    stats = dataset_stats(load_annotations(a.csv))
    write_stats(stats, a.out)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_visualize(a: argparse.Namespace) -> int:
    paths = render_samples(load_annotations(a.csv), Path(a.images), Path(a.out), n=a.n, seed=a.seed)
    print(f"Wrote {len(paths)} images to {a.out}")
    return 0 if paths else 1


def cmd_convert(a: argparse.Namespace) -> int:
    out = Path(a.out)
    names = a.classes
    done: list[str] = []
    for split, csv in (("train", a.train_csv), ("val", a.val_csv), ("test", a.test_csv)):
        if not csv:
            continue
        df = load_annotations(csv)
        if a.limit:
            df = select_subset(df, a.limit, seed=a.seed)
        stats = convert_split(
            df, Path(a.images), out, split, names, a.image_mode, strict=not a.lenient
        )
        print(stats)
        done.append(split)
    if not done:
        print("No CSVs given", file=sys.stderr)
        return 2
    print(f"Wrote {write_dataset_yaml(out, names, done)}")
    return 0


def cmd_split(a: argparse.Namespace) -> int:
    sources = pd.read_csv(a.sources) if a.sources else None
    manifest = build_manifest(
        a.images, seed=a.seed, near_dup_threshold=a.threshold, sources=sources
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(a.out, index=False)
    print(manifest["split"].value_counts().to_string())
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    problems = verify_splits(pd.read_csv(a.manifest), a.images, a.threshold)
    for p in problems:
        print(p)
    fatal = [p for p in problems if not p.startswith("WARNING")]
    print("OK: no leakage or integrity problems" if not fatal else f"{len(fatal)} problem(s)")
    return 1 if fatal else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("make-synthetic")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_make_synthetic)

    s = sub.add_parser("validate")
    s.add_argument("--csv", required=True)
    s.add_argument("--images", required=True)
    s.add_argument("--deep", action="store_true", help="fully decode every image (slow)")
    s.add_argument("--skip-images", action="store_true", help="validate boxes only")
    s.add_argument("--report", help="write JSON report here")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("stats")
    s.add_argument("--csv", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_stats)

    s = sub.add_parser("visualize")
    s.add_argument("--csv", required=True)
    s.add_argument("--images", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("-n", type=int, default=8)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_visualize)

    s = sub.add_parser("convert")
    s.add_argument("--images", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--train-csv")
    s.add_argument("--val-csv")
    s.add_argument("--test-csv")
    s.add_argument("--classes", nargs="+", default=["object"])
    s.add_argument("--limit", type=int, help="use only N random images per split (dev subset)")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--image-mode", choices=["copy", "hardlink"], default="copy")
    s.add_argument("--lenient", action="store_true", help="skip missing images instead of failing")
    s.set_defaults(fn=cmd_convert)

    s = sub.add_parser("split")
    s.add_argument("--images", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--sources", help="CSV with columns image,source,license")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument(
        "--threshold", type=int, default=4, help="max dHash Hamming distance = duplicate"
    )
    s.set_defaults(fn=cmd_split)

    s = sub.add_parser("verify-splits")
    s.add_argument("--manifest", required=True)
    s.add_argument("--images")
    s.add_argument("--threshold", type=int, default=4)
    s.set_defaults(fn=cmd_verify)
    return p


def main() -> int:
    setup_logging("INFO")
    args = build_parser().parse_args()
    try:
        return args.fn(args)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
