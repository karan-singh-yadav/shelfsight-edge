"""Build the benchmark comparison table (Markdown) from saved result files.

Benchmark files: reports/benchmarks/<name>.json   Accuracy files: <name>_eval.json
Missing values are printed as 'Not measured'.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shelfsight.evaluation.benchmark import comparison_table


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="reports/benchmarks")
    p.add_argument("--out", default="reports/benchmark_table.md")
    a = p.parse_args()
    d = Path(a.dir)
    files = (
        sorted(f for f in d.glob("*.json") if not f.stem.endswith("_eval")) if d.is_dir() else []
    )
    if not files:
        print(f"No benchmark JSON files found in {d}", file=sys.stderr)
        return 2
    table = comparison_table(files, {f.stem: d / f"{f.stem}_eval.json" for f in files})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
