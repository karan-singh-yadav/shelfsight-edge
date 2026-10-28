"""Verify the environment: required imports, optional ML stack, and system details.

Usage: python scripts/check_env.py [--json]
Exit code 0 if all required packages import, 1 otherwise. CUDA is never required.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

from shelfsight.system_info import collect_system_info

REQUIRED = ["numpy", "pandas", "cv2", "PIL", "yaml", "psutil", "shelfsight"]
OPTIONAL = ["torch", "ultralytics", "onnx", "onnxruntime", "streamlit"]


def try_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        return True, getattr(mod, "__version__", "ok")
    except Exception as exc:  # noqa: BLE001 - report any import failure, not just ImportError
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args()

    required = {n: try_import(n) for n in REQUIRED}
    optional = {n: try_import(n) for n in OPTIONAL}
    info = collect_system_info()

    cuda = "n/a (torch not installed)"
    if optional["torch"][0]:
        import torch

        cuda = f"available={torch.cuda.is_available()} (not required locally)"

    if args.json:
        print(
            json.dumps(
                {"required": required, "optional": optional, "cuda": cuda, "system": info}, indent=2
            )
        )
    else:
        print("== Required ==")
        for n, (ok, msg) in required.items():
            print(f"  [{'OK' if ok else 'FAIL'}] {n}: {msg}")
        print("== Optional (training, export, PyTorch backend) ==")
        for n, (ok, msg) in optional.items():
            print(f"  [{'OK' if ok else '--'}] {n}: {msg}")
        print(f"CUDA: {cuda}")
        print("== System ==")
        for k, v in info.items():
            if k != "packages":
                print(f"  {k}: {v}")
    return 0 if all(ok for ok, _ in required.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
