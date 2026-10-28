# ShelfSight Edge

CPU-first retail shelf monitoring: object detection, multi-object tracking, counting and **estimated** shelf occupancy, with PyTorch and ONNX Runtime backends, FP32/INT8 comparison, a reproducible benchmark harness and a Streamlit dashboard.

> **Disclaimer:** shelf occupancy is an *estimate* of visible product coverage, not an audited inventory count.
> **No performance numbers are published here yet.** All results in `docs/EVALUATION.md` read "Not measured" until you run the benchmark on your hardware and data.

## Quick start (details: [docs/VSCODE_GUIDE.md](docs/VSCODE_GUIDE.md))
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt && pip install -e . --no-deps
python scripts/check_env.py && pytest
streamlit run app/streamlit_app.py
```
Python 3.11 or 3.12. No GPU or CUDA needed locally. The dashboard and ONNX path need no PyTorch. PyTorch/Ultralytics (training, export, baseline) are installed separately, see `requirements-torch.txt`.

## What is in the box
| Area | Where | Status |
|---|---|---|
| Detector interface, PyTorch + ONNX Runtime CPU backends, own YOLOv8 decode/NMS | `src/shelfsight/inference/` | tested, incl. real YOLOv8n PyTorch-vs-ONNX agreement on one image |
| Dataset validation, YOLO conversion, stats, leak-free splits, synthetic fixture | `src/shelfsight/data/`, `scripts/prepare_dataset.py` | tested on synthetic data; real SKU-110K not seen |
| Tracking (ByteTrack-style), counting, line crossing | `tracking/`, `analytics/counting.py` | unit-tested; not validated on MOT benchmarks |
| Shelf regions, occupancy, low-occupancy alerts, CSV reports | `analytics/` | unit-tested |
| Video pipeline (streaming, stride, annotated output, stats) | `video/processor.py`, `scripts/process_video.py` | tested with synthetic video and a real-model run |
| Training wrapper (CPU smoke + GPU), Colab notebook | `scripts/train.py`, `notebooks/` | CPU smoke run executed; GPU/Colab run NOT executed |
| ONNX export + validation, static INT8 quantization | `scripts/export_onnx.py`, `scripts/quantize.py` | executed on YOLOv8n; INT8 accuracy on real data not measured |
| Metrics (mAP, P/R, count MAE), evaluator, benchmark, failure analysis | `src/shelfsight/evaluation/` | tested; mAP cross-checked against Ultralytics |
| Streamlit dashboard (6 tabs) | `app/` | headless startup and loaded-model render tested; manual click-through by you still needed |
| Docker (CPU, ONNX only) | `Dockerfile` | written, NOT built or run |
| CI | `.github/workflows/ci.yml` | written, not run |

## Common commands
```bash
python scripts/download_weights.py                         # explicit weights download
python scripts/detect.py --image img.jpg --model models/yolov8n.onnx
python scripts/prepare_dataset.py --help                   # validate | stats | visualize | convert | split | verify-splits
python scripts/train.py --data data/yolo/dataset.yaml --smoke --device cpu
python scripts/export_onnx.py --weights models/best.pt --check-image img.jpg
python scripts/quantize.py --model models/best.onnx --calib-dir data/yolo/images/train --out models/best_int8.onnx --exclude-head
python scripts/evaluate.py --model models/best.onnx --images <test/images> --labels <test/labels> --out reports/benchmarks/onnx_fp32_eval.json
python scripts/benchmark.py --model models/best.onnx --images <test/images> --threads 4 --out reports/benchmarks/onnx_fp32.json
python scripts/make_report.py
```

## Docs
[Architecture](docs/ARCHITECTURE.md) | [Datasets](docs/DATASETS.md) | [Evaluation protocol and findings](docs/EVALUATION.md) | [Limitations](docs/LIMITATIONS.md) | [Licenses](docs/LICENSES.md) | [Interview prep](docs/INTERVIEW_PREP.md) | [Resume bullets](docs/RESUME_BULLETS.md)

## Engineering notes
* Backends are interchangeable behind `Detections`/`Detector`; the PyTorch backend letterboxes to a square input (`rect=False`) so it sees exactly what the static ONNX model sees.
* Benchmarks record CPU, RAM, OS, Python and library versions, input size, batch size, warm-up/timed runs, thread settings and model hash, and state which stages each FPS figure includes.
* Large files (data, weights, videos) are git-ignored; publish weights only where licences allow ([docs/LICENSES.md](docs/LICENSES.md)).

Code is MIT licensed. Third-party components (notably Ultralytics, AGPL-3.0) keep their own licences.
