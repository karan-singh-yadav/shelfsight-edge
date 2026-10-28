# Resume bullets

Use **only** bullets whose evidence exists in your repo. Fill the `<...>` after you have run the commands; delete any bullet you cannot support. Completing the project does not guarantee a job.

## Supported by the code and tests already in the repo
* Built a modular CPU-first retail vision system (detection, ByteTrack-style tracking, shelf-region analytics, Streamlit dashboard) with PyTorch and ONNX Runtime backends behind one interface; 130 automated tests, Ruff-clean.
* Implemented YOLOv8 ONNX post-processing and class-aware NMS from scratch; validated against PyTorch (matching scores to 3 decimals after fixing a letterbox-geometry mismatch).
* Wrote dataset tooling (annotation validation, YOLO conversion, perceptual-hash group splits that prevent train/test leakage) with regression tests.
* Implemented COCO-style mAP/precision/recall and count-MAE evaluation, cross-validated against Ultralytics' AP implementation.
* Built a reproducible benchmark harness (median/p95 latency, throughput, memory, system and library versions, model hash) and a static-INT8 quantization pipeline calibrated on training data only.

## Requires your measurements (do NOT use until filled)
* Fine-tuned YOLOv8n on `<dataset, N images>`, reaching `<mAP50>` mAP50 / `<mAP50-95>` mAP50-95 on a held-out test split.
* Exported to ONNX Runtime and measured `<median ms>` ms median / `<p95 ms>` ms p95 latency on `<your CPU>` at `<threads>` threads, `<x>`x vs PyTorch.
* INT8: `<result: size, latency, accuracy change, or documented why FP32 was kept>`.
* Reduced object-count MAE to `<value>` on `<N>` images.
