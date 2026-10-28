# Interview preparation notes

Answers are short on purpose: expand with your own measured numbers from `reports/`.

**Why YOLOv8n?** Small (about 3M parameters, 6 MB), runs on CPU, mature export path to ONNX, and a pretrained baseline exists. Trade-off: lower accuracy than larger variants on tiny, dense products; I would compare against `s` if latency allows (measure, don't assume).

**Why is SKU-110K appropriate, and what are its limits?** Dense shelf scenes with per-instance boxes, official splits, a recognised benchmark. Limits: one generic class (no SKU identity), academic/non-commercial licence, a fixed set of store photos, so the custom dataset covers my deployment domain.

**mAP vs precision vs recall.** Precision = TP/(TP+FP) at a chosen threshold; recall = TP/(TP+FN). mAP averages, over classes (and IoU thresholds for 50-95), the area under the precision-recall curve obtained by sweeping the confidence threshold, so it does not depend on one threshold but does depend on ranking quality and IoU strictness. My P/R are micro-averaged at a fixed confidence, so they do not equal Ultralytics' printed values.

**How does quantization work?** Map FP32 tensors to 8-bit integers with a scale and zero-point. Static quantization calibrates activation ranges on representative data (here: train images), then inserts quantize/dequantize nodes (QDQ). Risks: outlier-heavy layers lose accuracy. I observed that quantizing everything removed all detections on a test image while keeping the Detect head in FP32 recovered them; whether INT8 is faster depends on the CPU's integer instruction support, so I measure.

**Why can ONNX Runtime differ from PyTorch?** Different preprocessing/input shape (I found Ultralytics' rectangular letterbox vs a static 640x640 export changed scores), operator fusion and floating-point ordering, different NMS implementations, and graph simplification. After aligning input geometry, my scores matched to three decimals on a test image.

**How do I measure p95 latency?** Fix threads, warm up, time N single-image calls with a monotonic clock around `predict` only, store every sample, report median and the 95th percentile (not the mean). One model per process so memory and thread pools do not interfere. State which stages are included.

**How do I avoid duplicate counting in video?** Track objects (Kalman filter + IoU association with low-score rescue), count confirmed track IDs or line crossings, never sum per-frame detections. Remaining errors: ID switches and fragmentation after long occlusions.

**Dense overlapping products?** Raise `max_det`, tune NMS IoU (too low merges neighbours, too high keeps duplicates), use higher input resolution, evaluate count MAE, and inspect failure overlays. Class-aware NMS only separates different classes.

**Why can CPU inference be slower than expected?** Thread oversubscription, too few/too many threads, memory-bound ops, first-call overhead, input too large, preprocessing in Python, background load, thermal throttling, INT8 without VNNI-class hardware, decode/render dominating end-to-end FPS.

**How do I identify data leakage?** Exact hashes and perceptual hashes across splits (`verify-splits`), grouping near-duplicates before splitting, split by store/session when possible, never calibrating or tuning on test.

**Reproducible deployment?** Pinned requirements, container image (ONNX only), model hash and config recorded in every benchmark JSON, config files instead of constants, tests in CI.

**Current limitations?** See `docs/LIMITATIONS.md`: occupancy is an estimate, single-class benchmark, tracker not validated on MOT, static input size, untested on Windows/Docker until you run it.

**A bug you found and fixed (good story).** Three examples from this repo: backend preprocessing mismatch (rect letterbox), a library monkeypatch that changed an exception type and broke corrupt-file validation, and a YAML path that would have broken moving data between laptop and Colab.
