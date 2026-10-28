# Limitations

**Shelf occupancy is an estimate**, not audited inventory. It is the fraction of a region covered by detection boxes and is biased by occlusion, perspective (back rows hidden), overlapping products, packaging height, glare and camera angle changes. Regions are static rectangles; a moved camera invalidates them.

* **SKU-110K has one generic class**: this project detects "product instance", not brand/SKU.
* **Tracker is not validated on MOT benchmarks**; class-agnostic; large frame strides degrade association; fragmentation inflates unique-track counts.
* **Dense shelves**: `max_det` (default 1000) caps output; raise it for crowded images and check count MAE.
* **ONNX is static 640x640**; other sizes require re-export. Letterboxing shrinks tiny products.
* **INT8** is not assumed to help. Observed on one test image: quantizing the whole YOLOv8n graph produced zero detections; keeping the Detect head in FP32 recovered them. Real accuracy/speed must be measured on your CPU with `evaluate.py`/`benchmark.py`.
* **Video output** uses OpenCV `mp4v`, which browsers cannot play; the dashboard re-encodes with ffmpeg if installed.
* **Metrics**: precision/recall are micro-averaged at a fixed confidence; AP is COCO-style 101-point. Values are not identical to Ultralytics' validator printouts.
* **Tested environment**: Linux x86_64 sandbox, Python 3.12, 1 CPU core. Not run on Windows, not run in Docker, no GPU training run, no real SKU-110K or custom data seen yet.
