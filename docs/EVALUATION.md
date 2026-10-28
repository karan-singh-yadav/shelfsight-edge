# Evaluation report

> Every number below is **Not measured** until you run the commands on your own hardware and data. Do not fill cells from memory or from other people's results.

## Protocol
* Same held-out **test** split, same `imgsz`, `iou`, `max_det`, `pr_conf`, `eval_conf` for every model.
* Quantization calibration uses **train** images only (`quantize.py` refuses paths containing a `test` folder).
* Latency: `scripts/benchmark.py`, one model per process, fixed threads, batch 1, warm-up then timed runs, median and p95.
* Accuracy: `scripts/evaluate.py` (add `--agnostic` for the COCO-pretrained baseline on single-class SKU-110K).

## Commands
```bash
python scripts/evaluate.py  --model models/yolov8n.pt   --images <test/images> --labels <test/labels> --agnostic --split-name test --out reports/benchmarks/pretrained_eval.json
python scripts/evaluate.py  --model models/best.pt      --images <test/images> --labels <test/labels> --split-name test --out reports/benchmarks/finetuned_pt_eval.json --failures-dir reports/failures
python scripts/evaluate.py  --model models/best.onnx    ... --out reports/benchmarks/onnx_fp32_eval.json
python scripts/evaluate.py  --model models/best_int8.onnx ... --out reports/benchmarks/onnx_int8_eval.json
python scripts/benchmark.py --model <model> --images <test/images> --threads 4 --runs 100 --out reports/benchmarks/<name>.json
```
File names must be `<name>.json` (benchmark) and `<name>_eval.json` (accuracy) to be merged in the dashboard table.

## Results (fill after measuring)
| Model | mAP50 | mAP50-95 | Precision | Recall | Count MAE | Median ms | p95 ms | FPS | Size MB |
|---|---|---|---|---|---|---|---|---|---|
| Pretrained YOLOv8n (class-agnostic) | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| Fine-tuned PyTorch | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| ONNX FP32 | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| ONNX INT8 | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |

`python scripts/make_report.py` builds this table from the saved JSON files (rows missing accuracy show "Not measured").

## Verified engineering findings (single test image, NOT benchmarks)
1. **ONNX vs PyTorch input geometry.** On Ultralytics' `bus.jpg`, default PyTorch inference (minimal-rectangle letterbox) and static 640x640 ONNX gave different scores and one different borderline detection. With `rect=False` PyTorch scores matched ONNX to 3 decimals, confirming the decode/NMS implementation and explaining the difference. The backend now defaults to `rect=False`.
2. **INT8 sensitivity.** Static INT8 of the full graph (MinMax, 2 calibration images) shrank the model 12.2 -> 3.4 MB but returned 0 detections; keeping the `/model.22/` Detect head in FP32 (5.9 MB) reproduced all detections (match rate 1.0, mean IoU 0.98). Accuracy and speed on real data: Not measured.
3. **Metric cross-check.** Recall matched Ultralytics' validator (0.683 vs 0.6829) on a smoke-trained model; AP on identical precision-recall curves matched Ultralytics' `compute_ap`; a test compares mAP against `ap_per_class` on continuous scores. Saturated/tied confidences (as in a 1-epoch smoke model) make AP ordering ambiguous and implementations disagree there.
4. **Library side effect.** Importing Ultralytics patches `PIL.Image.open`, changing the exception raised for corrupt images; validators now catch it (regression test included).

## Failure analysis
`evaluate.py --failures-dir DIR` writes `failures.csv` (per-image FN/FP/count error/dense/tiny flags) and overlays (green = ground truth, coloured = predictions) for the worst images. Occlusion and glare cannot be detected from boxes: review the overlays and add tags in the `manual_tags` column. Summarise 2-3 examples per category here after reviewing: occlusion, tiny products, dense overlap, glare, bad boxes, false positives, misses.
