# From zip to running project in VS Code (Windows)

## 1. Open and set up (about 10 minutes)
1. Unzip `shelfsight-edge-final.zip`, then in VS Code: **File > Open Folder** and pick `shelfsight-edge`. Accept the "install recommended extensions" prompt (Python, Ruff, Pylance).
2. Install Python 3.12 (or 3.11) from python.org, ticking **Add to PATH**. Check in a terminal: `py -0`.
3. Open the terminal (**Ctrl+`**, PowerShell) and run:
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e . --no-deps
```
   If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
4. **Ctrl+Shift+P > Python: Select Interpreter > `.venv`**.
5. Verify:
```powershell
python scripts/check_env.py     # required packages all [OK]; send me the system block if anything fails
pytest                          # expect: all pass; 3 skipped until torch is installed
ruff check .
```

## 2. Try the pipeline with no downloads
```powershell
python scripts/prepare_dataset.py make-synthetic --out data/synthetic
python scripts/prepare_dataset.py validate --csv data/synthetic/annotations_train.csv --images data/synthetic/images
```

## 3. Baseline detector on your CPU
```powershell
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-torch.txt
pip install ultralytics==8.4.185 --no-deps
python scripts/download_weights.py                       # explicit 6 MB download into models/
python scripts/detect.py --image <any photo>.jpg --model models/yolov8n.pt
python scripts/export_onnx.py --weights models/yolov8n.pt --check-image <any photo>.jpg
python scripts/detect.py --image <any photo>.jpg --model models/yolov8n.onnx
```
The export script prints a PyTorch-vs-ONNX consistency check (expect a match rate near 1.0).

## 4. Dashboard
```powershell
streamlit run app/streamlit_app.py
```
Pick a model in the sidebar, press **Load model**. Install ffmpeg (`winget install ffmpeg`) for in-browser video preview.

## 5. Real data (you must do these; I could not)
1. **SKU-110K**: download from the repository linked in `docs/DATASETS.md`; check the file size and free disk space *before* extracting; respect the non-commercial licence. Then run `validate`, `stats`, `visualize` on the val CSV; if validation reports a column-format error, send me the first 3 lines of the CSV.
2. **Custom dataset**: photograph or source images you may use, list them in `sources.csv` (image,source,license), annotate in CVAT (YOLO export), then `prepare_dataset.py split` and `verify-splits`.
3. Convert to YOLO: `prepare_dataset.py convert ... --limit 50` for a dev subset first.

## 6. Train on Colab, finish on your laptop
Open `notebooks/train_colab.ipynb` in Colab (GPU runtime), run it, download `best.pt` into `models/`. Smoke-test locally first: `python scripts/train.py --data <dataset.yaml> --model models/yolov8n.pt --smoke --device cpu`.

## 7. Evaluate, quantize, benchmark (all on your CPU)
```powershell
python scripts/export_onnx.py --weights models/best.pt --out models/best.onnx
python scripts/quantize.py --model models/best.onnx --calib-dir <train/images> --out models/best_int8.onnx --num-calib 100 --exclude-head
python scripts/evaluate.py --model models/best.onnx --images <test/images> --labels <test/labels> --out reports/benchmarks/onnx_fp32_eval.json --failures-dir reports/failures
python scripts/benchmark.py --model models/best.onnx --images <test/images> --threads 4 --runs 100 --out reports/benchmarks/onnx_fp32.json
python scripts/make_report.py
```
Repeat for each model (see `docs/EVALUATION.md`), close other apps while benchmarking, and keep INT8 only if it measures better.

## 8. Video
```powershell
python scripts/process_video.py --video clip.mp4 --model models/best.onnx --out reports/out.mp4 --stride 2 --regions configs/regions.example.yaml --csv-dir reports/video_run
```

## 9. Publish
`git init`, commit, create a GitHub repo, push. `.gitignore` already excludes data, weights, videos and runs. Do not commit SKU-110K or weights trained on it (see `docs/LICENSES.md`). Record a 60-90 s screen capture of the dashboard (image, video, shelf tab, benchmark table) for the README.

## 10. Docker (not validated by me: no Docker in my sandbox)
```powershell
docker build -t shelfsight-edge .
docker run --rm -p 8501:8501 -v ${PWD}/models:/app/models -v ${PWD}/reports:/app/reports shelfsight-edge
```
The image is ONNX-only (no PyTorch). Needs roughly 1-2 GB disk and 1-2 GB RAM; confirm after building.

## Troubleshooting
* `ModuleNotFoundError: shelfsight`: you skipped `pip install -e . --no-deps` or the wrong interpreter is selected.
* cv2 import errors after installing Ultralytics: you installed it without `--no-deps`; run `pip uninstall -y opencv-python` then `pip install --force-reinstall opencv-python-headless==4.13.0.92`.
* `models/...` not found: weights are never downloaded automatically; run `download_weights.py`.
* Slow benchmark: pin `--threads` to your physical core count, plug in power, close browsers.
