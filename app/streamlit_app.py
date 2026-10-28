"""ShelfSight Edge dashboard. Run: streamlit run app/streamlit_app.py

Models are never loaded or downloaded at startup: pick a local model and press "Load model".
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import components as ui  # noqa: E402
from shelfsight.analytics.reporting import (  # noqa: E402
    OCCUPANCY_DISCLAIMER,
    AnalyticsRecorder,
    append_history,
    read_history,
)
from shelfsight.analytics.shelf_regions import RegionError, analyze_regions, load_regions  # noqa: E402
from shelfsight.config import find_project_root  # noqa: E402
from shelfsight.drawing import draw_detections, draw_regions  # noqa: E402
from shelfsight.errors import ShelfSightError  # noqa: E402
from shelfsight.evaluation.benchmark import benchmark_detector, comparison_table, save_benchmark  # noqa: E402
from shelfsight.inference.factory import create_detector  # noqa: E402
from shelfsight.system_info import collect_system_info  # noqa: E402
from shelfsight.tracking.tracker import ByteTracker  # noqa: E402
from shelfsight.video.processor import process_video  # noqa: E402

ROOT = find_project_root()
MODELS_DIR, REPORTS_DIR = ROOT / "models", ROOT / "reports"
HISTORY = REPORTS_DIR / "history.jsonl"
IMG_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]

st.set_page_config(page_title="ShelfSight Edge", page_icon="🛒", layout="wide")

st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #EC4899 100%);
       border-radius: 18px; padding: 1.5rem 1.8rem; color: #fff; margin-bottom: 1.1rem;}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero p {margin: .3rem 0 0 0; opacity: .92; font-size: 1.02rem;}
.pill {display: inline-block; padding: .18rem .7rem; border-radius: 999px; font-size: .8rem;
       font-weight: 600; margin-top: .7rem; background: rgba(255,255,255,.2);}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 1rem 1.1rem;
       height: 100%; background: rgba(128,128,128,.06);}
.card h4 {margin: 0 0 .3rem 0;}
.card p {margin: 0; font-size: .92rem; opacity: .85;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
       padding: .7rem .9rem; background: rgba(128,128,128,.06);}
.stTabs [data-baseweb="tab-list"] {gap: .3rem;}
.stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: .5rem 1rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model…")
def get_detector(path: str, backend: str, imgsz: int, threads: int, max_det: int):
    """Cached model load; only called after the user presses 'Load model'."""
    return create_detector(backend, path, imgsz=imgsz, max_det=max_det, num_threads=threads)


@st.cache_data
def cached_system_info() -> dict:
    return collect_system_info()


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🛒 ShelfSight Edge")
st.sidebar.markdown("**Model**")
models = ui.list_model_files(MODELS_DIR)
choice = st.sidebar.selectbox(
    "Model file", [str(p) for p in models], index=None, placeholder="No model files in models/"
)
with st.sidebar.expander("Performance", expanded=False):
    imgsz = st.select_slider("Image size", [320, 416, 512, 640, 800, 960], value=640)
    threads = st.number_input("CPU threads (0 = library default)", 0, 64, 2)
    max_det = st.number_input("Max detections per image", 10, 5000, 1000, step=100)
st.sidebar.markdown("**Detection settings**")
conf = st.sidebar.slider("Confidence threshold", 0.01, 0.99, 0.25, 0.01)
iou = st.sidebar.slider("NMS IoU threshold", 0.1, 0.95, 0.45, 0.05)
if st.sidebar.button("Load model", disabled=choice is None, type="primary"):
    st.session_state["model_key"] = (
        choice,
        ui.backend_for(choice),
        int(imgsz),
        int(threads),
        int(max_det),
    )

detector = None
if "model_key" in st.session_state:
    try:
        detector = get_detector(*st.session_state["model_key"])
        detector.conf_threshold, detector.iou_threshold = conf, iou
    except ShelfSightError as exc:
        st.sidebar.error(str(exc))
        del st.session_state["model_key"]
    except Exception as exc:  # noqa: BLE001 - surface any load failure instead of a stack trace
        st.sidebar.error(f"Failed to load model: {exc}")
        del st.session_state["model_key"]
st.sidebar.caption(f"Loaded: {detector.name}" if detector else "No model loaded.")

status = (
    f"✅ Model ready: {detector.name}"
    if detector
    else "⚪ No model loaded: pick one in the sidebar"
)
st.markdown(
    f"""<div class="hero"><h1>ShelfSight Edge</h1>
<p>Retail shelf monitoring on CPU: detect, track and estimate shelf occupancy.</p>
<span class="pill">{status}</span></div>""",
    unsafe_allow_html=True,
)

NEED_MODEL = "Load a model from the sidebar first (put .pt/.onnx files in `models/`)."
tabs = st.tabs(
    [
        "Overview",
        "Image detection",
        "Video analytics",
        "Shelf monitoring",
        "Benchmarking",
        "Reports",
    ]
)

# ---------------------------------------------------------------- overview
with tabs[0]:
    cols = st.columns(4)
    cards = [
        ("🖼️ Image detection", "Upload a photo, get boxes, counts and a CSV."),
        ("🎬 Video analytics", "Track products across frames without double counting."),
        ("🗄️ Shelf monitoring", "Split a shelf into regions: estimated occupancy and alerts."),
        ("⏱️ Benchmarking", "Median and p95 latency, FPS and model size on your CPU."),
    ]
    for col, (title, text) in zip(cols, cards, strict=True):
        col.markdown(
            f'<div class="card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True
        )
    st.write("")
    if detector is None:
        st.markdown(
            "**Get started:** 1) put a `.onnx` or `.pt` file in `models/`  2) choose it in the "
            "sidebar  3) press **Load model**  4) open any tab above."
        )
    st.info(OCCUPANCY_DISCLAIMER)
    c1, c2, c3 = st.columns(3)
    c1.metric("Backend", detector.name.split(":")[0] if detector else "—")
    c2.metric("Model status", "loaded" if detector else "not loaded")
    c3.metric("Model files found", len(models))
    with st.expander("System information", expanded=False):
        st.json(cached_system_info())

# ---------------------------------------------------------------- image
with tabs[1]:
    st.header("Image detection")
    up = st.file_uploader("Upload an image", type=IMG_TYPES, key="img_up")
    if st.button("Run detection", key="run_img"):
        if detector is None:
            st.warning(NEED_MODEL)
        elif up is None:
            st.warning("Upload an image first.")
        else:
            try:
                img = ui.decode_image_bytes(up.getvalue())
                with st.spinner("Detecting…"):
                    det = detector.predict(img)
                st.session_state["image_result"] = (img, det, detector.last_timings.total_ms)
                append_history(HISTORY, {"kind": "image", "file": up.name, "model": detector.name,
                                         "detections": len(det)})  # fmt: skip
            except ShelfSightError as exc:
                st.error(str(exc))
    res = st.session_state.get("image_result")
    if not res:
        st.caption("Upload an image and press **Run detection**. Results appear here.")
    if res:
        img, det, ms = res
        annotated = draw_detections(img, det)
        left, right = st.columns(2)
        left.image(ui.bgr_to_rgb(img), caption="Original", use_container_width=True)
        right.image(
            ui.bgr_to_rgb(annotated),
            caption=f"{len(det)} detections ({ms:.0f} ms, single run)",
            use_container_width=True,
        )
        st.metric("Detections", len(det))
        counts = pd.DataFrame(sorted(det.counts().items()), columns=["class", "detections"])
        st.dataframe(counts, hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button(
            "Download annotated image", ui.png_bytes(annotated), "annotated.png", "image/png"
        )
        d2.download_button(
            "Download detections CSV", det.to_dataframe().to_csv(index=False), "detections.csv"
        )

# ---------------------------------------------------------------- video
with tabs[2]:
    st.header("Video analytics")
    st.caption(
        "Detections per frame, unique tracks and line crossings are different quantities; "
        "see docs/ARCHITECTURE.md."
    )
    vup = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid_up")
    c1, c2, c3 = st.columns(3)
    stride = c1.number_input("Process every Nth frame", 1, 30, 2)
    max_frames = c2.number_input("Max frames (0 = all)", 0, 100000, 300)
    track_on = c3.checkbox("Enable tracking", value=True)
    line_on = st.checkbox("Count crossings of a horizontal line (needs tracking)")
    line_y = st.slider("Line height (fraction of frame)", 0.05, 0.95, 0.5, disabled=not line_on)
    if st.button("Process video", key="run_vid"):
        if detector is None:
            st.warning(NEED_MODEL)
        elif vup is None:
            st.warning("Upload a video first.")
        else:
            src = ui.save_upload(vup.getvalue(), Path(vup.name).suffix or ".mp4")
            out = Path(src).with_name(Path(src).stem + "_annotated.mp4")
            bar = st.progress(0.0, text="Processing…")
            rec = AnalyticsRecorder()

            def on_progress(done: int, total: int) -> None:
                bar.progress(
                    min(done / total, 1.0) if total else 0.0, text=f"Frame {done}/{total or '?'}"
                )

            try:
                stats = process_video(
                    src, detector, out, ByteTracker() if track_on else None, None, 0.2,
                    int(stride), int(max_frames) or None, rec,
                    ((0.0, line_y), (1.0, line_y)) if line_on and track_on else None, on_progress,
                )  # fmt: skip
                playable = ui.maybe_reencode_h264(out)
                st.session_state["video_result"] = (stats.to_dict(), playable, rec)
                append_history(HISTORY, {"kind": "video", "file": vup.name, "model": detector.name,
                                         "frames": stats.frames_processed,
                                         "unique_tracks": stats.unique_tracks})  # fmt: skip
            except ShelfSightError as exc:
                st.error(str(exc))
            finally:
                bar.empty()
                src.unlink(missing_ok=True)
    vres = st.session_state.get("video_result")
    if not vres:
        st.caption("Upload a video, choose options, then press **Process video**.")
    if vres:
        stats_d, vpath, rec = vres
        if vpath.name.endswith("_h264.mp4"):
            st.video(str(vpath))
        else:
            found = ui.find_ffmpeg()
            if found:
                st.info(f"ffmpeg found ({found}) but conversion failed; download below.")
            else:
                st.info("ffmpeg not found: no preview. You can still download below.")
        st.download_button(
            "Download annotated video", vpath.read_bytes(), vpath.name, "video/mp4"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frames processed", stats_d["frames_processed"])
        m2.metric("Mean detections/frame", f"{stats_d['mean_detections_per_frame']:.1f}")
        m3.metric(
            "Unique tracks",
            stats_d["unique_tracks"] if stats_d["unique_tracks"] is not None else "n/a",
        )
        m4.metric(
            "Line crossings",
            stats_d["line_crossings"] if stats_d["line_crossings"] is not None else "n/a",
        )
        st.write(
            f"**End-to-end FPS** (decode+detect+track+analytics+render+write): "
            f"{stats_d['end_to_end_fps']:.2f} | **Detect-only FPS** (pre+inference+post): "
            f"{stats_d['detect_only_fps']:.2f}"
        )
        st.download_button(
            "Download frame analytics CSV",
            rec.frame_dataframe().to_csv(index=False),
            "video_frames.csv",
        )

# ---------------------------------------------------------------- shelf
with tabs[3]:
    st.header("Shelf monitoring")
    st.warning(OCCUPANCY_DISCLAIMER)
    example = ROOT / "configs" / "regions.example.yaml"
    if "regions_df" not in st.session_state:
        st.session_state["regions_df"] = (
            ui.regions_to_df(load_regions(example))
            if example.is_file()
            else pd.DataFrame(columns=ui.REGION_COLUMNS)
        )
    st.caption("Regions are normalized rectangles (0–1). Edit, add or delete rows.")
    edited = st.data_editor(
        st.session_state["regions_df"], num_rows="dynamic", key="regions_editor"
    )
    low_default = st.slider("Default low-occupancy threshold", 0.0, 1.0, 0.2, 0.05)
    sup = st.file_uploader("Shelf image", type=IMG_TYPES, key="shelf_up")
    if st.button("Analyze shelf", key="run_shelf"):
        if detector is None:
            st.warning(NEED_MODEL)
        elif sup is None:
            st.warning("Upload a shelf image first.")
        else:
            try:
                regions = ui.df_to_regions(edited)
                img = ui.decode_image_bytes(sup.getvalue())
                det = detector.predict(img)
                stats_r = analyze_regions(det, regions, img.shape[:2], low_default)
                canvas = draw_regions(
                    draw_detections(img, det, show_conf=False, thickness=1),
                    [(r.name, r.to_pixels(img.shape[:2])) for r in regions],
                    {s.name: s.low_occupancy for s in stats_r},
                    {s.name: f"{s.name}: {s.count} ({s.occupancy_ratio:.0%})" for s in stats_r},
                )
                st.image(ui.bgr_to_rgb(canvas))
                table = pd.DataFrame(
                    [
                        {
                            "region": s.name,
                            "detections": s.count,
                            "est. occupancy": f"{s.occupancy_ratio:.1%}",
                            "low occupancy": "⚠️" if s.low_occupancy else "",
                        }
                        for s in stats_r
                    ]
                )
                st.dataframe(table, hide_index=True)
                for s in stats_r:
                    if s.low_occupancy:
                        st.error(
                            f"Low estimated occupancy in '{s.name}' "
                            f"({s.occupancy_ratio:.1%}); verify physically."
                        )
            except (RegionError, ShelfSightError) as exc:
                st.error(str(exc))

# ---------------------------------------------------------------- benchmark
with tabs[4]:
    st.header("Benchmarking")
    st.caption(
        "Latency = predict() wall time on a decoded image, batch 1 (pre + inference + post); "
        "excludes decode, tracking and rendering. For publishable numbers use scripts/benchmark.py."
    )
    bup = st.file_uploader("Image for the benchmark", type=IMG_TYPES, key="bench_up")
    b1, b2 = st.columns(2)
    warm = b1.number_input("Warm-up runs", 0, 50, 5)
    runs = b2.number_input("Timed runs", 5, 200, 30)
    if st.button("Run bounded benchmark", key="run_bench"):
        if detector is None:
            st.warning(NEED_MODEL)
        elif bup is None:
            st.warning("Upload an image first.")
        else:
            try:
                with st.spinner("Benchmarking…"):
                    r = benchmark_detector(
                        detector, [ui.decode_image_bytes(bup.getvalue())], int(warm), int(runs)
                    )
                st.session_state["bench_result"] = r
            except ShelfSightError as exc:
                st.error(str(exc))
    br = st.session_state.get("bench_result")
    if br:
        lat = br["latency_wall_ms"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Median ms", f"{lat['median_ms']:.1f}")
        k2.metric("p95 ms", f"{lat['p95_ms']:.1f}")
        k3.metric("Throughput FPS", br["throughput_fps"])
        k4.metric("Model size MB", br["model_size_mb"])
        st.caption("Accuracy: Not measured here; run scripts/evaluate.py on a held-out split.")
        if st.button("Save result to reports/benchmarks"):
            name = f"ui_{time.strftime('%Y%m%d_%H%M%S')}.json"
            st.success(f"Saved {save_benchmark(br, REPORTS_DIR / 'benchmarks' / name)}")
    saved = (
        sorted((REPORTS_DIR / "benchmarks").glob("*.json"))
        if (REPORTS_DIR / "benchmarks").is_dir()
        else []
    )
    bench_files = [p for p in saved if not p.stem.endswith("_eval")]
    if bench_files:
        st.subheader("Saved benchmark results")
        st.markdown(
            comparison_table(
                bench_files, {p.stem: p.with_name(p.stem + "_eval.json") for p in bench_files}
            )
        )
    else:
        st.info("No saved benchmarks yet.")

# ---------------------------------------------------------------- reports
with tabs[5]:
    st.header("Reports")
    hist = read_history(HISTORY)
    st.subheader("Processing history")
    if hist.empty:
        st.info("Nothing processed yet.")
    else:
        st.dataframe(hist.iloc[::-1], hide_index=True)
        st.download_button("Export history CSV", hist.to_csv(index=False), "history.csv")
    st.subheader("Dataset and evaluation summaries")
    summaries = sorted(p for p in REPORTS_DIR.glob("*.json")) if REPORTS_DIR.is_dir() else []
    summaries += (
        sorted((REPORTS_DIR / "benchmarks").glob("*_eval.json"))
        if (REPORTS_DIR / "benchmarks").is_dir()
        else []
    )
    if not summaries:
        st.info("No dataset statistics or evaluation files found in reports/ yet.")
    for p in summaries:
        with st.expander(p.name):
            st.json(p.read_text(encoding="utf-8"))
    if bench_files:
        rows = []
        for p in bench_files:
            import json

            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d.get("latency_wall_ms"), dict):
                rows.append({"result": p.stem, "median_ms": d["latency_wall_ms"]["median_ms"]})
        if rows:
            st.plotly_chart(
                px.bar(pd.DataFrame(rows), x="result", y="median_ms", title="Median latency (ms)")
            )