from pathlib import Path

import pandas as pd
import pytest

from shelfsight.analytics.shelf_regions import RegionError, ShelfRegion
from shelfsight.errors import InvalidImageError

ROOT = Path(__file__).resolve().parents[1]


def load_components():
    import sys

    sys.path.insert(0, str(ROOT / "app"))
    import components

    return components


def test_streamlit_app_starts_on_cpu_without_loading_a_model():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "model_key" not in at.session_state  # nothing auto-loaded
    assert len(at.tabs) == 6
    assert any("No model loaded" in c.value for c in at.sidebar.caption)


def test_decode_image_bytes_errors():
    c = load_components()
    with pytest.raises(InvalidImageError, match="empty"):
        c.decode_image_bytes(b"")
    with pytest.raises(InvalidImageError, match="not a readable"):
        c.decode_image_bytes(b"definitely not an image")


def test_decode_image_bytes_ok_and_png_roundtrip():
    import numpy as np

    c = load_components()
    img = np.zeros((8, 9, 3), np.uint8)
    img[:, :, 2] = 200
    assert c.decode_image_bytes(c.png_bytes(img)).shape == (8, 9, 3)


def test_list_model_files_and_backend(tmp_path):
    c = load_components()
    (tmp_path / "a.onnx").write_bytes(b"x")
    (tmp_path / "b.pt").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")
    assert [p.name for p in c.list_model_files(tmp_path)] == ["a.onnx", "b.pt"]
    assert c.list_model_files(tmp_path / "missing") == []
    assert c.backend_for("x.ONNX") == "onnx" and c.backend_for("y.pt") == "pytorch"


def test_regions_table_validation():
    c = load_components()
    good = c.regions_to_df([ShelfRegion("a", 0, 0, 1, 0.5, 0.3)])
    assert c.df_to_regions(good)[0].low_threshold == 0.3
    bad_name = pd.DataFrame(
        [{"name": "", "x1": 0, "y1": 0, "x2": 1, "y2": 1, "low_threshold": None}]
    )
    with pytest.raises(RegionError, match="name"):
        c.df_to_regions(bad_name)
    bad_box = pd.DataFrame(
        [{"name": "a", "x1": 0.6, "y1": 0, "x2": 0.4, "y2": 1, "low_threshold": None}]
    )
    with pytest.raises(RegionError):
        c.df_to_regions(bad_box)
    nan = pd.DataFrame(
        [{"name": "a", "x1": None, "y1": 0, "x2": 1, "y2": 1, "low_threshold": None}]
    )
    with pytest.raises(RegionError):
        c.df_to_regions(nan)
    with pytest.raises(RegionError, match="at least one"):
        c.df_to_regions(pd.DataFrame(columns=c.REGION_COLUMNS))
    dup = pd.concat([good, good])
    with pytest.raises(RegionError, match="unique"):
        c.df_to_regions(dup)


def test_app_renders_with_a_loaded_model(tiny_onnx):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.session_state["model_key"] = (str(tiny_onnx), "onnx", 64, 1, 100)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Loaded: onnx:tiny.onnx" in c.value for c in at.sidebar.caption)
