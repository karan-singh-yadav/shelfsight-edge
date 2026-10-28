import importlib

import pytest


@pytest.mark.parametrize("name", ["numpy", "pandas", "cv2", "PIL", "yaml", "psutil", "shelfsight"])
def test_required_imports(name):
    importlib.import_module(name)


def test_opencv_roundtrip_without_gpu(tmp_path):
    import cv2
    import numpy as np

    img = np.zeros((32, 48, 3), dtype=np.uint8)
    img[:, :24] = (0, 255, 0)
    path = tmp_path / "x.png"
    assert cv2.imwrite(str(path), img)
    loaded = cv2.imread(str(path))
    assert loaded is not None and loaded.shape == (32, 48, 3)
    assert (loaded == img).all()
