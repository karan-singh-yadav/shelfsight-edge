import pytest

from shelfsight.data.synthetic import make_synthetic_sku_dataset


@pytest.fixture
def synth(tmp_path):
    """Small synthetic SKU-format dataset: returns (root, {split: csv_path})."""
    root = tmp_path / "synth"
    csvs = make_synthetic_sku_dataset(root, n_train=4, n_val=2, n_test=2, seed=1)
    return root, csvs


@pytest.fixture
def tiny_onnx(tmp_path):
    """Hand-built YOLOv8-shaped ONNX model: input (1,3,64,64) -> output (1,5,64).

    Zero conv weights + constant bias => every one of the 64 anchors predicts the same box
    (cx=32, cy=32, w=20, h=20, score=0.9), so NMS must collapse them to ONE detection at
    (22,22,42,42) for a 64x64 input image.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.zeros((5, 3, 8, 8), dtype=np.float32), "W")
    bias = numpy_helper.from_array(np.array([32, 32, 20, 20, 0.9], dtype=np.float32), "B")
    shape = numpy_helper.from_array(np.array([1, 5, 64], dtype=np.int64), "S")
    nodes = [
        helper.make_node("Conv", ["images", "W", "B"], ["c"], kernel_shape=[8, 8], strides=[8, 8]),
        helper.make_node("Reshape", ["c", "S"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 64])],
        [weight, bias, shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    entry = model.metadata_props.add()
    entry.key, entry.value = "names", "{0: 'object'}"
    onnx.checker.check_model(model)
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path
