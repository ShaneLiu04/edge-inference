"""Run a ``QConv`` through ONNX Runtime's QLinearConv, the deployed int8 kernel on x86/ARM CPUs."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from .layer import QConv


def qlinearconv_model(layer: QConv) -> onnx.ModelProto:
    h, w, ic = layer.x.shape
    oh, ow = layer.out_hw()
    k, p = layer.k, layer.pad
    w_oihw = np.ascontiguousarray(layer.w.transpose(0, 3, 1, 2))  # [OC][ICg][KH][KW]
    inits = [
        numpy_helper.from_array(np.array(layer.sx, dtype=np.float32), "x_scale"),
        numpy_helper.from_array(np.array(layer.zx, dtype=np.uint8), "x_zp"),
        numpy_helper.from_array(w_oihw.astype(np.int8), "w"),
        numpy_helper.from_array(layer.sw.astype(np.float32), "w_scale"),
        numpy_helper.from_array(np.zeros(layer.oc, dtype=np.int8), "w_zp"),
        numpy_helper.from_array(np.array(layer.sy, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.array(layer.zy, dtype=np.uint8), "y_zp"),
        numpy_helper.from_array(layer.bias.astype(np.int32), "bias"),
    ]
    node = helper.make_node(
        "QLinearConv",
        ["x", "x_scale", "x_zp", "w", "w_scale", "w_zp", "y_scale", "y_zp", "bias"],
        ["y"], kernel_shape=[k, k], pads=[p, p, p, p], strides=[layer.stride] * 2, group=layer.groups)
    graph = helper.make_graph(
        [node], layer.name,
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, ic, h, w])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [1, layer.oc, oh, ow])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def run_qlinearconv(layer: QConv, threads: int = 1) -> np.ndarray:
    """Returns y as (OH, OW, OC) uint8."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(qlinearconv_model(layer).SerializeToString(), so,
                                providers=["CPUExecutionProvider"])
    x = np.ascontiguousarray(layer.x.transpose(2, 0, 1)[None])
    (y,) = sess.run(None, {"x": x})
    return y[0].transpose(1, 2, 0)
