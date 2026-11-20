"""ONNX export and ONNX Runtime static int8 quantization (QDQ), the CPU stand-in for an MCU int8 kernel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


def export_onnx(model: nn.Module, resolution: int, path: Path, dynamic: bool = True) -> Path:
    x = torch.zeros(1, 3, resolution, resolution)
    axes = {"x": {0: "n"}, "logits": {0: "n"}} if dynamic else None
    torch.onnx.export(model.eval(), x, str(path), input_names=["x"], output_names=["logits"],
                      dynamic_axes=axes, opset_version=17, dynamo=False)
    return path


def session(path: Path, threads: int = 1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def quantize(fp32: Path, out: Path, calib: np.ndarray, method: str = "minmax", per_channel: bool = True,
             activation: str = "uint8", batch: int = 64) -> Path:
    """Static QDQ int8: int8 weights, uint8 or int8 activations, ranges from ``calib`` (N x 3 x r x r).

    ``method="entropy"`` is ORT's default: a 128-bin histogram searched for the best of 128 quantized
    bins, which leaves one candidate threshold, the full range, so it returns MinMax byte for byte.
    ``entropy2048`` passes ``num_bins=2048`` to the calibrator, which ``quantize_static`` does not expose.
    """
    import importlib

    import onnxruntime as ort
    qmod = importlib.import_module("onnxruntime.quantization.quantize")  # the package re-exports a function of this name
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.batches = [calib[i:i + batch] for i in range(0, len(calib), batch)]
            self.i = 0

        def get_next(self):
            if self.i >= len(self.batches):
                return None
            self.i += 1
            return {"x": self.batches[self.i - 1]}

        def rewind(self):
            self.i = 0

    pre = out.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    methods = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
               "entropy2048": CalibrationMethod.Entropy, "percentile": CalibrationMethod.Percentile}
    orig_calibrator = qmod.create_calibrator

    def calibrator(*a, **kw):
        if method == "entropy2048":
            kw["extra_options"] = {**(kw.get("extra_options") or {}), "num_bins": 2048}
        return orig_calibrator(*a, **kw)
    act = {"uint8": QuantType.QUInt8, "int8": QuantType.QInt8}[activation]
    # The calibrator opens its own sessions with one thread per core; keep them small.
    base = ort.InferenceSession

    class Few(base):
        def __init__(self, p, sess_options=None, *a, **kw):
            so = sess_options or ort.SessionOptions()
            so.intra_op_num_threads = 4
            so.inter_op_num_threads = 1
            super().__init__(p, so, *a, **kw)

    ort.InferenceSession = Few
    qmod.create_calibrator = calibrator
    try:
        quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ, per_channel=per_channel,
                        activation_type=act, weight_type=QuantType.QInt8, calibrate_method=methods[method],
                        extra_options={"ActivationSymmetric": activation == "int8"})
    finally:
        ort.InferenceSession = base
        qmod.create_calibrator = orig_calibrator
        pre.unlink(missing_ok=True)
    return out


def ort_accuracy(path: Path, x: np.ndarray, y: np.ndarray, threads: int = 8, batch: int = 256) -> tuple[float, np.ndarray]:
    sess = session(path, threads)
    logits = np.concatenate([sess.run(None, {"x": x[i:i + batch]})[0] for i in range(0, len(x), batch)])
    return float((logits.argmax(1) == y).mean() * 100), logits
