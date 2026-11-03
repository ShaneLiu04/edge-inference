"""Conv-BN fold and symmetric int8 weight scales.

The deployment contract used by CMSIS-NN style runtimes: fold batch-norm into
the conv, quantize weights per output channel with zero-point 0, and add bias
in the int32 accumulator whose unit is weight_scale * input_scale.
"""

from __future__ import annotations

import numpy as np

QMAX = 127


def fold_bn(
    weight: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold inference BN into conv weights of shape (OC, IC, KH, KW)."""
    inv = gamma / np.sqrt(var + eps)
    folded_w = weight * inv[:, None, None, None]
    folded_b = (bias - mean) * inv + beta
    return folded_w, folded_b


def bn_inference(
    y: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    inv = gamma / np.sqrt(var + eps)
    return (y - mean.reshape(1, -1, 1, 1)) * inv.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)


def conv2d_valid(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Stride-1 valid conv. x is NCHW, weight is (OC, IC, KH, KW)."""
    n, channels, height, width = x.shape
    oc, ic, kh, kw = weight.shape
    if channels != ic:
        raise ValueError(f"channel mismatch: input {channels}, weight {ic}")
    oh, ow = height - kh + 1, width - kw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("kernel larger than the input")
    y = np.empty((n, oc, oh, ow), dtype=np.float64)
    for i in range(oh):
        for j in range(ow):
            patch = x[:, :, i : i + kh, j : j + kw]
            y[:, :, i, j] = np.einsum("nchw,ochw->no", patch, weight, optimize=True)
    return y + bias.reshape(1, -1, 1, 1)


def quantize_symmetric(values: np.ndarray, channel_axis: int | None, qmax: int = QMAX):
    """Return integer codes, the scale broadcast to ``values.shape``, and dequantized values.

    channel_axis None is per-tensor. Otherwise the scale is reduced over every
    axis except that one, which is the CMSIS-NN per-output-channel weight scale.
    """
    if channel_axis is None:
        axes = tuple(range(values.ndim))
    else:
        axes = tuple(i for i in range(values.ndim) if i != channel_axis)
    amax = np.max(np.abs(values), axis=axes, keepdims=True)
    scale = np.maximum(amax, 1e-12) / qmax
    codes = np.clip(np.rint(values / scale), -qmax, qmax)
    return codes, scale, codes * scale


def rel_rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    denom = np.sqrt(np.mean(reference**2))
    if denom == 0:
        raise ValueError("reference is zero")
    return float(np.sqrt(np.mean((candidate - reference) ** 2)) / denom)


def per_channel_rel_rmse(reference: np.ndarray, candidate: np.ndarray, axis: int = 0) -> np.ndarray:
    """Relative RMSE reduced over every axis except ``axis``."""
    reduce_axes = tuple(i for i in range(reference.ndim) if i != axis)
    diff = np.mean((candidate - reference) ** 2, axis=reduce_axes)
    denom = np.mean(reference**2, axis=reduce_axes)
    if np.any(denom == 0):
        raise ValueError("a channel of the reference is zero")
    return np.sqrt(diff / denom)
