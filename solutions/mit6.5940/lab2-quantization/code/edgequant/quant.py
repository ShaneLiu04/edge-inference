"""Linear quantization primitives and activation calibrators.

Conventions follow the TFLite / CMSIS-NN integer contract:

* weights: symmetric, zero point 0, range [-(2^(b-1)-1), 2^(b-1)-1], per output
  channel or per tensor;
* activations: asymmetric uint-style range [0, 2^b - 1] with a zero point, one
  scale per tensor. A ReLU6 output then uses the whole range.

Calibrators map a sample of activation values to a clipping range [lo, hi].
``minmax`` keeps everything, ``percentile`` clips the tails, ``mse`` searches
the clip that minimizes the quantization MSE on the sample, ``kl`` is the
TensorRT entropy calibrator (2048-bin histogram, KL divergence between the
clipped reference and its quantized version).
"""

from __future__ import annotations

import numpy as np
import torch


def weight_qmax(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def weight_scale(w: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    """Symmetric scale, broadcastable to ``w`` (output channel on dim 0)."""
    if per_channel:
        amax = w.detach().abs().reshape(w.shape[0], -1).amax(1)
        shape = (-1,) + (1,) * (w.dim() - 1)
        amax = amax.reshape(shape)
    else:
        amax = w.detach().abs().amax()
    return torch.clamp(amax, min=1e-12) / weight_qmax(bits)


def fake_quant_weight(w: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    scale = weight_scale(w, bits, per_channel)
    q = weight_qmax(bits)
    return torch.clamp(torch.round(w / scale), -q, q) * scale


def act_qparams(lo: float, hi: float, bits: int) -> tuple[float, int]:
    """Asymmetric scale and zero point for [lo, hi]. The range always includes 0."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    qmax = 2 ** bits - 1
    scale = max(hi - lo, 1e-12) / qmax
    zero_point = int(np.clip(round(-lo / scale), 0, qmax))
    return scale, zero_point


def fake_quant_act(x: torch.Tensor, scale: float, zero_point: int, bits: int) -> torch.Tensor:
    qmax = 2 ** bits - 1
    q = torch.clamp(torch.round(x / scale) + zero_point, 0, qmax)
    return (q - zero_point) * scale


class STEQuant(torch.autograd.Function):
    """Straight-through estimator: forward is the fake quant, backward passes the
    gradient inside the clip range and zeroes it outside."""

    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        q = torch.round(x / scale) + zero_point
        ctx.save_for_backward((q >= qmin) & (q <= qmax))
        return (torch.clamp(q, qmin, qmax) - zero_point) * scale

    @staticmethod
    def backward(ctx, grad):
        (inside,) = ctx.saved_tensors
        return grad * inside, None, None, None, None


def ste_weight(w: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    scale = weight_scale(w, bits, per_channel)
    q = weight_qmax(bits)
    return STEQuant.apply(w, scale, 0, -q, q)


def ste_act(x: torch.Tensor, scale: float, zero_point: int, bits: int) -> torch.Tensor:
    return STEQuant.apply(x, scale, zero_point, 0, 2 ** bits - 1)


# ----------------------------------------------------------------------------- calibration

def quant_mse(x: np.ndarray, lo: float, hi: float, bits: int) -> float:
    scale, zp = act_qparams(lo, hi, bits)
    q = np.clip(np.round(x / scale) + zp, 0, 2 ** bits - 1)
    return float(np.mean(((q - zp) * scale - x) ** 2))


def calib_minmax(x: np.ndarray, bits: int = 8) -> tuple[float, float]:
    return float(x.min()), float(x.max())


def calib_percentile(x: np.ndarray, bits: int = 8, pct: float = 99.99) -> tuple[float, float]:
    lo = float(np.percentile(x, 100.0 - pct)) if x.min() < 0 else 0.0
    return lo, float(np.percentile(x, pct))


def calib_mse(x: np.ndarray, bits: int = 8, grid: int = 100) -> tuple[float, float]:
    """Shrink [min, max] by a common factor in (0, 1] and keep the lowest MSE."""
    lo0, hi0 = float(x.min()), float(x.max())
    best, best_range = float("inf"), (lo0, hi0)
    for r in np.linspace(1.0 / grid, 1.0, grid):
        lo, hi = lo0 * r, hi0 * r
        err = quant_mse(x, lo, hi, bits)
        if err < best:
            best, best_range = err, (lo, hi)
    return best_range


def _smooth(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Give empty bins eps of mass, taken evenly from the non-empty ones (MXNet's calibrator)."""
    zeros = p == 0
    n_zero, n_nonzero = int(zeros.sum()), int((~zeros).sum())
    if n_nonzero == 0:
        return p
    return p + eps * zeros - (eps * n_zero / n_nonzero) * ~zeros


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p, q = _smooth(p), _smooth(q)
    mask = p > 0
    q = np.where(q > 0, q, 1e-12)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def calib_kl(x: np.ndarray, bits: int = 8, bins: int = 2048) -> tuple[float, float]:
    """TensorRT-style entropy calibration on |x|, symmetric threshold search.

    For a non-negative tensor the returned range is [0, threshold]. For a signed
    tensor it is [-threshold, threshold].
    """
    levels = 2 ** (bits - 1) if x.min() < 0 else 2 ** bits
    ax = np.abs(x).ravel()
    amax = float(ax.max())
    if amax == 0:
        return 0.0, 0.0
    hist, edges = np.histogram(ax, bins=bins, range=(0.0, amax))
    hist = hist.astype(np.float64)
    tail = np.concatenate([np.cumsum(hist[::-1])[::-1], [0.0]])
    best_kl, best_i = float("inf"), bins
    for i in range(levels, bins + 1):
        ref = hist[:i].copy()
        ref[i - 1] += tail[i]
        # Merge the first i bins into `levels` groups, then spread each group's
        # mass evenly over the bins that are non-empty in the reference.
        starts = (np.arange(levels) * i) // levels
        group_of = np.repeat(np.arange(levels), np.diff(np.append(starts, i)))
        nonzero = ref > 0
        sums = np.add.reduceat(hist[:i], starts)
        counts = np.add.reduceat(nonzero.astype(np.float64), starts)
        per_bin = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        cand = np.where(nonzero, per_bin[group_of], 0.0)
        p = ref / ref.sum()
        q = cand / max(cand.sum(), 1e-12)
        kl = _kl(p, q)
        if kl < best_kl:
            best_kl, best_i = kl, i
    threshold = float(edges[best_i])
    return (-threshold, threshold) if x.min() < 0 else (0.0, threshold)


def calib_kl_nonzero(x: np.ndarray, bits: int = 8, bins: int = 2048) -> tuple[float, float]:
    """KL search on the non-zero values only.

    After ReLU most of the mass sits exactly at 0, which the zero point represents
    exactly. Left in, that spike gets smeared over i/levels bins and the divergence
    rewards small thresholds; on ReLU6 outputs plain `calib_kl` clips at about 1/4
    of the range.
    """
    nz = x[x != 0]
    if nz.size == 0:
        return 0.0, 0.0
    lo, hi = calib_kl(nz, bits, bins)
    return (lo, hi) if x.min() < 0 else (0.0, hi)


CALIBRATORS = {
    "minmax": calib_minmax,
    "percentile": calib_percentile,
    "mse": calib_mse,
    "kl": calib_kl,
    "kl_nonzero": calib_kl_nonzero,
}


def sqnr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Signal-to-quantization-noise ratio in dB."""
    noise = np.sum((reference - candidate) ** 2)
    signal = np.sum(reference ** 2)
    if noise == 0:
        return float("inf")
    return float(10.0 * np.log10(signal / noise))
