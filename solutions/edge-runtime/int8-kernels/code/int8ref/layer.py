"""A quantized conv layer: generation, int32 reference, and the file format ``int8 --mode=conv`` reads.

Layout conventions:
* x is uint8 NHWC (batch 1) with zero point zx.
* w is int8 symmetric per output channel, stored [OC][KH][KW][IC/groups].
* bias is int32 in units of sx * sw[oc].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class QConv:
    x: np.ndarray        # (H, W, IC) uint8
    w: np.ndarray        # (OC, K, K, ICg) int8
    bias: np.ndarray     # (OC,) int32
    stride: int
    pad: int
    groups: int
    zx: int
    sx: float
    sw: np.ndarray       # (OC,) float64
    zy: int
    sy: float
    name: str = "layer"
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return int(self.w.shape[1])

    @property
    def oc(self) -> int:
        return int(self.w.shape[0])

    def out_hw(self) -> tuple[int, int]:
        h, w, _ = self.x.shape
        return ((h + 2 * self.pad - self.k) // self.stride + 1,
                (w + 2 * self.pad - self.k) // self.stride + 1)


def acc_int32(layer: QConv) -> np.ndarray:
    """sum (x - zx) * w + bias in int64, returned as (OH, OW, OC). Padding is real zero, i.e. x == zx."""
    x = layer.x.astype(np.int64) - layer.zx
    h, w, ic = x.shape
    k, s, p, g = layer.k, layer.stride, layer.pad, layer.groups
    xp = np.zeros((h + 2 * p, w + 2 * p, ic), dtype=np.int64)
    xp[p:p + h, p:p + w] = x
    oh, ow = layer.out_hw()
    icg, ocg = ic // g, layer.oc // g
    wt = layer.w.astype(np.int64)
    out = np.zeros((oh, ow, layer.oc), dtype=np.int64)
    for kh in range(k):
        for kw in range(k):
            patch = xp[kh:kh + s * oh:s, kw:kw + s * ow:s]  # (OH, OW, IC)
            for gi in range(g):
                xs = patch[:, :, gi * icg:(gi + 1) * icg]
                ws = wt[gi * ocg:(gi + 1) * ocg, kh, kw, :]  # (ocg, icg)
                out[:, :, gi * ocg:(gi + 1) * ocg] += xs @ ws.T
    return out + layer.bias.astype(np.int64)


def random_layer(rng: np.random.Generator, h: int, ic: int, oc: int, k: int, stride: int = 1,
                 groups: int = 1, name: str = "layer") -> QConv:
    """Realistic statistics: post-ReLU activations, Gaussian weights with per-channel spread."""
    pad = k // 2
    xr = np.maximum(rng.normal(0.2, 1.0, (h, h, ic)), 0)
    sx = float(xr.max() / 255)
    zx = 0
    x = np.clip(np.round(xr / sx) + zx, 0, 255).astype(np.uint8)
    icg = ic // groups
    spread = np.exp(rng.normal(0, 0.7, oc))
    wr = rng.normal(0, 1, (oc, k, k, icg)) * spread[:, None, None, None] / np.sqrt(k * k * icg)
    sw = np.abs(wr).reshape(oc, -1).max(1) / 127
    w = np.clip(np.round(wr / sw[:, None, None, None]), -127, 127).astype(np.int8)
    br = rng.normal(0, 0.1, oc)
    bias = np.round(br / (sx * sw)).astype(np.int32)
    layer = QConv(x, w, bias, stride, pad, groups, zx, sx, sw, 0, 1.0, name)
    real = acc_int32(layer) * (sx * sw)
    # Output is the pre-activation of the next layer: asymmetric range from the observed min/max.
    lo, hi = float(min(real.min(), 0)), float(max(real.max(), 0))
    layer.sy = (hi - lo) / 255
    layer.zy = int(np.clip(round(-lo / layer.sy), 0, 255))
    return layer


def write_layer(layer: QConv, d: Path) -> None:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    h, w, ic = layer.x.shape
    head = [h, w, ic, layer.oc, layer.k, layer.stride, layer.pad, layer.groups, layer.zx, layer.zy]
    # float() first: under NumPy 2 repr(np.float64) is "np.float64(...)", which C++ cannot parse.
    lines = [" ".join(map(str, map(int, head))) + f" {float(layer.sx)!r} {float(layer.sy)!r}"]
    lines += [repr(float(v)) for v in layer.sw]
    (d / "meta.txt").write_text("\n".join(lines) + "\n")
    layer.x.astype(np.uint8).tofile(d / "x.bin")
    np.ascontiguousarray(layer.w, dtype=np.int8).tofile(d / "w.bin")
    layer.bias.astype(np.int32).tofile(d / "b.bin")


def read_outputs(layer: QConv, d: Path) -> dict[str, np.ndarray]:
    oh, ow = layer.out_hw()
    shape = (oh, ow, layer.oc)
    out = {"acc": np.fromfile(Path(d) / "acc.bin", dtype=np.int32).reshape(shape)}
    for style in ("fixed", "float", "double"):
        out[style] = np.fromfile(Path(d) / f"y_{style}.bin", dtype=np.uint8).reshape(shape)
    return out
