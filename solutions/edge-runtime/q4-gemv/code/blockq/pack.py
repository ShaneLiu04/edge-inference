"""llama.cpp block quantizers in NumPy, byte-compatible with ggml's Q4_0, Q4_1 and Q8_0.

Q8_0  { fp16 d;          int8  qs[32] }  34 B / 32 weights   w = d * q,            d = amax / 127
Q4_0  { fp16 d;          uint8 qs[16] }  18 B / 32 weights   w = d * (q - 8),      d = max / -8
Q4_1  { fp16 d; fp16 m;  uint8 qs[16] }  20 B / 32 weights   w = d * q + m,        d = (max - min) / 15

Q4_0's ``max`` keeps the sign of the largest-magnitude element, so that element maps exactly
to q = 0 (value -8 d) and the grid spends its extra level on the side the outlier is not.
In Q4_0 and Q4_1 byte j holds element j (low nibble) and element j + 16 (high nibble).
"""

from __future__ import annotations

import numpy as np

QK = 32
BLOCK_BYTES = {"q8_0": 34, "q4_0": 18, "q4_1": 20}


def _blocks(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32)
    if w.shape[-1] % QK:
        raise ValueError(f"last dimension {w.shape[-1]} is not a multiple of {QK}")
    return w.reshape(-1, QK)


def quantize_q8_0(w: np.ndarray) -> np.ndarray:
    b = _blocks(w)
    d = np.abs(b).max(axis=1) / 127
    with np.errstate(divide="ignore"):
        inv = np.where(d > 0, 1 / d, 0).astype(np.float32)
    q = np.rint(b * inv[:, None]).astype(np.int8)
    out = np.empty((len(b), 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(*np.shape(w)[:-1], -1)


def quantize_q4_0(w: np.ndarray) -> np.ndarray:
    b = _blocks(w)
    idx = np.abs(b).argmax(axis=1)
    mx = b[np.arange(len(b)), idx]
    d = (mx / -8).astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(d != 0, 1 / d, 0).astype(np.float32)
    # ggml: (int8_t)(x * id + 8.5f) truncates toward zero, then clamps to 15.
    q = np.minimum(15, np.trunc(b * inv[:, None] + np.float32(8.5))).astype(np.uint8)
    out = np.empty((len(b), 18), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out.reshape(*np.shape(w)[:-1], -1)


def quantize_q4_1(w: np.ndarray) -> np.ndarray:
    b = _blocks(w)
    lo, hi = b.min(axis=1), b.max(axis=1)
    d = ((hi - lo) / 15).astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(d != 0, 1 / d, 0).astype(np.float32)
    q = np.minimum(15, np.trunc((b - lo[:, None]) * inv[:, None] + np.float32(0.5))).astype(np.uint8)
    out = np.empty((len(b), 20), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = lo.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 4:] = q[:, :16] | (q[:, 16:] << 4)
    return out.reshape(*np.shape(w)[:-1], -1)


def dequantize(packed: np.ndarray, fmt: str) -> np.ndarray:
    nbytes = BLOCK_BYTES[fmt]
    p = np.asarray(packed, dtype=np.uint8).reshape(-1, nbytes)
    d = p[:, :2].copy().view(np.float16).astype(np.float32)
    if fmt == "q8_0":
        vals = d * p[:, 2:].view(np.int8).astype(np.float32)
    else:
        qs = p[:, 4:] if fmt == "q4_1" else p[:, 2:]
        q = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.float32)
        if fmt == "q4_0":
            vals = d * (q - 8)
        else:
            m = p[:, 2:4].copy().view(np.float16).astype(np.float32)
            vals = d * q + m
    lead = np.shape(packed)[:-1]
    return vals.reshape(*lead, -1)


QUANTIZERS = {"q8_0": quantize_q8_0, "q4_0": quantize_q4_0, "q4_1": quantize_q4_1}


def bits_per_weight(fmt: str) -> float:
    return BLOCK_BYTES[fmt] * 8 / QK


def roundtrip(w: np.ndarray, fmt: str) -> np.ndarray:
    return dequantize(QUANTIZERS[fmt](w), fmt).reshape(np.shape(w))


def sqnr_db(ref: np.ndarray, got: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    noise = ((ref - got) ** 2).sum()
    return float(10 * np.log10((ref ** 2).sum() / max(noise, 1e-300)))
