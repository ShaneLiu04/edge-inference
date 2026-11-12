"""NumPy model of TinyChatEngine's W4A8 x86 kernel, to test the arithmetic without the engine.

QM_x86 layout: 64 consecutive weights (two 32-wide blocks) live in 32 bytes; byte j holds weight j
in its low nibble and weight j + 32 in its high nibble, each stored as w + 8 in [0, 15].
The AVX2 kernel multiplies with `_mm256_maddubs_epi16(|w|, sign(a, w))`, which needs an unsigned
left operand; with |w| <= 8 and |a| <= 128 every int16 pair sum is at most 2048, so it never saturates.
`sign(a, w)` negates a, and -(-128) wraps to -128, so activations must stay in [-127, 127]; the
engine's quantizer guarantees that with scale = max|a| / 127.
"""

from __future__ import annotations

import numpy as np

BLOCK = 32


def pack_qm_x86(w: np.ndarray) -> np.ndarray:
    """w: int8 in [-8, 7], length a multiple of 64 -> uint8 bytes (half the length)."""
    w = w.reshape(-1, 2, BLOCK).astype(np.int16) + 8
    return (w[:, 0] | (w[:, 1] << 4)).astype(np.uint8).reshape(-1)


def unpack_qm_x86(packed: np.ndarray) -> np.ndarray:
    p = packed.reshape(-1, BLOCK).astype(np.int16)
    lo = (p & 0x0F) - 8
    hi = (p >> 4) - 8
    return np.stack([lo, hi], axis=1).reshape(-1).astype(np.int8)


def sign_epi8(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """_mm256_sign_epi8: a where b > 0, -a where b < 0, 0 where b == 0 (int8 wraparound on -(-128))."""
    a16 = a.astype(np.int16)
    out = np.where(b > 0, a16, np.where(b < 0, -a16, 0))
    return ((out + 128) % 256 - 128).astype(np.int8)


def maddubs_epi16(u: np.ndarray, s: np.ndarray) -> np.ndarray:
    """_mm256_maddubs_epi16: u as uint8, s as int8, adjacent products summed with int16 saturation."""
    prod = u.astype(np.uint8).astype(np.int32) * s.astype(np.int32)
    pair = prod.reshape(-1, 2).sum(axis=1)
    return np.clip(pair, -32768, 32767).astype(np.int16)


def block_dot_avx2(packed64: np.ndarray, a64: np.ndarray) -> tuple[int, int]:
    """Integer dot products of the two 32-wide blocks, computed the way the patched kernel does."""
    p = packed64.astype(np.int16)
    w0 = ((p & 0x0F) - 8).astype(np.int8)
    w1 = ((p >> 4) - 8).astype(np.int8)
    sums = []
    for w, a in ((w0, a64[:BLOCK]), (w1, a64[BLOCK:])):
        ax = sign_epi8(w, w)  # |w|
        sy = sign_epi8(a, w)
        pairs = maddubs_epi16(ax, sy)
        sums.append(int(pairs.astype(np.int32).sum()))
    return sums[0], sums[1]


def w4a8_matvec(packed: np.ndarray, w_scales: np.ndarray, a_int8: np.ndarray, a_scales: np.ndarray,
                n: int, k: int) -> np.ndarray:
    """C[col] = sum over blocks of s_w * s_a * int_dot, the reference kernel's math."""
    out = np.zeros(n, dtype=np.float64)
    for col in range(n):
        acc = 0.0
        for b2 in range(k // 64):
            base = (col * k + b2 * 64) // 2
            s0, s1 = block_dot_avx2(packed[base:base + 32], a_int8[b2 * 64:(b2 + 1) * 64])
            acc += s0 * w_scales[(col * k) // BLOCK + 2 * b2] * a_scales[2 * b2]
            acc += s1 * w_scales[(col * k) // BLOCK + 2 * b2 + 1] * a_scales[2 * b2 + 1]
        out[col] = acc
    return out
