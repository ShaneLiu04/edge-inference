"""Integer requantization in NumPy, bit-exact with ``cpp/int8.cpp``.

Three ways to turn an int32 accumulator back into uint8:

* ``fixed``  TFLite / gemmlowp: Q31 multiplier, SaturatingRoundingDoublingHighMul,
  RoundingDivideByPOT. Pure integer, what CMSIS-NN and TFLite Micro run.
* ``float``  ONNX Runtime / MLAS: multiply in fp32, round half to even.
* ``fixed_single`` TFLite built with TFLITE_SINGLE_ROUNDING: one 64-bit rounding.
* ``double`` exact real multiply, round half away from zero. The reference.

``fixed`` rounds twice, the first time at resolution 2^-shift of the output
unit, so it disagrees with exact rounding by one step on a fraction of
outputs even without ties. The others differ only on exact ties.
"""

from __future__ import annotations

import math

import numpy as np

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def quantize_multiplier(m: float) -> tuple[int, int]:
    """Real multiplier m > 0 as (m0, shift) with m ~= m0 * 2^-31 * 2^-shift, m0 in [2^30, 2^31)."""
    if m <= 0:
        return 0, 0
    q, exp = math.frexp(m)
    q31 = round(q * (1 << 31))
    if q31 == 1 << 31:
        q31 //= 2
        exp += 1
    return int(q31), -exp


def srdhm(a: np.ndarray, b: int | np.ndarray) -> np.ndarray:
    """SaturatingRoundingDoublingHighMul: round(a * b / 2^31) in int64.

    The nudge is +2^30 or 1 - 2^30 followed by truncation, so exact ties go toward +inf.
    """
    a64 = np.asarray(a, dtype=np.int64)
    ab = a64 * np.asarray(b, dtype=np.int64)
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    t = ab + nudge
    # C++ integer division truncates toward zero; numpy // floors.
    out = np.where(t >= 0, t >> 31, -((-t) >> 31))
    sat = (a64 == INT32_MIN) & (np.asarray(b) == INT32_MIN)
    return np.where(sat, INT32_MAX, out).astype(np.int64)


def rounding_divide_by_pot(x: np.ndarray, exponent: int | np.ndarray) -> np.ndarray:
    """x / 2^exponent rounded to nearest, ties away from zero (gemmlowp RoundingDivideByPOT)."""
    x = np.asarray(x, dtype=np.int64)
    e = np.asarray(exponent, dtype=np.int64)
    mask = (np.int64(1) << e) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (x < 0)
    return (x >> e) + (remainder > threshold)


def requant_fixed(acc: np.ndarray, m0: np.ndarray, shift: np.ndarray) -> np.ndarray:
    m0 = np.asarray(m0, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    left = np.maximum(-shift, 0)
    right = np.maximum(shift, 0)
    a = np.asarray(acc, dtype=np.int64) << left
    return rounding_divide_by_pot(srdhm(a, m0), right)


def requant_fixed_single(acc: np.ndarray, m0: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """TFLITE_SINGLE_ROUNDING: one rounding of acc * m0 / 2^(31 + shift), ties toward +inf."""
    m0 = np.asarray(m0, dtype=np.int64)
    total = 31 + np.asarray(shift, dtype=np.int64)
    prod = np.asarray(acc, dtype=np.int64) * m0
    return (prod + (np.int64(1) << (total - 1))) >> total


def requant_float(acc: np.ndarray, m: np.ndarray) -> np.ndarray:
    """fp32 multiply and round half to even; np.rint matches std::nearbyintf in the default mode."""
    prod = np.asarray(acc, dtype=np.float32) * np.asarray(m, dtype=np.float32)
    return np.rint(prod).astype(np.int64)


def requant_double(acc: np.ndarray, m: np.ndarray) -> np.ndarray:
    v = np.asarray(acc, dtype=np.float64) * np.asarray(m, dtype=np.float64)
    return (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int64)


def clamp_u8(v: np.ndarray, zy: int) -> np.ndarray:
    return np.clip(np.asarray(v, dtype=np.int64) + zy, 0, 255).astype(np.uint8)


def multipliers(sx: float, sw: np.ndarray, sy: float) -> dict[str, np.ndarray]:
    """Per-channel multipliers for the three styles, computed the way each runtime computes them."""
    sw = np.asarray(sw, dtype=np.float64)
    md = sx * sw / sy
    mf = (np.float32(sx) * sw.astype(np.float32)) / np.float32(sy)
    fixed = [quantize_multiplier(float(m)) for m in md]
    return {
        "double": md,
        "float": mf.astype(np.float32),
        "m0": np.array([f[0] for f in fixed], dtype=np.int64),
        "shift": np.array([f[1] for f in fixed], dtype=np.int64),
    }


def requant_all(acc: np.ndarray, sx: float, sw: np.ndarray, sy: float, zy: int) -> dict[str, np.ndarray]:
    """acc has output channels on the last axis."""
    m = multipliers(sx, sw, sy)
    return {
        "fixed": clamp_u8(requant_fixed(acc, m["m0"], m["shift"]), zy),
        "fixed_single": clamp_u8(requant_fixed_single(acc, m["m0"], m["shift"]), zy),
        "float": clamp_u8(requant_float(acc, m["float"]), zy),
        "double": clamp_u8(requant_double(acc, m["double"]), zy),
    }
