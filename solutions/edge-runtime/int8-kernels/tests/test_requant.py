import math

import numpy as np
import pytest
from int8ref.requant import (
    clamp_u8,
    quantize_multiplier,
    requant_double,
    requant_fixed,
    requant_fixed_single,
    requant_float,
    rounding_divide_by_pot,
    srdhm,
)


@pytest.mark.parametrize("m", [0.5, 0.75, 1e-3, 3.3e-5, 0.999999, 1.5, 3.0])
def test_multiplier_represents_m(m):
    m0, shift = quantize_multiplier(m)
    assert (1 << 30) <= m0 < (1 << 31)
    assert m0 * 2.0 ** (-31 - shift) == pytest.approx(m, rel=2 ** -30)


def test_multiplier_rounding_up_to_two_pow_31_renormalizes():
    m = 1 - 2 ** -40  # q * 2^31 rounds to exactly 2^31
    m0, shift = quantize_multiplier(m)
    assert m0 == 1 << 30
    assert shift == -1


def test_srdhm_rounds_ties_up():
    half = 1 << 30  # a * b / 2^31 == +-0.5 or +-1.5 exactly
    assert srdhm(np.array([1, 3]), half).tolist() == [1, 2]
    # The negative nudge is 1 - 2^30 followed by truncation, so -0.5 -> 0 and -1.5 -> -1.
    assert srdhm(np.array([-1, -3]), half).tolist() == [0, -1]
    assert srdhm(np.array([-(1 << 31)]), -(1 << 31))[0] == (1 << 31) - 1


@pytest.mark.parametrize("x,e,want", [(5, 1, 3), (-5, 1, -3), (6, 2, 2), (-6, 2, -2), (7, 0, 7), (-1, 3, 0)])
def test_rounding_divide_by_pot(x, e, want):
    assert rounding_divide_by_pot(np.array([x]), e)[0] == want


def test_float_ties_to_even_double_ties_away():
    acc = np.array([1, 3, -1, -3])
    m = np.full(4, 0.5)
    assert requant_float(acc, m).tolist() == [0, 2, 0, -2]
    assert requant_double(acc, m).tolist() == [1, 2, -1, -2]


def test_fixed_double_rounding_shows_up_without_ties():
    rng = np.random.default_rng(1)
    acc = rng.integers(-(1 << 20), 1 << 20, 200_000)
    m = 0.0037
    m0, shift = quantize_multiplier(m)
    exact = requant_double(acc, m)
    two = requant_fixed(acc, m0, shift)
    one = requant_fixed_single(acc, m0, shift)
    assert np.abs(two - exact).max() == 1
    assert (two != exact).mean() > 1e-4
    # Single rounding only differs on (near-)ties: never more than one step, almost never at all.
    assert (one != exact).mean() < 1e-4


def test_shift_resolution_predicts_double_rounding_rate():
    """The first rounding is at 2^-shift of the output unit; P(mismatch) is about 2^-(shift+1)."""
    rng = np.random.default_rng(2)
    acc = rng.integers(-(1 << 22), 1 << 22, 400_000)
    # Irrational multipliers: a rational one such as 0.02 = 16/25 * 2^-5 can make the bad band empty.
    for m in (math.pi * 1e-2, math.e * 1e-4):
        m0, shift = quantize_multiplier(m)
        rate = (requant_fixed(acc, m0, shift) != requant_double(acc, m)).mean()
        assert rate == pytest.approx(2.0 ** -(shift + 1), rel=0.35)


def test_clamp():
    assert clamp_u8(np.array([-300, -5, 0, 100, 300]), 5).tolist() == [0, 0, 5, 105, 255]


def test_multiplier_fixed_matches_frexp():
    m0, shift = quantize_multiplier(0.0007)
    q, e = math.frexp(0.0007)
    assert shift == -e
    assert m0 == round(q * 2 ** 31)
