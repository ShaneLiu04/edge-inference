from pathlib import Path

import numpy as np
import pytest
from convlab.winograd import (
    conv2d_direct,
    conv2d_winograd,
    correlate1d,
    direct_tile,
    tile2d,
    transform,
)
from edge_harness.cpp import build, compiler, run_json

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]


def test_f23_matches_lavin_matrices():
    t = transform(2)
    # Lavin & Gray 2016, up to the sign/order convention of the finite points (0, 1, -1).
    assert np.allclose(np.abs(t.BT), np.abs(np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, -1, 0, 1]])))
    assert np.allclose(np.abs(t.G), np.abs(np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]])))
    assert np.allclose(np.abs(t.AT), np.abs(np.array([[1, 1, 1, 0], [0, 1, -1, 1]])))


@pytest.mark.parametrize("m", [2, 4, 6])
def test_1d_correlation_exact_in_fp64(m):
    t = transform(m)
    rng = np.random.default_rng(m)
    d, g = rng.normal(size=t.n), rng.normal(size=3)
    want = np.array([d[i:i + 3] @ g for i in range(m)])
    assert np.allclose(correlate1d(t, d, g), want, atol=1e-12)


@pytest.mark.parametrize("m", [2, 4, 6])
def test_2d_tile_exact_in_fp64(m):
    t = transform(m)
    rng = np.random.default_rng(10 + m)
    d, g = rng.normal(size=(t.n, t.n)), rng.normal(size=(3, 3))
    assert np.allclose(tile2d(t, d, g), direct_tile(d, g, m), atol=1e-10)


def test_r5_kernel_also_works():
    t = transform(2, 5, [0, 1, -1, 2, -2, None])
    rng = np.random.default_rng(0)
    d, g = rng.normal(size=t.n), rng.normal(size=5)
    assert np.allclose(correlate1d(t, d, g), [d[i:i + 5] @ g for i in range(2)], atol=1e-12)


def test_bad_point_count_rejected():
    with pytest.raises(ValueError):
        transform(4, 3, [0, 1, -1, None])
    with pytest.raises(ValueError):
        transform(2, 3, [0, 1, 1, None])


def test_multichannel_conv_and_partial_tiles():
    rng = np.random.default_rng(5)
    x, w = rng.normal(size=(5, 11, 13)), rng.normal(size=(4, 5, 3, 3))
    ref = conv2d_direct(x, w)
    for m in (2, 4, 6):
        y = conv2d_winograd(x, w, transform(m), dtype=np.float64)
        assert y.shape == ref.shape
        assert np.allclose(y, ref, atol=1e-9)


def test_error_grows_with_tile_size_in_fp32():
    rng = np.random.default_rng(6)
    x, w = rng.normal(size=(32, 18, 18)), rng.normal(size=(16, 32, 3, 3))
    ref = conv2d_direct(x, w)
    errs = [np.abs(conv2d_winograd(x, w, transform(m), np.float32) - ref).max() / np.abs(ref).max() for m in (2, 4, 6)]
    assert errs[0] < errs[1] < errs[2]
    assert errs[2] > 10 * errs[0]


def test_arithmetic_reduction():
    assert transform(2).reduction_2d() == pytest.approx(2.25)
    assert transform(4).reduction_2d() == pytest.approx(4.0)
    assert transform(6).reduction_2d() == pytest.approx(81 / 64 * 4)


def test_int8_input_growth():
    # F(2,3): B^T rows have L1 norm 2, so a tile of int8 inputs needs 8 + 2 = 10 bits after B^T d B.
    assert transform(2).input_growth_2d() == pytest.approx(4)
    assert transform(4).input_growth_2d() > 50


def test_cpp_kernels_agree_with_fp64(tmp_path):
    if compiler() is None:
        pytest.skip("no C++ compiler")
    exe = build(LAB / "cpp/conv.cpp", tmp_path, includes=(ROOT / "edge_harness/include",))
    rows = run_json(exe, ["--mode=check"])
    assert rows
    assert max(r["rel_err"] for r in rows) < 1e-5
