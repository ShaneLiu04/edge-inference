from pathlib import Path

import numpy as np
import pytest
from edge_harness.cpp import build, compiler, run_json
from int8ref.layer import acc_int32, random_layer, read_outputs, write_layer
from int8ref.requant import requant_all

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]

CASES = [
    ("conv3x3", 12, 8, 16, 3, 1, 1),
    ("pointwise", 10, 16, 24, 1, 1, 1),
    ("depthwise", 12, 16, 16, 3, 1, 16),
    ("depthwise_s2", 13, 16, 16, 3, 2, 16),
    ("grouped", 10, 8, 8, 3, 1, 2),
]


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    if compiler() is None:
        pytest.skip("no C++ compiler")
    return build(LAB / "cpp/int8.cpp", tmp_path_factory.mktemp("bin"), includes=(ROOT / "edge_harness/include",))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_cpp_conv_bit_exact_with_numpy(exe, tmp_path, case):
    name, h, ic, oc, k, s, g = case
    layer = random_layer(np.random.default_rng(3), h, ic, oc, k, s, g, name)
    write_layer(layer, tmp_path)
    run_json(exe, ["--mode=conv", f"--dir={tmp_path}"])
    out = read_outputs(layer, tmp_path)
    ref = acc_int32(layer)
    assert np.array_equal(out["acc"], ref)
    py = requant_all(ref, layer.sx, layer.sw, layer.sy, layer.zy)
    for style in ("fixed", "float", "double"):
        assert np.array_equal(out[style], py[style]), style


def test_cpp_kernels_bit_exact(exe):
    rows = run_json(exe, ["--mode=bench", "--repeats=1"])
    assert rows and all(r["exact"] for r in rows)
    methods = {r["method"] for r in rows}
    assert "int8_scalar_autovec" in methods


def test_maddubs_saturates_only_above_64(exe):
    rows = {r["wmax"]: r for r in run_json(exe, ["--mode=saturate"])}
    assert rows[63]["wrong"] == 0 and rows[64]["wrong"] == 0
    assert rows[127]["wrong_frac"] > 0.5


def tie_layer(case, seed=4):
    """Per-tensor power-of-two multiplier 2^-6: about 1/64 of the accumulators land on an exact tie."""
    name, h, ic, oc, k, s, g = case
    layer = random_layer(np.random.default_rng(seed), h, ic, oc, k, s, g, name)
    layer.sw = np.full(oc, float(layer.sw.max()))
    layer.sy = float(layer.sx * layer.sw[0] * 64)
    layer.zy = 128
    return layer


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_ort_qlinearconv_is_float_round_half_even(case):
    pytest.importorskip("onnxruntime")
    from int8ref.ort import run_qlinearconv

    layer = tie_layer(case)
    py = requant_all(acc_int32(layer), layer.sx, layer.sw, layer.sy, layer.zy)
    assert np.array_equal(run_qlinearconv(layer), py["float"])


def test_ties_separate_half_even_from_half_away():
    pytest.importorskip("onnxruntime")
    from int8ref.ort import run_qlinearconv

    layer = tie_layer(("big", 28, 32, 64, 3, 1, 1))
    py = requant_all(acc_int32(layer), layer.sx, layer.sw, layer.sy, layer.zy)
    split = py["float"] != py["double"]
    assert split.sum() > 20
    y = run_qlinearconv(layer)
    assert np.array_equal(y[split], py["float"][split])
