from pathlib import Path

import numpy as np
import pytest
from blockq.pack import (
    QUANTIZERS,
    bits_per_weight,
    dequantize,
    quantize_q4_0,
    quantize_q8_0,
    roundtrip,
    sqnr_db,
)
from edge_harness.cpp import build, compiler, run_json

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]


def weights(shape=(16, 256), seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.02, shape).astype(np.float32)
    w.reshape(-1)[::997] *= 10  # outliers
    return w


def test_sizes():
    w = weights()
    assert quantize_q8_0(w).shape == (16, 256 // 32 * 34)
    assert quantize_q4_0(w).shape == (16, 256 // 32 * 18)
    assert bits_per_weight("q4_0") == 4.5 and bits_per_weight("q8_0") == 8.5 and bits_per_weight("q4_1") == 5.0


def test_q4_0_largest_element_is_exact():
    w = np.zeros(32, np.float32)
    w[5] = -0.8
    w[6] = 0.3
    back = roundtrip(w, "q4_0")
    assert back[5] == pytest.approx(-0.8, rel=1e-3)   # maps to q = 0, value -8 d
    assert abs(back[6] - 0.3) <= 0.8 / 8 / 2 + 1e-3


def test_zero_block():
    w = np.zeros((2, 64), np.float32)
    for fmt in QUANTIZERS:
        assert np.array_equal(roundtrip(w, fmt), w)


def test_error_ordering():
    w = weights((64, 1024))
    s = {fmt: sqnr_db(w, roundtrip(w, fmt)) for fmt in QUANTIZERS}
    assert s["q8_0"] > s["q4_1"] > s["q4_0"]
    assert 35 < s["q8_0"] < 45   # ~6 dB per bit
    assert 15 < s["q4_0"] < 25


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "q4_1"])
def test_byte_exact_with_gguf_reference(fmt):
    gguf = pytest.importorskip("gguf")
    from gguf import quants

    w = weights((8, 512), seed=3)
    qtype = {"q8_0": gguf.GGMLQuantizationType.Q8_0, "q4_0": gguf.GGMLQuantizationType.Q4_0,
             "q4_1": gguf.GGMLQuantizationType.Q4_1}[fmt]
    ref = quants.quantize(w, qtype)
    ours = QUANTIZERS[fmt](w)
    assert ours.shape == ref.shape
    assert np.array_equal(ours, ref)
    assert np.allclose(dequantize(ours, fmt), quants.dequantize(ref, qtype))


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    if compiler() is None:
        pytest.skip("no C++ compiler")
    return build(LAB / "cpp/gemv.cpp", tmp_path_factory.mktemp("bin"), includes=(ROOT / "edge_harness/include",))


def test_cpp_packer_byte_exact(exe, tmp_path):
    w = weights((24, 384), seed=7)
    (tmp_path / "shape.txt").write_text("24 384\n")
    w.tofile(tmp_path / "w.f32")
    run_json(exe, ["--mode=pack", f"--dir={tmp_path}"])
    assert np.array_equal(np.fromfile(tmp_path / "q4_0.bin", np.uint8), quantize_q4_0(w).reshape(-1))
    assert np.array_equal(np.fromfile(tmp_path / "q8_0.bin", np.uint8), quantize_q8_0(w).reshape(-1))


def test_cpp_kernels(exe):
    rows = run_json(exe, ["--mode=check"])
    by = {(r["shape"], r["method"]): r for r in rows}
    for (shape, method), r in by.items():
        if r["rel_err_vs_own_weights"] >= 0:
            assert r["rel_err_vs_own_weights"] < 1e-5, (shape, method)
    for shape in {s for s, _ in by}:
        # Quantizing the activation to Q8_0 adds only a little on top of the weight error.
        w_only = by[(shape, "q4_0_fused_f32")]["rel_err_vs_fp32"]
        if (shape, "q4_0_x_q8_vnni") in by:
            assert by[(shape, "q4_0_x_q8_vnni")]["rel_err_vs_fp32"] < 1.1 * w_only
            # More accumulators only reorder the float sums of the same integer block dots.
            for fmt in ("q4_0", "q8_0"):
                one, four = by[(shape, f"{fmt}_x_q8_vnni")], by[(shape, f"{fmt}_x_q8_vnni_4acc")]
                assert abs(one["rel_err_vs_fp32"] - four["rel_err_vs_fp32"]) < 1e-6, (shape, fmt)


def test_gpt2_forward_is_causal_and_finds_its_linears():
    torch = pytest.importorskip("torch")
    from blockq.gpt2 import GPT2

    torch.manual_seed(0)
    model = GPT2(vocab=50, ctx=16, d=32, layers=2, heads=4, eps=1e-5).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    ids = torch.randint(0, 50, (1, 12))
    changed = ids.clone()
    changed[0, 8:] = (changed[0, 8:] + 1) % 50
    with torch.no_grad():
        a, b = model(ids), model(changed)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5)
    assert not torch.allclose(a[:, 8:], b[:, 8:])
    names = [n for n, m in model.named_modules() if type(m).__name__ == "Conv1D" and ".h." in f".{n}"]
    assert len(names) == 2 * 4
