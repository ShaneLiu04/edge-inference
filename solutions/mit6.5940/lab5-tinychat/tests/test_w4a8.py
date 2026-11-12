import re
from pathlib import Path

import numpy as np
from w4a8_model import block_dot_avx2, maddubs_epi16, pack_qm_x86, sign_epi8, unpack_qm_x86, w4a8_matvec

LAB = Path(__file__).resolve().parents[1]
STARTER = {"all_techniques.cc", "loop_unrolling.cc", "multithreading.cc", "multithreading_loop_unrolling.cc",
           "simd_programming.cc"}


def test_pack_unpack_roundtrip_and_nibble_order():
    w = np.arange(64, dtype=np.int16) % 16 - 8
    p = pack_qm_x86(w.astype(np.int8))
    assert p.shape == (32,)
    assert p[0] == (w[0] + 8) | ((w[32] + 8) << 4)
    assert np.array_equal(unpack_qm_x86(p), w.astype(np.int8))


def _all_pairs(a_lo):
    w = np.repeat(np.arange(-8, 8, dtype=np.int8), 256 - (a_lo + 128))
    a = np.tile(np.arange(a_lo, 128, dtype=np.int16), 16).astype(np.int8)
    w2, a2 = np.repeat(w, 2), np.repeat(a, 2)
    prod = maddubs_epi16(sign_epi8(w2, w2), sign_epi8(a2, w2)).astype(np.int32)
    return prod, 2 * w.astype(np.int32) * a.astype(np.int32), w, a


def test_sign_trick_is_exact_for_every_4bit_weight_and_symmetric_int8_activation():
    prod, ref, _, _ = _all_pairs(-127)
    assert np.array_equal(prod, ref)


def test_sign_trick_breaks_at_minus_128_with_negative_weight():
    """-(-128) wraps to -128 in int8, so the product flips sign; the engine's quantizer uses amax / 127."""
    prod, ref, w, a = _all_pairs(-128)
    wrong = prod != ref
    assert wrong.any()
    assert np.all((a[wrong] == -128) & (w[wrong] < 0))


def test_block_dot_matches_integer_reference():
    r = np.random.default_rng(0)
    for _ in range(50):
        w = r.integers(-8, 8, 64).astype(np.int8)
        a = r.integers(-127, 128, 64).astype(np.int8)
        s0, s1 = block_dot_avx2(pack_qm_x86(w), a)
        assert s0 == int(w[:32].astype(np.int32) @ a[:32].astype(np.int32))
        assert s1 == int(w[32:].astype(np.int32) @ a[32:].astype(np.int32))


def test_matvec_matches_dequantized_float():
    r = np.random.default_rng(1)
    n, k = 8, 256
    w = r.integers(-8, 8, (n, k)).astype(np.int8)
    a = r.integers(-127, 128, k).astype(np.int8)
    ws = r.random(n * k // 32) + 0.5
    as_ = r.random(k // 32) + 0.5
    out = w4a8_matvec(pack_qm_x86(w.reshape(-1)), ws, a, as_, n, k)
    ref = (w.reshape(n, -1, 32) * ws.reshape(n, -1, 1)).reshape(n, k) @ (a.reshape(-1, 32) * as_[:, None]).reshape(k)
    assert np.allclose(out, ref)


def test_patch_touches_only_the_five_starter_files_and_fills_every_x86_todo():
    text = (LAB / "patch/x86.patch").read_text(encoding="utf-8")
    files = set(re.findall(r"^\+\+\+ b/kernels/starter_code/(\S+)", text, flags=re.M))
    assert files == STARTER
    assert "_mm256_maddubs_epi16(ax, sy)" in text and "pthread_join" in text
