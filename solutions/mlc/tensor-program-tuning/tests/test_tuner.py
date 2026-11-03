import random
from pathlib import Path

import numpy as np
import pytest
from edge_harness.cpp import compiler
from tuner.runner import measure, source
from tuner.schedule import ORDERS, Problem, Schedule, Space, render
from tuner.search import STRATEGIES, Ridge, best_so_far, features

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]
SMALL = Problem("t", 32, 64, 48)


def test_space_respects_divisibility():
    sp = Space(SMALL)
    assert all(SMALL.M % t == 0 for t in sp.ti)
    assert all(SMALL.N % t == 0 for t in sp.tj)
    assert all(SMALL.K % t == 0 for t in sp.tk)
    assert sp.size() == len(sp.all())
    rng = random.Random(0)
    for _ in range(50):
        s = sp.sample(rng)
        assert s.ti in sp.ti and s.tj in sp.tj and s.tk in sp.tk
        for n in sp.neighbors(s, rng, 3):
            assert n.ti in sp.ti and n.outer in ORDERS


def test_render_emits_all_loops_and_pragma():
    code = render(SMALL, Schedule("kji", "jki", 4, 16, 8, True, 4), "f")
    for v in ("i0", "j0", "k0", "ii", "jj", "kk", "#pragma GCC unroll 4", "bt["):
        assert v in code
    assert code.index("k0 = 0") < code.index("j0 = 0") < code.index("i0 = 0")


def test_schedule_key_round_trips():
    for s in Space(SMALL).all()[::97]:
        assert Schedule.from_key(s.key()) == s


def test_harness_source_lists_every_kernel():
    batch = [Schedule(tk=16), Schedule("jik", "ijk", 8, 16, 16, True, 1)]
    src = source(SMALL, batch)
    assert src.count("static void k") == 2
    assert batch[1].key() in src


def test_ridge_recovers_linear_function():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    w = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    m = Ridge(lam=1e-6)
    m.fit(X, X @ w + 4)
    assert np.allclose(m.predict(X[:3]), X[:3] @ w + 4, atol=1e-4)


def test_features_shape_is_constant():
    assert len({len(features(s)) for s in Space(SMALL).all()[:200]}) == 1


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategies_respect_budget_with_fake_measure(name):
    sp = Space(Problem("p", 64, 128, 64))

    def fake(batch):  # prefers j innermost and big tj: a model should find that quickly
        return {s.key(): (10 if s.inner[-1] == "j" else 1) * s.tj / 16 + s.pack_b for s in batch}

    hist = STRATEGIES[name](sp, fake, budget=32, seed=1)
    assert 0 < len(hist) <= 32
    assert len({h["key"] for h in hist}) == len(hist)
    curve = best_so_far(hist)
    assert curve == sorted(curve)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategies_terminate_when_space_is_smaller_than_the_pool(name):
    sp = Space(Problem("mbv2_pw_expand_14", 196, 576, 96))
    assert sp.size() < 4000

    def fake(batch):
        return {s.key(): float(s.tj) for s in batch}

    hist = STRATEGIES[name](sp, fake, budget=64, seed=0)
    assert len({h["key"] for h in hist}) == len(hist) <= 64


def test_compiled_kernels_are_correct(tmp_path):
    if compiler() is None:
        pytest.skip("no C++ compiler")
    sp = Space(SMALL)
    rng = random.Random(3)
    batch = [sp.sample(rng) for _ in range(6)] + [Schedule("kij", "kji", 32, 64, 48, True, 4)]
    res = measure(SMALL, batch, tmp_path, ROOT / "edge_harness/include", batch=4, jobs=2, repeats=1)
    assert set(res) == {s.key() for s in batch}
    assert max(r["max_rel_err"] for r in res.values()) < 1e-5
    wide = measure(SMALL, batch[-2:], tmp_path, ROOT / "edge_harness/include", batch=2, jobs=1, repeats=1,
                   flags=("-mprefer-vector-width=512",))
    assert max(r["max_rel_err"] for r in wide.values()) < 1e-5
    assert len(list(tmp_path.glob("tune_t_*.cpp"))) == 3  # two batches of 4, plus a separate 512-bit binary


def test_render_rejects_tiles_that_do_not_divide():
    p = Problem("mbv2_pw_expand_14", 196, 576, 96)
    with pytest.raises(ValueError, match="does not divide 196"):
        render(p, Schedule("ijk", "ikj", 32, 64, 64), "k0")
    assert all(196 % t == 0 for t in Space(p).ti)
