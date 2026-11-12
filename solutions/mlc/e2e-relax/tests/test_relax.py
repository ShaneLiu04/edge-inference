import numpy as np
import pytest
from mlce2e.reference import forward_numpy
from test_e2e import _weights

pytest.importorskip("tvm")


def test_relax_variants_match_numpy():
    from mlce2e.relax_model import (
        build,
        create_model_via_emit_te,
        create_model_with_extern_conv,
        run,
        schedule_conv2d,
        schedule_linear,
    )
    wm = _weights()
    x = np.random.default_rng(3).random((4, 1, 28, 28), dtype=np.float32)
    ref = forward_numpy(x, wm)
    mod = create_model_via_emit_te(wm)
    conv = schedule_conv2d(mod).mod
    for m in (mod, create_model_with_extern_conv(wm), conv, schedule_linear(conv).mod):
        assert np.allclose(run(build(m), x), ref, rtol=1e-4, atol=1e-6)


def _prims(mod):
    import tvm
    return {gv.name_hint: f for gv, f in mod.functions.items() if isinstance(f, tvm.tirx.PrimFunc)}


def test_fuse_is_the_key_space_compile_relax_queries():
    """Tune fuse(mod), compile mod: fusing twice adds op_pattern to the fused funcs and the keys diverge."""
    import tvm
    from mlce2e.relax_model import create_model_via_emit_te, fuse
    mod = create_model_via_emit_te(_weights())
    once, twice = _prims(fuse(mod)), _prims(fuse(fuse(mod)))
    assert {"fused_conv2d_te_add_channel_bias_te_relu", "fused_te_linear_te_relu1"} <= set(once)
    again = _prims(fuse(mod))
    for k, f in once.items():
        tvm.ir.assert_structural_equal(f, again[k])
    fused = [k for k in once if k.startswith("fused_")]
    assert all("op_pattern" not in once[k].attrs and "op_pattern" in twice[k].attrs for k in fused)


def test_schedule_keeps_one_parallel_and_one_vectorized_loop():
    from mlce2e.relax_model import create_model_via_emit_te, schedule_conv2d
    text = schedule_conv2d(create_model_via_emit_te(_weights())).mod["conv2d"].script()
    assert "T.parallel" in text and "T.vectorized" in text and "T.unroll" in text and "conv2d_init" in text
