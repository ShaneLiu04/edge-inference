"""Parts 2-4 of the assignment against the TVM 0.27 API.

Part 2: hand-written tensor expressions emitted with `BlockBuilder.emit_te`.
Part 3: the conv2d replaced by a registered NumPy function through `call_dps_packed`.
Part 4: a `s_tir.Schedule` on the conv2d PrimFunc (split, reorder, decompose, fuse, parallel/unroll/vectorize).

API names that moved since the mlc-ai-nightly notebook: StructInfo is `relax.TensorType`,
`tir.Schedule` is `s_tir.Schedule`, `register_func` is `register_global_func`, `tvm.nd.array`
is `tvm.runtime.tensor`, `relax.vm.build` is `tvm.compile`.
"""

from __future__ import annotations

import numpy as np
import tvm
from tvm import relax, te

from .reference import BATCH, INPUT_SHAPE, conv2d_nchw


def te_conv2d(x: te.Tensor, w: te.Tensor) -> te.Tensor:
    n, c, h, wd = x.shape
    f, _, kh, kw = w.shape
    rc = te.reduce_axis((0, c), "rc")
    ry = te.reduce_axis((0, kh), "ry")
    rx = te.reduce_axis((0, kw), "rx")
    return te.compute((n, f, h - kh + 1, wd - kw + 1),
                      lambda b, o, i, j: te.sum(x[b, rc, i + ry, j + rx] * w[o, rc, ry, rx], axis=[rc, ry, rx]),
                      name="conv2d")


def te_add_channel_bias(x: te.Tensor, b: te.Tensor) -> te.Tensor:
    return te.compute(x.shape, lambda n, c, i, j: x[n, c, i, j] + b[0, c, 0, 0], name="bias_add")


def te_relu(x: te.Tensor) -> te.Tensor:
    return te.compute(x.shape, lambda *i: te.max(x(*i), tvm.tirx.const(0, x.dtype)), name="relu")


def te_maxpool2x2(x: te.Tensor) -> te.Tensor:
    n, c, h, w = x.shape
    di = te.reduce_axis((0, 2), "di")
    dj = te.reduce_axis((0, 2), "dj")
    return te.compute((n, c, h // 2, w // 2),
                      lambda b, k, i, j: te.max(x[b, k, i * 2 + di, j * 2 + dj], axis=[di, dj]), name="maxpool")


def te_flatten(x: te.Tensor) -> te.Tensor:
    n, c, h, w = x.shape
    return te.compute((n, c * h * w), lambda b, i: x[b, i // (h * w), i % (h * w) // w, i % w], name="flatten")


def te_linear(x: te.Tensor, w: te.Tensor, b: te.Tensor) -> te.Tensor:
    n, k = x.shape
    m = w.shape[0]
    rk = te.reduce_axis((0, k), "rk")
    mm = te.compute((n, m), lambda i, j: te.sum(x[i, rk] * w[j, rk], axis=rk), name="matmul")
    return te.compute((n, m), lambda i, j: mm[i, j] + b[0, j], name="linear_bias")


def te_softmax(x: te.Tensor) -> te.Tensor:
    n, m = x.shape
    k1 = te.reduce_axis((0, m), "k1")
    k2 = te.reduce_axis((0, m), "k2")
    mx = te.compute((n,), lambda i: te.max(x[i, k1], axis=k1), name="sm_max")
    ex = te.compute((n, m), lambda i, j: te.exp(x[i, j] - mx[i]), name="sm_exp")
    s = te.compute((n,), lambda i: te.sum(ex[i, k2], axis=k2), name="sm_sum")
    return te.compute((n, m), lambda i, j: ex[i, j] / s[i], name="softmax")


def _consts(wm: dict[str, np.ndarray]) -> dict[str, relax.Constant]:
    return {
        "cw": relax.const(wm["conv2d_weight"], "float32"),
        "cb": relax.const(wm["conv2d_bias"].reshape(1, 32, 1, 1), "float32"),
        "w0": relax.const(wm["linear0_weight"], "float32"),
        "b0": relax.const(wm["linear0_bias"].reshape(1, 100), "float32"),
        "w1": relax.const(wm["linear1_weight"], "float32"),
        "b1": relax.const(wm["linear1_bias"].reshape(1, 10), "float32"),
    }


def _tail(bb: relax.BlockBuilder, conv, k):
    y = bb.emit_te(te_add_channel_bias, conv, k["cb"])
    y = bb.emit_te(te_relu, y)
    y = bb.emit_te(te_maxpool2x2, y)
    y = bb.emit_te(te_flatten, y)
    y = bb.emit_te(te_linear, y, k["w0"], k["b0"])
    y = bb.emit_te(te_relu, y)
    y = bb.emit_te(te_linear, y, k["w1"], k["b1"])
    y = bb.emit_te(te_softmax, y)
    return bb.emit_output(y)


def create_model_via_emit_te(wm: dict[str, np.ndarray]) -> tvm.IRModule:
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(INPUT_SHAPE, "float32"))
    k = _consts(wm)
    with bb.function("main", [x]):
        with bb.dataflow():
            conv = bb.emit_te(te_conv2d, x, k["cw"], primfunc_name_hint="conv2d")
            gv = _tail(bb, conv, k)
        bb.emit_func_output(gv)
    return bb.get()


@tvm.register_global_func("env.conv2d", override=True)
def _env_conv2d(x, w, out) -> None:
    out.copyfrom(conv2d_nchw(x.numpy(), w.numpy()))


def create_model_with_extern_conv(wm: dict[str, np.ndarray]) -> tvm.IRModule:
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType(INPUT_SHAPE, "float32"))
    k = _consts(wm)
    with bb.function("main", [x]):
        with bb.dataflow():
            conv = bb.emit(relax.call_dps_packed("env.conv2d", [x, k["cw"]],
                                                 relax.TensorType((BATCH, 32, 26, 26), "float32")))
            gv = _tail(bb, conv, k)
        bb.emit_func_output(gv)
    return bb.get()


def schedule_conv2d(mod: tvm.IRModule, func_name: str = "conv2d") -> tvm.s_tir.Schedule:
    """One output row of 4 channels per task: (n, f/4, y) parallel, rc ry rx, then f%4 unrolled, x vectorized.

    The whole conv is 0.78 M MACs, so a task must stay small enough to give 8 threads ~100 tasks each
    and still keep a full 26-wide row in the vector loop; coarser tiles lose to the naive loop.
    """
    sch = tvm.s_tir.Schedule(mod)
    block = sch.get_sblock("conv2d", func_name=func_name) if hasattr(sch, "get_sblock") else \
        sch.get_block("conv2d", func_name=func_name)
    n, f, y, x, rc, ry, rx = sch.get_loops(block)
    f0, f1 = sch.split(f, [None, 4])
    sch.reorder(n, f0, y, rc, ry, rx, f1, x)
    sch.parallel(sch.fuse(n, f0, y))
    sch.unroll(f1)
    sch.vectorize(x)
    # After decompose_reduction the update block has no init and fails the compact-dataflow
    # check that vectorize/unroll require, so annotate first.
    sch.decompose_reduction(block, rc)
    return sch


def schedule_linear(mod: tvm.IRModule, func_name: str = "te_linear", lanes: int = 16) -> tvm.s_tir.Schedule:
    """Split k by `lanes` and rfactor it: (j) parallel, k/lanes serial, the lanes vectorized.

    A float reduction over k cannot be vectorized in place (it would reorder the sum); rfactor gives
    each lane its own partial sum, and the final 16-way sum stays scalar.
    """
    sch = tvm.s_tir.Schedule(mod)
    mm = sch.get_sblock("matmul", func_name=func_name)
    _, _, k = sch.get_loops(mm)
    _, k1 = sch.split(k, [None, lanes])
    rf = sch.rfactor(k1, factor_axis=2)
    i, j, k0, k1 = sch.get_loops(rf)
    sch.reorder(j, i, k0, k1)
    sch.parallel(j)
    sch.vectorize(k1)
    sch.decompose_reduction(rf, k0)
    return sch


def fuse(mod: tvm.IRModule) -> tvm.IRModule:
    """The fusion prefix of MetaSchedule's compile_relax; tune its output so the database keys match."""
    return tvm.transform.Sequential([relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(),
                                     relax.transform.FoldConstant(), relax.transform.FuseOps(),
                                     relax.transform.FuseTIR()])(mod)


def build(mod: tvm.IRModule, target="llvm") -> relax.VirtualMachine:
    return relax.VirtualMachine(tvm.compile(mod, target=target), tvm.cpu())


def run(vm: relax.VirtualMachine, x: np.ndarray) -> np.ndarray:
    return vm["main"](tvm.runtime.tensor(x, tvm.cpu())).numpy()
