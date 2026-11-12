"""MLC assignment 1 end to end, plus a MetaSchedule run on the same module. Linux + TVM 0.27.

  MLC_WEIGHTS=.../fasionmnist_mlp_assignment_params.pkl MLC_FMNIST_RAW=.../FashionMNIST/raw \
  python scripts/run_experiments.py

Writes artifacts/raw/summary.json, runs.jsonl, the TVMScript of each module, and two figures.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[2]
sys.path[:0] = [str(LAB / "code"), str(REPO)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tvm  # noqa: E402
from edge_harness.timing import time_calls  # noqa: E402
from edge_harness.writer import JsonlWriter, write_json  # noqa: E402
from mlce2e.reference import BATCH, forward_numpy, load_fashion_mnist_test, load_weights, pytorch_model  # noqa: E402
from mlce2e.relax_model import (  # noqa: E402
    build,
    create_model_via_emit_te,
    create_model_with_extern_conv,
    fuse,
    run,
    schedule_conv2d,
    schedule_linear,
)

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"
WORK = LAB / "artifacts/work"
WEIGHTS = Path(os.environ.get("MLC_WEIGHTS", WORK / "fasionmnist_mlp_assignment_params.pkl"))
FMNIST = Path(os.environ.get("MLC_FMNIST_RAW", WORK / "FashionMNIST/raw"))
TARGET = json.loads(os.environ.get("MLC_TARGET", '{"kind": "llvm", "mcpu": "sapphirerapids", "num-cores": 8}'))
TRIALS = int(os.environ.get("MLC_MS_TRIALS", "640"))


def sweep(fn, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Run fn over the whole test set in batches of 4; return outputs and seconds."""
    outs = []
    t0 = time.perf_counter()
    for i in range(0, len(x), BATCH):
        outs.append(fn(x[i:i + BATCH]))
    return np.concatenate(outs), time.perf_counter() - t0


def per_batch_us(fn, x4: np.ndarray, repeats: int = 300) -> float:
    return time_calls(lambda: fn(x4), warmup=30, repeats=repeats).p50_s * 1e6


def metaschedule(mod: tvm.IRModule, target: tvm.target.Target, work_dir: Path):
    """Tune the fused module, compile the unfused one.

    compile_relax fuses before it queries the database and tune_relax does not, so tuning `mod`
    as is leaves every fused kernel without a record. Passing the fused module to compile_relax
    fails too: its second AnnotateTIROpPattern adds an op_pattern attr to the fused PrimFuncs and
    their structural keys no longer match. Both cases fall back to the default schedule silently.
    """
    from tvm.s_tir import meta_schedule as ms
    fused = fuse(mod)
    (WORK / "mod_fused.py").write_text(fused.script(), encoding="utf-8")
    reuse = os.environ.get("MLC_MS_REUSE") == "1" and (work_dir / "database_tuning_record.json").exists()
    t0 = time.perf_counter()
    if reuse:
        db = ms.database.JSONDatabase(work_dir=str(work_dir))
    else:
        # LocalBuilder defaults to os.cpu_count() workers (208 here) and a 30 s timeout; inside an
        # 8-core taskset that many fresh TVM processes time out and every candidate is a build error.
        workers = len(os.sched_getaffinity(0))
        db = ms.relax_integration.tune_relax(mod=fused, params={}, target=target, work_dir=str(work_dir),
                                            max_trials_global=TRIALS, num_trials_per_iter=32,
                                            builder=ms.builder.LocalBuilder(max_workers=workers, timeout_sec=120),
                                            runner=ms.runner.LocalRunner(timeout_sec=60))
    tune_s = None if reuse else time.perf_counter() - t0
    ex = ms.relax_integration.compile_relax(db, mod=mod, target=target, params={})
    return tvm.relax.VirtualMachine(ex, tvm.cpu()), tune_s, len(db.get_all_tuning_records())


def conv_kernel_us(prim, target) -> float:
    """Time the conv2d PrimFunc alone on random inputs, outside the VM."""
    dev = tvm.cpu()
    lib = tvm.compile(tvm.IRModule({"main": prim.with_attr("global_symbol", "main")}), target=target)
    args = [tvm.runtime.tensor(np.random.rand(*s).astype("float32"), dev)
            for s in ((BATCH, 1, 28, 28), (32, 1, 3, 3), (BATCH, 32, 26, 26))]
    te = lib.mod.time_evaluator("main", dev, number=100, repeat=5)
    return float(np.median(te(*args).results)) * 1e6


STAGES = ["conv2d", "bias", "relu", "maxpool", "flatten", "linear0", "relu", "linear1", "softmax"]


def prefix_model(wm, n: int) -> tvm.IRModule:
    """The first n stages of the network as a Relax function, for a per-stage time breakdown."""
    from mlce2e import relax_model as rm
    from mlce2e.reference import INPUT_SHAPE
    from tvm import relax
    bb = relax.BlockBuilder()
    xv = relax.Var("x", relax.TensorType(INPUT_SHAPE, "float32"))
    k = rm._consts(wm)
    steps = [lambda y: bb.emit_te(rm.te_conv2d, y, k["cw"], primfunc_name_hint="conv2d"),
             lambda y: bb.emit_te(rm.te_add_channel_bias, y, k["cb"]),
             lambda y: bb.emit_te(rm.te_relu, y),
             lambda y: bb.emit_te(rm.te_maxpool2x2, y),
             lambda y: bb.emit_te(rm.te_flatten, y),
             lambda y: bb.emit_te(rm.te_linear, y, k["w0"], k["b0"]),
             lambda y: bb.emit_te(rm.te_relu, y),
             lambda y: bb.emit_te(rm.te_linear, y, k["w1"], k["b1"]),
             lambda y: bb.emit_te(rm.te_softmax, y)]
    with bb.function("main", [xv]):
        with bb.dataflow():
            y = xv
            for s in steps[:n]:
                y = s(y)
            gv = bb.emit_output(y)
        bb.emit_func_output(gv)
    return bb.get()


def vm_us(vm, x4: np.ndarray) -> float | None:
    """VM-side timer: excludes the Python call and the host copies."""
    dev = tvm.cpu()
    try:
        te = vm.time_evaluator("main", dev, number=50, repeat=5)
        return float(np.median(te(tvm.runtime.tensor(x4, dev)).results)) * 1e6
    except Exception:  # the extern variant calls back into Python, which time_evaluator rejects
        return None


def plot(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("torch_us", "PyTorch eager"), ("relax_us", "Relax, default\nschedule"),
            ("extern_us", "Relax + NumPy\nconv (extern)"),
            ("scheduled_us", "Relax, hand\nconv schedule"), ("scheduled_conv_linear_us", "Relax, hand\nconv + linear0"),
            ("ms_us", "Relax,\nMetaSchedule")]
    keys = [(k, label) for k, label in keys if k in summary]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    vals = [summary[k] for k, _ in keys]
    ax.bar(range(len(keys)), vals, color="#2c5f8a")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(keys)), [label for _, label in keys], fontsize=8)
    ax.set_ylabel("us per batch of 4 (p50)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "e2e_latency.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    wm = load_weights(WEIGHTS)
    x, y = load_fashion_mnist_test(FMNIST)
    x4 = x[:BATCH]
    target = tvm.target.Target(TARGET)
    summary = {"tvm": tvm.__version__, "torch": torch.__version__, "target": TARGET, "n_test": len(x),
               "tvm_num_threads": os.environ.get("TVM_NUM_THREADS", "default"),
               "tvm_bind_threads": os.environ.get("TVM_BIND_THREADS", "default")}

    # Part 1: the PyTorch model and the NumPy reference.
    tm = pytorch_model(wm)
    with torch.inference_mode():
        torch_fn = lambda b: tm(torch.from_numpy(b)).numpy()  # noqa: E731
        p_torch, _ = sweep(torch_fn, x)
        summary["torch_us"] = per_batch_us(torch_fn, x4)
    summary["torch_acc"] = float((p_torch.argmax(1) == y).mean())
    p_np, _ = sweep(lambda b: forward_numpy(b, wm), x)
    summary["numpy_vs_torch_max_abs"] = float(np.abs(p_np - p_torch).max())

    rows = []
    variants = {}
    mod = create_model_via_emit_te(wm)
    (WORK / "mod_emit_te.py").write_text(mod.script(), encoding="utf-8")
    variants["relax"] = build(mod, target)
    variants["extern"] = build(create_model_with_extern_conv(wm), target)
    sch = schedule_conv2d(mod)
    (WORK / "conv2d_scheduled.py").write_text(sch.mod["conv2d"].script(), encoding="utf-8")
    variants["scheduled"] = build(sch.mod, target)
    sch2 = schedule_linear(sch.mod)
    (WORK / "linear0_scheduled.py").write_text(sch2.mod["te_linear"].script(), encoding="utf-8")
    variants["scheduled_conv_linear"] = build(sch2.mod, target)
    summary["conv_kernel_us"] = {"naive": conv_kernel_us(mod["conv2d"], target),
                                 "hand_schedule": conv_kernel_us(sch.mod["conv2d"], target)}
    stages = []
    for n in range(1, len(STAGES) + 1):
        stages.append({"through": STAGES[n - 1], "vm_us": vm_us(build(prefix_model(wm, n), target), x4)})
    summary["default_build_prefix_us"] = stages
    print(summary["conv_kernel_us"], stages, flush=True)
    if TRIALS > 0:
        vm, tune_s, n_records = metaschedule(mod, target, WORK / "ms")
        variants["ms"] = vm
        old = json.loads((RAW / "summary.json").read_text()) if (RAW / "summary.json").exists() else {}
        summary["ms_tune_s"] = tune_s if tune_s is not None else old.get("ms_tune_s")
        summary["ms_trials"] = TRIALS
        summary["ms_records"] = n_records

    with JsonlWriter(RAW / "runs.jsonl") as w:
        for name, vm in variants.items():
            fn = lambda b, vm=vm: run(vm, b)  # noqa: E731
            p, secs = sweep(fn, x)
            row = {"variant": name, "acc": float((p.argmax(1) == y).mean()),
                   "max_abs_vs_torch": float(np.abs(p - p_torch).max()),
                   "allclose_rtol_1e-4": bool(np.allclose(p, p_torch, rtol=1e-4, atol=1e-6)),
                   "argmax_agree": float((p.argmax(1) == p_torch.argmax(1)).mean()),
                   "us_per_batch": per_batch_us(fn, x4), "vm_time_evaluator_us": vm_us(vm, x4),
                   "sweep_s": secs}
            rows.append(row)
            w.write(row)
            summary[f"{name}_acc"] = row["acc"]
            summary[f"{name}_max_abs"] = row["max_abs_vs_torch"]
            summary[f"{name}_allclose"] = row["allclose_rtol_1e-4"]
            summary[f"{name}_us"] = row["us_per_batch"]
            print(row, flush=True)
    summary["rows"] = rows
    write_json(RAW / "summary.json", summary)
    plot(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
