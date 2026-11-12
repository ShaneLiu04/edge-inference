"""Run the folded micronet on torch, ORT fp32, ORT int8, LiteRT, ExecuTorch and TVM.

One thread each, one process per runtime. Wall time is recorded but not used:
this container's cgroup quota is below one core, so wall time measures the
quota. ``cpu_us`` (process time) is the number the summary compares.

Each runtime has its own interpreter because their torch pins conflict
(ExecuTorch 1.6 wants torch 2.15, litert-torch 0.10 wants torch < 2.15) and
because the 2 GB cgroup cannot hold two of them at once. Set the interpreter
per runtime with ``RTCMP_PY_<NAME>`` (e.g. ``RTCMP_PY_EXECUTORCH=/venv/bin/python``).
``RTCMP_EVICT`` (directories joined by os.pathsep) drops their page cache before each
process; see evict_page_cache.py. ``scripts/run_autodl.sh`` sets all of this up.

Usage (from the repo root, on the Linux box):
  python solutions/edge-runtime/runtime-compare/scripts/run_experiments.py
  python solutions/edge-runtime/runtime-compare/scripts/run_experiments.py --only tvm
  python solutions/edge-runtime/runtime-compare/scripts/run_experiments.py --merge
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[2]
sys.path[:0] = [
    str(LAB / "code"),
    str(REPO / "solutions/mit6.5940/lab2-quantization/code"),
    str(REPO),
]

from rtcmp.measure import agreement, combine, cpu_quota_cores, time_call  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"
WORK = LAB / "artifacts/work"  # model files and reference logits: gitignored
CKPT = REPO / "solutions/mit6.5940/lab2-quantization/checkpoints/micronet.pt"
REPEATS = 200
WARMUP = 20
TRIALS = 3
CALIB_N = 512
TVM_TARGET = json.loads(os.environ.get("RTCMP_TVM_TARGET", '{"kind": "llvm", "mcpu": "sapphirerapids", "num-cores": 1}'))
# TVM's relax.base_py_module JIT-compiles a torch C++ addon whenever torch is importable,
# and that g++ build alone pushes a 2 GB cgroup over its limit. The TVM row needs no torch.
TORCH_FREE = {"tvm"}


def load_folded():
    import torch
    from edgequant.model import MicroMobileNetV2
    from edgequant.qsim import fold_model

    torch.set_num_threads(1)
    model = MicroMobileNetV2()
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    folded = fold_model(model).eval()
    for p in folded.parameters():
        p.requires_grad_(False)
    return folded


def read_quota() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.max")
    if not path.exists():
        return None
    return cpu_quota_cores(path.read_text().strip())


def accuracy(predict, x, y) -> tuple[float, np.ndarray, float]:
    """Batch-1 pass over the test set. Also returns process time per image over the whole pass."""
    logits = []
    cpu0 = time.process_time()
    for i in range(len(x)):
        logits.append(np.asarray(predict(x[i:i + 1]), dtype=np.float64))
    cpu_us = (time.process_time() - cpu0) / len(x) * 1e6
    out = np.concatenate(logits, 0)
    return float(np.mean(out.argmax(1) == y)), out, cpu_us


def time_one(predict, sample: np.ndarray) -> dict:
    lone = sample[:1].copy()

    def once():
        predict(lone)

    return time_call(once, REPEATS, WARMUP)


def export_onnx(model, sample) -> Path:
    import torch

    path = WORK / "micronet_folded_b1.onnx"
    torch.onnx.export(
        model, torch.from_numpy(sample[:1]), str(path),
        input_names=["x"], output_names=["logits"], opset_version=17, dynamo=False,
    )
    return path


def ort_session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def predict(x):
        return sess.run(None, {"x": np.ascontiguousarray(x)})[0]

    return predict, ort.__version__


def backend_torch(model, sample, ctx):
    import torch

    def predict(x):
        with torch.inference_mode():
            return model(torch.from_numpy(np.ascontiguousarray(x))).numpy()

    return predict, {"version": torch.__version__, "threads": 1}


def backend_ort(model, sample, ctx):
    path = export_onnx(model, sample)
    predict, version = ort_session(path)
    return predict, {"version": version, "file_bytes": path.stat().st_size, "threads": 1}


def backend_ort_int8(model, sample, ctx):
    """lab2's deployed recipe: QDQ, per-channel int8 weights, uint8 activations, MinMax on 512 train images.

    Same as lab2's step_ort: a dynamic-batch export calibrated in batches of 64. The calibrator builds
    its own sessions with one spinning thread per physical core (104 here); under a 0.5-core quota
    that turns a seconds-long calibration into many minutes, so those sessions are pinned to 1 thread.
    """
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    dyn = WORK / "micronet_folded_dyn.onnx"
    torch.onnx.export(model, torch.from_numpy(sample[:1]), str(dyn), input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    pre = WORK / "micronet_folded_dyn_pre.onnx"
    quant_pre_process(str(dyn), str(pre), skip_symbolic_shape=True)
    calib = ctx["calib"]

    class Reader(CalibrationDataReader):
        def __init__(self, batch: int = 64):
            self.batches = [calib[i:i + batch] for i in range(0, len(calib), batch)]
            self.i = 0

        def get_next(self):
            if self.i >= len(self.batches):
                return None
            self.i += 1
            return {"x": self.batches[self.i - 1]}

        def rewind(self):
            self.i = 0

    out = WORK / "micronet_int8_minmax.onnx"
    base = ort.InferenceSession

    class OneThread(base):
        def __init__(self, path, sess_options=None, *args, **kwargs):
            so = sess_options or ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            super().__init__(path, so, *args, **kwargs)

    ort.InferenceSession = OneThread
    try:
        quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax)
    finally:
        ort.InferenceSession = base
    pre.unlink(missing_ok=True)
    dyn.unlink(missing_ok=True)
    predict, version = ort_session(out)
    return predict, {"version": version, "file_bytes": out.stat().st_size, "threads": 1, "calib_n": len(calib)}


def backend_litert(model, sample, ctx):
    import ai_edge_litert
    import litert_torch
    import torch
    from ai_edge_litert.interpreter import Interpreter

    edge = litert_torch.convert(model, (torch.from_numpy(sample[:1]),))
    path = WORK / "micronet_folded.tflite"
    edge.export(str(path))
    interp = Interpreter(model_path=str(path), num_threads=1)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]["index"]
    out = interp.get_output_details()[0]["index"]

    def predict(x):
        interp.set_tensor(inp, np.ascontiguousarray(x))
        interp.invoke()
        return np.array(interp.get_tensor(out), copy=True)

    version = getattr(ai_edge_litert, "__version__", "2.3.0")
    return predict, {"version": version, "torch": torch.__version__, "file_bytes": path.stat().st_size, "threads": 1}


def backend_executorch(model, sample, ctx):
    import executorch
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from executorch.runtime import Runtime

    try:
        from executorch.extension.pybindings.portable_lib import _unsafe_reset_threadpool

        _unsafe_reset_threadpool(1)
    except ImportError:
        pass
    exported = torch.export.export(model, (torch.from_numpy(sample[:1]),))
    program = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()]).to_executorch()
    path = WORK / "micronet_folded.pte"
    path.write_bytes(program.buffer)
    method = Runtime.get().load_program(str(path)).load_method("forward")

    def predict(x):
        return np.asarray(method.execute([torch.from_numpy(np.ascontiguousarray(x))])[0])

    version = getattr(executorch, "__version__", "1.6.0")
    return predict, {"version": version, "torch": torch.__version__, "file_bytes": path.stat().st_size,
                     "delegate": "xnnpack", "threads": 1}


def backend_tvm(model, sample, ctx):
    import onnx
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    path = WORK / "micronet_folded_b1.onnx"
    mod = from_onnx(onnx.load(path), {"x": (1, 3, 32, 32)})
    target = tvm.target.Target(TVM_TARGET)
    t0 = time.perf_counter()
    ex = tvm.compile(mod, target=target)
    compile_s = time.perf_counter() - t0
    vm = relax.VirtualMachine(ex, tvm.cpu())
    fn = vm["main"]

    def predict(x):
        out = fn(tvm.runtime.tensor(np.ascontiguousarray(x)))
        return (out if hasattr(out, "numpy") else out[0]).numpy()

    return predict, {"version": tvm.__version__, "target": str(target), "compile_wall_s": compile_s, "threads": 1}


BUILDERS = [
    ("torch", backend_torch),
    ("ort", backend_ort),
    ("ort-int8", backend_ort_int8),
    ("litert", backend_litert),
    ("executorch", backend_executorch),
    ("tvm", backend_tvm),
]
LABELS = {"torch": "PyTorch", "ort": "ORT fp32", "ort-int8": "ORT int8", "litert": "LiteRT",
          "executorch": "ExecuTorch", "tvm": "TVM"}


def plot(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r.get("status") == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    names = [LABELS.get(r["runtime"], r["runtime"]) for r in ok]
    axes[0].bar(names, [r["acc"] * 100 for r in ok], color="#2c5f8a")
    axes[0].set_ylabel("CIFAR-10 top-1 (%)")
    axes[0].set_ylim(min(r["acc"] for r in ok) * 100 - 1, max(r["acc"] for r in ok) * 100 + 0.5)
    axes[1].bar(names, [r["cpu_us_eval"] for r in ok], color="#b36b2c")
    axes[1].set_ylabel("CPU time per image (us)")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "runtime_compare.png", dpi=150)
    plt.close(fig)


def plot_trials() -> None:
    """Per-process CPU time, both clocks, log scale: how far apart fresh processes land."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = [name for name, _ in BUILDERS]
    for i, name in enumerate(names):
        trials = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RAW.glob(f"part-{name}-t*.json"))]
        ok = [t for t in trials if t.get("status") == "ok"]
        ax.scatter([i - 0.08] * len(ok), [t["cpu_us_eval"] for t in ok], color="#b36b2c", s=22,
                   label="10,000 distinct images" if i == 0 else None)
        ax.scatter([i + 0.08] * len(ok), [t["cpu_us"] for t in ok], color="#2c5f8a", marker="x", s=22,
                   label="one image x 200" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)), [LABELS[n] for n in names])
    ax.set_ylabel("CPU time per image (us)")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "runtime_trials.png", dpi=150)
    plt.close(fig)


def run_only(name: str, trial: int = 0) -> None:
    import resource

    from edgequant.data import load_cifar10, normalize

    RAW.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    build = dict(BUILDERS)[name]
    xtr, _, xte, yte = load_cifar10()
    x = normalize(xte).astype(np.float32)
    y = yte.astype(np.int64)
    sample = x[:1].copy()
    perm = np.random.default_rng(0).permutation(len(xtr))
    ctx = {"calib": normalize(xtr[perm[:CALIB_N]]).astype(np.float32)}
    del xtr
    if name in TORCH_FREE:
        sys.modules["torch"] = None
        model = None
        reference = np.load(WORK / "torch_logits.npy")[:1]
    else:
        import torch

        model = load_folded()
        with torch.inference_mode():
            reference = model(torch.from_numpy(sample)).numpy()
    print(f"== {name}", flush=True)
    try:
        predict, meta = build(model, sample, ctx)
        check = predict(sample)
        if check.shape != reference.shape:
            raise RuntimeError(f"shape {check.shape} != {reference.shape}")
        timed = time_one(predict, sample)
        acc, logits, cpu_us_eval = accuracy(predict, x, y)
        if name == "torch":
            np.save(WORK / "torch_logits.npy", logits)
            compared = {"logit_max_abs": 0.0, "argmax_agree": 1.0}
        else:
            compared = agreement(np.load(WORK / "torch_logits.npy"), logits)
        row = {"runtime": name, "status": "ok", "acc": acc, "cpu_us_eval": cpu_us_eval, **timed, **compared, **meta,
               "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    except Exception as exc:
        traceback.print_exc()
        row = {"runtime": name, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    row["trial"] = trial
    print(json.dumps(row, ensure_ascii=False), flush=True)
    (RAW / f"part-{name}-t{trial}.json").write_text(json.dumps(row) + "\n", encoding="utf-8")


def flat(rows: list[dict]) -> dict:
    """Headline fields for claims.json: ``<runtime>_<metric>`` with '-' replaced by '_'."""
    out = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = r["runtime"].replace("-", "_")
        for metric in ("acc", "cpu_us_eval", "cpu_us_eval_min", "cpu_us_eval_max", "cpu_us", "logit_max_abs",
                       "argmax_agree", "file_bytes"):
            if metric in r:
                out[f"{key}_{metric}"] = r[metric]
    by = {r["runtime"]: r for r in rows if r.get("status") == "ok"}
    if "torch" in by:
        for name, r in by.items():
            if name != "torch":
                out[f"speedup_{name.replace('-', '_')}_vs_torch"] = by["torch"]["cpu_us_eval"] / r["cpu_us_eval"]
    if "ort" in by and "ort-int8" in by:
        out["speedup_ort_int8_vs_ort"] = by["ort"]["cpu_us_eval"] / by["ort-int8"]["cpu_us_eval"]
    return out


def merge() -> None:
    from edge_harness.writer import JsonlWriter

    rows = []
    for name, _ in BUILDERS:
        parts = sorted(RAW.glob(f"part-{name}-t*.json"))
        if parts:
            rows.append(combine([json.loads(p.read_text(encoding="utf-8")) for p in parts]))
        else:
            rows.append({"runtime": name, "status": "error", "error": "no result file; the process was killed"})
    quota = read_quota()
    mem = Path("/sys/fs/cgroup/memory.max")
    raw_mem = mem.read_text().strip() if mem.exists() else ""
    summary = {
        "host": platform.node(),
        "cpu_quota_cores": quota,
        "cpu_count": os.cpu_count(),
        "memory_max_bytes": int(raw_mem) if raw_mem.isdigit() else None,
        "n_test": 10000,
        "repeats": REPEATS,
        **flat(rows),
        "rows": rows,
    }
    (RAW / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    writer = JsonlWriter(RAW / "runs.jsonl", cwd=REPO)
    writer.write({"kind": "host", "cpu_quota_cores": quota, "cpu_count": os.cpu_count()})
    for row in rows:
        writer.write(row)
    writer.close()
    try:
        plot(rows)
        plot_trials()
    except Exception:
        traceback.print_exc()
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


def interpreter(name: str) -> str:
    return os.environ.get(f"RTCMP_PY_{name.replace('-', '_').upper()}", sys.executable)


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--only":
        run_only(sys.argv[2], int(sys.argv[3]) if len(sys.argv) >= 4 else 0)
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "--merge":
        merge()
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "--plot":
        plot(json.loads((RAW / "summary.json").read_text(encoding="utf-8"))["rows"])
        plot_trials()
        return
    evict_dirs = [d for d in os.environ.get("RTCMP_EVICT", "").split(os.pathsep) if d]
    for trial in range(TRIALS):
        for name, _ in BUILDERS:
            if evict_dirs:
                from evict_page_cache import evict

                evict(evict_dirs)
            proc = subprocess.run([interpreter(name), __file__, "--only", name, str(trial)], check=False)
            if proc.returncode != 0:
                print(f"{name} trial {trial} exited {proc.returncode}", flush=True)
    merge()


if __name__ == "__main__":
    main()
