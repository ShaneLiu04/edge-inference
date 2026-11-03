"""int8-kernels experiments. CPU only.

  python scripts/run_experiments.py                 # everything
  python scripts/run_experiments.py --only requant rounding saturate
  python scripts/run_experiments.py --only gemm ort  # timing; run on an idle machine

Outputs under artifacts/raw and artifacts/figures.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]
sys.path[:0] = [str(LAB / "code"), str(ROOT)]

from edge_harness.cpp import build, run_json  # noqa: E402
from edge_harness.timing import time_calls  # noqa: E402
from edge_harness.writer import JsonlWriter, read_rows, write_csv, write_json  # noqa: E402
from int8ref.layer import acc_int32, random_layer, read_outputs, write_layer  # noqa: E402
from int8ref.requant import (  # noqa: E402
    quantize_multiplier,
    requant_all,
    requant_double,
    requant_fixed,
    requant_fixed_single,
)

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"
WORK = LAB / "artifacts/work"

LAYERS = [
    ("conv3x3_28", 28, 32, 64, 3, 1, 1),
    ("pointwise_28", 28, 64, 128, 1, 1, 1),
    ("depthwise_28", 28, 96, 96, 3, 1, 96),
    ("depthwise_s2_28", 28, 96, 96, 3, 2, 96),
    ("stem_s2_56", 56, 3, 16, 3, 2, 1),
    ("grouped4_28", 28, 64, 64, 3, 1, 4),
]
STYLES = ("fixed", "fixed_single", "float", "double")


def exe() -> Path:
    return build(LAB / "cpp/int8.cpp", LAB / "cpp/bin", includes=(ROOT / "edge_harness/include",))


def requant_agreement() -> None:
    from int8ref.ort import run_qlinearconv

    binary = exe()
    rows = []
    for ties in (False, True):
        for name, h, ic, oc, k, s, g in LAYERS:
            layer = random_layer(np.random.default_rng(0), h, ic, oc, k, s, g, name)
            if ties:
                layer.sw = np.full(oc, float(layer.sw.max()))
                layer.sy = float(layer.sx * layer.sw[0] * 64)
                layer.zy = 128
            d = WORK / f"{name}{'_ties' if ties else ''}"
            write_layer(layer, d)
            info = run_json(binary, ["--mode=conv", f"--dir={d}"])[0]
            cpp = read_outputs(layer, d)
            ref = acc_int32(layer)
            py = requant_all(ref, layer.sx, layer.sw, layer.sy, layer.zy)
            y = run_qlinearconv(layer)
            row = {"layer": name, "ties": ties, "outputs": int(y.size), "acc_absmax": info["acc_absmax"],
                   "acc_bound": int(255 * 127 * k * k * (ic // g)),
                   "cpp_acc_exact": bool(np.array_equal(cpp["acc"], ref))}
            for st in ("fixed", "float", "double"):
                row[f"cpp_{st}_exact"] = bool(np.array_equal(cpp[st], py[st]))
            for st in STYLES:
                row[f"ort_match_{st}"] = float((y == py[st]).mean())
                row[f"ort_maxdiff_{st}"] = int(np.abs(y.astype(int) - py[st]).max())
            rows.append(row)
            print(name, "ties" if ties else "", {k: v for k, v in row.items() if k.startswith("ort_match")})
    write_csv(RAW / "requant_agreement.csv", rows)


def rounding_rate() -> None:
    """Double-rounding mismatch of the two-step TFLite path versus exact rounding, against 2^-(shift+1)."""
    rng = np.random.default_rng(1)
    acc = rng.integers(-(1 << 22), 1 << 22, 1_000_000)
    rows = []
    for m in np.exp(rng.uniform(math.log(1e-5), math.log(0.5), 40)):
        m = float(m)
        m0, shift = quantize_multiplier(m)
        exact = requant_double(acc, m)
        rows.append({"m": m, "shift": shift, "predicted": 2.0 ** -(shift + 1),
                     "two_step": float((requant_fixed(acc, m0, shift) != exact).mean()),
                     "single": float((requant_fixed_single(acc, m0, shift) != exact).mean())})
    rows.sort(key=lambda r: r["m"])
    write_csv(RAW / "double_rounding.csv", rows)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.loglog([r["m"] for r in rows], [r["two_step"] for r in rows], "o", label="TFLite two-step (SRDHM + RDBPOT)")
    ax.loglog([r["m"] for r in rows], [r["predicted"] for r in rows], "-", label="predicted 2^-(shift+1)")
    single = [(r["m"], r["single"]) for r in rows if r["single"] > 0]
    if single:
        ax.loglog(*zip(*single), "x", label="TFLITE_SINGLE_ROUNDING")
    ax.set_xlabel("real multiplier sx*sw/sy")
    ax.set_ylabel("fraction of outputs off by one")
    ax.set_title("Two-step fixed-point requant: off-by-one rate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "double_rounding.png", dpi=150)
    plt.close(fig)


def saturate() -> None:
    write_csv(RAW / "maddubs_saturation.csv", run_json(exe(), ["--mode=saturate"]))


def gemm(repeats: int) -> None:
    rows = run_json(exe(), ["--mode=bench", f"--repeats={repeats}"])
    with JsonlWriter(RAW / "gemm_bench.jsonl", cwd=ROOT) as w:
        for r in rows:
            w.write(r)
    flat = [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows]
    write_csv(RAW / "gemm_bench.csv", flat)
    import matplotlib.pyplot as plt

    shapes = list(dict.fromkeys(r["shape"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / len(methods)
    for mi, m in enumerate(methods):
        vals = [next(r["gops"] for r in rows if r["shape"] == s and r["method"] == m) for s in shapes]
        ax.bar(np.arange(len(shapes)) + mi * width, vals, width, label=m)
    ax.set_xticks(np.arange(len(shapes)) + 0.4 - width / 2, shapes, rotation=15, fontsize=8)
    ax.set_ylabel("GOP/s (1 thread)")
    ax.set_title("Pointwise GEMM: layout and epilogue decide int8 speed")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "gemm_bench.png", dpi=150)
    plt.close(fig)


def ort_speed(repeats: int) -> None:
    """ONNX Runtime QLinearConv versus fp32 Conv on the same MobileNetV2 pointwise and depthwise layers."""
    import onnxruntime as ort
    from int8ref.ort import qlinearconv_model
    from onnx import TensorProto, helper, numpy_helper

    shapes = [("pw_expand_56", 56, 24, 144, 1, 1), ("pw_project_56", 56, 144, 24, 1, 1),
              ("pw_expand_14", 14, 96, 576, 1, 1), ("dw_56", 56, 144, 144, 3, 144), ("conv3x3_28", 28, 64, 64, 3, 1)]
    rows = []
    for name, h, ic, oc, k, g in shapes:
        layer = random_layer(np.random.default_rng(2), h, ic, oc, k, 1, g, name)
        wf = (layer.w.astype(np.float32) * layer.sw[:, None, None, None].astype(np.float32)).transpose(0, 3, 1, 2)
        node = helper.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[k, k], pads=[k // 2] * 4, group=g)
        graph = helper.make_graph([node], name, [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, ic, h, h])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
                                  initializer=[numpy_helper.from_array(np.ascontiguousarray(wf), "w"),
                                               numpy_helper.from_array(np.zeros(oc, np.float32), "b")])
        fp32 = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        fp32.ir_version = 8
        xq = np.ascontiguousarray(layer.x.transpose(2, 0, 1)[None])
        xf = xq.astype(np.float32) * np.float32(layer.sx)
        macs = h * h * oc * k * k * (ic // g)
        for threads in (1, 4):
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            for kind, model, feed in (("fp32", fp32, xf), ("int8", qlinearconv_model(layer), xq)):
                sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
                t = time_calls(lambda: sess.run(None, {"x": feed}), warmup=5, repeats=repeats)
                rows.append({"layer": name, "threads": threads, "kind": kind, "macs": macs,
                             "p50_us": t.p50_s * 1e6, "gops": 2 * macs / t.p50_s / 1e9})
        print(name, [f"{r['kind']}/{r['threads']}t {r['gops']:.1f}" for r in rows if r["layer"] == name])
    write_csv(RAW / "ort_int8_vs_fp32.csv", rows)
    summary = {}
    for name, *_ in shapes:
        for threads in (1, 4):
            f = next(r for r in rows if r["layer"] == name and r["threads"] == threads and r["kind"] == "fp32")
            q = next(r for r in rows if r["layer"] == name and r["threads"] == threads and r["kind"] == "int8")
            summary[f"{name}_t{threads}"] = f["p50_us"] / q["p50_us"]
    write_json(RAW / "ort_int8_speedup.json", summary)


def summarize() -> None:
    """Headline numbers that claims.json points at."""
    out = {}
    agree = RAW / "requant_agreement.csv"
    if agree.exists():
        import csv

        rows = list(csv.DictReader(open(agree, encoding="utf-8")))
        for st in STYLES:
            for ties in ("False", "True"):
                sel = [float(r[f"ort_match_{st}"]) for r in rows if r["ties"] == ties]
                out[f"ort_match_{st}_{'ties' if ties == 'True' else 'normal'}_min"] = min(sel)
        out["cpp_all_exact"] = all(r[k] == "True" for r in rows for k in r if k.startswith("cpp_"))
    sat = RAW / "maddubs_saturation.csv"
    if sat.exists():
        import csv

        for r in csv.DictReader(open(sat, encoding="utf-8")):
            out[f"maddubs_wrong_frac_w{r['wmax']}"] = float(r["wrong_frac"])
    bench = RAW / "gemm_bench.jsonl"
    if bench.exists():
        rows = read_rows(bench)
        for r in rows:
            out[f"gops_{r['shape']}_{r['method']}"] = r["gops"]
        out["gemm_all_exact"] = all(r["exact"] for r in rows)
    write_json(RAW / "summary.json", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["requant", "rounding", "saturate", "gemm", "ort"])
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args()
    steps = {"requant": requant_agreement, "rounding": rounding_rate, "saturate": saturate,
             "gemm": lambda: gemm(args.repeats), "ort": lambda: ort_speed(args.repeats * 4)}
    for name in args.only:
        steps[name]()
    summarize()


if __name__ == "__main__":
    main()
