"""conv-kernels experiments. CPU only.

  python scripts/run_experiments.py --only winograd   # numerics, seconds
  python scripts/run_experiments.py --only bench      # timing; run on an idle machine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]
sys.path[:0] = [str(LAB / "code"), str(ROOT)]

from convlab.winograd import conv2d_direct, conv2d_winograd, transform  # noqa: E402
from edge_harness.cpp import build, run_json  # noqa: E402
from edge_harness.writer import JsonlWriter, read_rows, write_csv, write_json  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"

VARIANTS = [
    ("F(2,3)", 2, None),
    ("F(4,3)", 4, None),
    ("F(4,3) points 0..4", 4, [0, 1, 2, 3, 4, None]),
    ("F(6,3)", 6, None),
    ("F(6,3) points 0..+-3", 6, [0, 1, -1, 2, -2, 3, -3, None]),
]


def winograd_numerics() -> None:
    rng = np.random.default_rng(0)
    C, K, H = 64, 32, 26
    x = rng.normal(size=(C, H, H))
    w = rng.normal(size=(K, C, 3, 3)) / np.sqrt(9 * C)
    ref = conv2d_direct(x, w)
    scale = np.abs(ref).max()
    rows = []
    for dtype in (np.float32, np.float16):
        direct = conv2d_direct(x.astype(dtype), w.astype(dtype), dtype=dtype).astype(np.float64)
        rows.append({"variant": "direct", "dtype": np.dtype(dtype).name, "m": 1,
                     "max_rel": float(np.abs(direct - ref).max() / scale),
                     "rms_rel": float(np.sqrt(((direct - ref) ** 2).mean()) / np.sqrt((ref ** 2).mean())),
                     "mult_reduction": 1.0, "input_growth": 1.0})
        for name, m, pts in VARIANTS:
            t = transform(m, 3, pts)
            y = conv2d_winograd(x, w, t, dtype=dtype).astype(np.float64)
            err = y - ref
            rows.append({"variant": name, "dtype": np.dtype(dtype).name, "m": m,
                         "max_rel": float(np.abs(err).max() / scale),
                         "rms_rel": float(np.sqrt((err ** 2).mean()) / np.sqrt((ref ** 2).mean())),
                         "mult_reduction": t.reduction_2d(), "input_growth": t.input_growth_2d()})
    for r in rows:
        print(f"{r['dtype']:8s} {r['variant']:24s} max_rel {r['max_rel']:.2e} rms_rel {r['rms_rel']:.2e} "
              f"x{r['mult_reduction']:.2f} growth {r['input_growth']:.0f}")
    write_csv(RAW / "winograd_numerics.csv", rows)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for dtype, marker in (("float32", "o"), ("float16", "s")):
        sel = [r for r in rows if r["dtype"] == dtype]
        ax.semilogy(range(len(sel)), [r["rms_rel"] for r in sel], marker, ls="-", label=dtype)
    labels = [r["variant"] for r in rows if r["dtype"] == "float32"]
    ax.set_xticks(range(len(labels)), labels, rotation=20, fontsize=8)
    ax.set_ylabel("RMS error / RMS output")
    ax.set_title("Winograd error grows with tile size and with bad points")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "winograd_numerics.png", dpi=150)
    plt.close(fig)


def bench(repeats: int) -> None:
    exe = build(LAB / "cpp/conv.cpp", LAB / "cpp/bin", includes=(ROOT / "edge_harness/include",))
    check = run_json(exe, ["--mode=check"])
    write_csv(RAW / "cpp_check.csv", check)
    rows = run_json(exe, ["--mode=bench", "--layer=all", f"--repeats={repeats}"])
    with JsonlWriter(RAW / "conv_bench.jsonl", cwd=ROOT) as w:
        for r in rows:
            w.write(r)
    write_csv(RAW / "conv_bench.csv", [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows])
    import matplotlib.pyplot as plt

    layers = list(dict.fromkeys(r["layer"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.8 / len(methods)
    for mi, m in enumerate(methods):
        xs, ys = [], []
        for li, layer in enumerate(layers):
            hit = [r for r in rows if r["layer"] == layer and r["method"] == m]
            if hit:
                xs.append(li + mi * width)
                ys.append(hit[0]["gflops"])
        ax.bar(xs, ys, width, label=m)
    ax.set_xticks(np.arange(len(layers)) + 0.4 - width / 2, layers, rotation=20, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("GFLOP/s (1 thread, log)")
    ax.set_title("Same layer, different algorithm: depthwise needs its own kernel, 3x3 likes Winograd")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "conv_bench.png", dpi=150)
    plt.close(fig)


def summarize() -> None:
    out = {}
    path = RAW / "conv_bench.jsonl"
    if path.exists():
        rows = read_rows(path)
        for r in rows:
            out[f"gflops_{r['layer']}_{r['method']}"] = r["gflops"]
            out[f"workspace_{r['layer']}_{r['method']}"] = r["workspace_bytes"]
        out["bench_max_rel_err"] = max(r["rel_err"] for r in rows)
    num = RAW / "winograd_numerics.csv"
    if num.exists():
        import csv

        for r in csv.DictReader(open(num, encoding="utf-8")):
            out[f"wino_rms_{r['dtype']}_{r['variant']}"] = float(r["rms_rel"])
    write_json(RAW / "summary.json", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["winograd", "bench"])
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args()
    for step in args.only:
        {"winograd": winograd_numerics, "bench": lambda: bench(args.repeats)}[step]()
    summarize()


if __name__ == "__main__":
    main()
