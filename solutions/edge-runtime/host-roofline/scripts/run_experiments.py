"""Measure this host's roofline and place the other labs' kernels on it. CPU only; run on an idle machine.

  python scripts/run_experiments.py              # sweep (1 and 4 threads), peaks, figure
  python scripts/run_experiments.py --plot-only  # redraw from artifacts/raw
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]
sys.path[:0] = [str(LAB / "code"), str(ROOT)]

from edge_harness import devices  # noqa: E402
from edge_harness.cpp import build, run_json  # noqa: E402
from edge_harness.writer import JsonlWriter, read_rows, write_csv, write_json  # noqa: E402
from hostroof.analysis import attainable, conv_intensity, efficiency, plateaus, ridge  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"
THREADS = (1, 4)


def measure(max_mib: int) -> None:
    exe = build(LAB / "cpp/roofline.cpp", LAB / "cpp/bin", includes=(ROOT / "edge_harness/include",))
    with JsonlWriter(RAW / "roofline.jsonl", cwd=ROOT) as w:
        for t in THREADS:
            for r in run_json(exe, ["--mode=bw", f"--threads={t}", f"--max-mib={max_mib}"], timeout=1800):
                w.write(r)
            for mode in ("fma", "int8"):
                for r in run_json(exe, [f"--mode={mode}", f"--threads={t}"]):
                    w.write(r)
            print(f"threads={t} done")


def analyse() -> dict:
    rows = read_rows(RAW / "roofline.jsonl")
    out: dict = {}
    for t in THREADS:
        reads = sorted((r for r in rows if r["kind"] == "bw" and r["kernel"] == "read" and r["threads"] == t),
                       key=lambda r: r["bytes"])
        triads = sorted((r for r in rows if r["kind"] == "bw" and r["kernel"] == "triad" and r["threads"] == t),
                        key=lambda r: r["bytes"])
        levels = plateaus([r["bytes"] for r in reads], [r["GBps"] for r in reads])
        # The ends of the sweep, not the plateau medians: a plateau can swallow a slow ramp.
        out[f"t{t}"] = {
            "levels": [lv.__dict__ for lv in levels],
            "dram_read_gbps": float(np.median([r["GBps"] for r in reads[-3:]])),
            "l1_read_gbps": float(np.median([r["GBps"] for r in reads[:3]])),
            "triad_dram_gbps": float(np.median([r["GBps"] for r in triads[-3:]])),
            "fp32_gflops": next(r["Gops"] for r in rows if r["kind"] == "fma" and r["threads"] == t),
            "int8_gops": next(r["Gops"] for r in rows if r["kind"] == "int8" and r["threads"] == t),
        }
        o = out[f"t{t}"]
        o["fp32_ridge"] = ridge(o["fp32_gflops"], o["dram_read_gbps"])
        o["int8_ridge"] = ridge(o["int8_gops"], o["dram_read_gbps"])
        o["int8_over_fp32"] = o["int8_gops"] / o["fp32_gflops"]
    out["edge_devices"] = [{"slug": d.slug, "dram_gbps": d.dram_bw / 1e9 if d.dram_bw else None}
                           for d in devices.all_devices()]
    write_json(RAW / "summary.json", out)
    return out


def lab_points(peak: float, bw: float) -> list[dict]:
    """Single-thread kernel results from conv-kernels and int8-kernels, with their best-case intensity."""
    pts = []
    conv = ROOT / "solutions/edge-runtime/conv-kernels/artifacts/raw/conv_bench.csv"
    if conv.exists():
        for r in csv.DictReader(open(conv, encoding="utf-8")):
            ic, oc, k = int(r["IC"]), int(r["OC"]), int(r["K"])
            wbytes = 4 * k * k * (1 if r["depthwise"] == "True" else ic) * oc
            i = conv_intensity(float(r["macs"]), float(r["act_bytes"]), wbytes)
            pts.append({"src": "conv", "label": f"{r['layer']}:{r['method']}", "intensity": i,
                        "gops": float(r["gflops"]), "eff": efficiency(float(r["gflops"]), i, peak, bw)})
    gemm = ROOT / "solutions/edge-runtime/int8-kernels/artifacts/raw/gemm_bench.csv"
    if gemm.exists():
        for r in csv.DictReader(open(gemm, encoding="utf-8")):
            M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
            elem = 4 if r["method"].startswith("fp32") else 1
            i = 2 * M * N * K / (elem * (M * K + K * N) + (4 if elem == 4 else 1) * M * N)
            pts.append({"src": "gemm", "label": f"{r['shape']}:{r['method']}", "intensity": i,
                        "gops": float(r["gops"]), "eff": None})
    return pts


def plot(summary: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    s1 = summary["t1"]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.logspace(-2, 3, 200)
    for t, ls in zip(THREADS, ("-", "--")):
        s = summary[f"t{t}"]
        ax.loglog(xs, attainable(xs, s["fp32_gflops"], s["dram_read_gbps"]), "C0" + ls, label=f"fp32 roof, {t} thread")
        ax.loglog(xs, attainable(xs, s["int8_gops"], s["dram_read_gbps"]), "C3" + ls, label=f"int8 VNNI roof, {t} thread")
    ax.loglog(xs, attainable(xs, s1["fp32_gflops"], s1["l1_read_gbps"]), "C0:", lw=1, label="fp32 roof from L1, 1 thread")
    pts = lab_points(s1["fp32_gflops"], s1["dram_read_gbps"])
    for src, marker, color in (("conv", "o", "C2"), ("gemm", "^", "C1")):
        sel = [p for p in pts if p["src"] == src]
        if sel:
            ax.scatter([p["intensity"] for p in sel], [p["gops"] for p in sel], marker=marker, color=color, s=18,
                       label=f"{src} kernels (1 thread)", zorder=5)
    for d in summary["edge_devices"]:
        if d["dram_gbps"]:
            ax.axhline(d["dram_gbps"], color="grey", lw=0.6, ls="-.")
            ax.text(0.012, d["dram_gbps"] * 1.08, f"{d['slug']} DRAM GB/s", fontsize=7, color="grey")
    ax.set_xlabel("arithmetic intensity (op / DRAM byte, best case)")
    ax.set_ylabel("GOP/s")
    ax.set_title("Host roofline (measured) with lab kernels")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "roofline.png", dpi=150)
    plt.close(fig)
    if pts:
        write_csv(RAW / "lab_points.csv", pts)

    rows = read_rows(RAW / "roofline.jsonl")
    fig, ax = plt.subplots(figsize=(7, 4))
    for t, ls in zip(THREADS, ("-", "--")):
        for kernel, c in (("read", "C0"), ("triad", "C1")):
            sel = sorted((r for r in rows if r["kind"] == "bw" and r["kernel"] == kernel and r["threads"] == t),
                         key=lambda r: r["bytes"])
            ax.semilogx([r["bytes"] for r in sel], [r["GBps"] for r in sel], c + ls, marker=".", label=f"{kernel}, {t}t")
    ax.set_xlabel("working set (bytes)")
    ax.set_ylabel("GB/s")
    ax.set_title("Bandwidth by working-set size: the cache hierarchy")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "bandwidth_sweep.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--max-mib", type=int, default=512)
    args = ap.parse_args()
    if not args.plot_only:
        measure(args.max_mib)
    summary = analyse()
    plot(summary)
    for t in THREADS:
        s = summary[f"t{t}"]
        print(t, {k: round(v, 2) for k, v in s.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
