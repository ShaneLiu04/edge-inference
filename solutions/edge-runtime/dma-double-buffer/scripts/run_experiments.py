"""Copy/compute overlap measured on this CPU, the event model, and a Dory-style tile search.

  python scripts/run_experiments.py --only model dory   # analytic, seconds
  python scripts/run_experiments.py --only measure      # timing, idle machine
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]
sys.path[:0] = [str(LAB / "code"), str(ROOT)]

from edge_harness.cpp import build, run_json  # noqa: E402
from edge_harness.writer import JsonlWriter, read_rows, write_csv, write_json  # noqa: E402
from tiling.dory import TARGETS, Conv, ideal_cycles, search  # noqa: E402
from tiling.pipeline import double_buffer_time, serial_time, simulate  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"


def measure(repeats: int) -> None:
    exe = build(LAB / "cpp/dma.cpp", LAB / "cpp/bin", includes=(ROOT / "edge_harness/include",))
    rows = run_json(exe, ["--mode=sweep", "--tile-kib=64", "--tiles=512", f"--repeats={repeats}"], timeout=3600)
    rows += run_json(exe, ["--mode=tiles", "--work=4", f"--repeats={repeats}"], timeout=3600)
    with JsonlWriter(RAW / "dma_measured.jsonl", cwd=ROOT) as w:
        for r in rows:
            w.write(r)
    write_csv(RAW / "dma_measured.csv", rows)
    import matplotlib.pyplot as plt

    sweep = [r for r in rows if r["tile_bytes"] == 64 * 1024 and r["tiles"] == 512]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = [r["ratio_c_over_m"] for r in sweep]
    ax.plot(x, [r["model_speedup"] for r in sweep], "k--", label="model (c+m) / (max + min/n)")
    ax.plot(x, [r["speedup"] for r in sweep], "o-", label="measured serial / double")
    ax.axvline(1.0, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("compute time / copy time per tile")
    ax.set_ylabel("speedup of double buffering")
    ax.set_title("Overlap pays most when copy and compute are balanced")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "dma_overlap.png", dpi=150)
    plt.close(fig)


def model() -> None:
    """Event simulation against the closed forms, including one shared DMA channel for in and out."""
    rows = []
    n = 64
    for ratio in (0.125, 0.25, 0.5, 1, 2, 4, 8):
        m, c, o = 1.0, ratio, 0.5
        rows.append({"c_over_m": ratio, "serial": serial_time(n, [m, c]), "double": double_buffer_time(n, c, m),
                     "sim_double": simulate(n, [m, c], 2), "sim_3stage_sep": simulate(n, [m, c, o], 3),
                     "sim_3stage_shared_dma": simulate(n, [m, c, o], 2, shared_dma=True)})
    write_csv(RAW / "pipeline_model.csv", rows)


def dory() -> None:
    layers = {"conv3x3_32x32x64": Conv(32, 64, 64), "conv3x3_16x16x128": Conv(16, 128, 128),
              "pointwise_32x32x96": Conv(32, 96, 96, r=1), "depthwise3x3_32x32x96": Conv(32, 96, 96, depthwise=True)}
    rows = []
    for lname, layer in layers.items():
        for tname, t in TARGETS.items():
            for l1_kib in (16, 32, 64, 128):
                tt = replace(t, l1_bytes=l1_kib * 1024)
                best = {bufs: search(layer, tt, bufs) for bufs in (1, 2)}
                if not best[1] or not best[2]:
                    continue
                for buffers in (1, 2):
                    b = best[buffers][0]
                    rows.append({"layer": lname, "target": tname, "l1_kib": l1_kib, "buffers": buffers,
                                 "th": b.th, "tk": b.tk, "tc": b.tc, "tile_l1_bytes": b.l1_bytes, "n_tiles": b.n_tiles,
                                 "dma_cycles": b.dma_cycles, "compute_cycles": b.compute_cycles,
                                 "bound": "dma" if b.dma_cycles > b.compute_cycles else "compute",
                                 "total_cycles": b.total_cycles, "ideal_cycles": ideal_cycles(layer, tt),
                                 "efficiency": ideal_cycles(layer, tt) / b.total_cycles,
                                 "double_over_single": best[1][0].total_cycles / best[2][0].total_cycles})
    write_csv(RAW / "dory_tiles.csv", rows)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(13, 3.8), sharey=True)
    for ax, tname in zip(axes, TARGETS):
        for lname in layers:
            for buffers, ls in ((1, "--"), (2, "-")):
                sel = [r for r in rows if r["layer"] == lname and r["target"] == tname and r["buffers"] == buffers]
                ax.plot([r["l1_kib"] for r in sel], [r["efficiency"] for r in sel], ls, marker="o",
                        label=f"{lname}, {buffers} buf")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("L1 size (KiB)")
        ax.set_title(tname)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("ideal compute time / tiled time")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "dory_efficiency.png", dpi=150)
    plt.close(fig)


def summarize() -> None:
    import csv

    out = {}
    if (RAW / "dory_tiles.csv").exists():
        for r in csv.DictReader(open(RAW / "dory_tiles.csv", encoding="utf-8")):
            if r["buffers"] == "2":
                key = f"dory:{r['layer']}:{r['target']}:l1_{r['l1_kib']}k"
                out[f"{key}:efficiency"] = float(r["efficiency"])
                out[f"{key}:double_over_single"] = float(r["double_over_single"])
    if (RAW / "dma_measured.jsonl").exists():
        for r in read_rows(RAW / "dma_measured.jsonl"):
            key = f"tile{r['tile_bytes'] // 1024}k_work{r['work']}"
            out[f"{key}:speedup"] = r["speedup"]
            out[f"{key}:model_speedup"] = r["model_speedup"]
            out[f"{key}:ratio"] = r["ratio_c_over_m"]
    write_json(RAW / "summary.json", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["model", "dory", "measure"])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    for step in args.only:
        {"model": model, "dory": dory, "measure": lambda: measure(args.repeats)}[step]()
    summarize()


if __name__ == "__main__":
    main()
