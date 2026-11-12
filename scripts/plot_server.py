"""Figures for the server reruns (Xeon 8470Q) and the RTX 6000D GEMV, drawn from raw files only.

Usage: python scripts/plot_server.py
Writes into each lab's artifacts/figures (xeon8470q/ for CPU reruns) and interview/figures/.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "solutions"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 160, "savefig.bbox": "tight"})
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def rows(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p, encoding="utf-8")))


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


# --------------------------------------------------------------------------- host roofline: thread scaling

def roofline_scaling() -> None:
    lab = SOL / "edge-runtime/host-roofline/artifacts"
    xr = jsonl(lab / "raw/xeon8470q/roofline.jsonl")
    lr = jsonl(lab / "raw/roofline.jsonl")
    xs = json.loads((lab / "raw/xeon8470q/summary.json").read_text(encoding="utf-8"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.1))
    threads = sorted({r["threads"] for r in xr if r["kind"] == "bw"})
    for i, t in enumerate(threads):
        pts = sorted((r["bytes"], r["GBps"]) for r in xr if r["kind"] == "bw" and r["kernel"] == "read" and r["threads"] == t)
        a.plot([p[0] for p in pts], [p[1] for p in pts], color=C[i], marker=".", lw=1.4, label=f"Xeon, {t} thr")
    for t, ls in ((1, "--"), (4, ":")):
        pts = sorted((r["bytes"], r["GBps"]) for r in lr
                     if r.get("kind") == "bw" and r.get("kernel") == "read" and r["threads"] == t)
        if pts:
            a.plot([p[0] for p in pts], [p[1] for p in pts], color="0.4", ls=ls, lw=1.2, label=f"local VM, {t} thr")
    for size, name in ((48 << 10, "L1 48K"), (2 << 20, "L2 2M"), (105 << 20, "L3 105M")):
        a.axvline(size, color="0.75", lw=0.8)
        a.text(size * 1.08, a.get_ylim()[1] if False else 3.2e3, name, fontsize=7, color="0.45", va="top")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xlabel("working set (bytes)")
    a.set_ylabel("read bandwidth (GB/s)")
    a.set_title("Read-bandwidth ladder (each thread scans its own slice)", fontsize=9)
    a.legend(fontsize=7, ncol=2)

    t = np.array([int(k[1:]) for k in xs if k.startswith("t")])
    t.sort()
    for key, label, col in (("dram_read_gbps", "DRAM read", C[0]), ("fp32_gflops", "fp32 FMA peak", C[1]),
                            ("int8_gops", "int8 VNNI peak", C[2])):
        v = np.array([xs[f"t{n}"][key] for n in t])
        b.plot(t, v / v[0], marker="o", color=col, label=f"{label} ({v[-1]:.0f} at 16)")
    b.plot(t, t, color="0.5", ls="--", lw=1, label="ideal (×threads)")
    b.axhline(307.2 / xs["t1"]["dram_read_gbps"], color=C[0], ls=":", lw=1)
    b.text(1.05, 307.2 / xs["t1"]["dram_read_gbps"] * 1.05, "8×DDR5-4800 = 307 GB/s", color=C[0], fontsize=7)
    b.set_xscale("log", base=2)
    b.set_yscale("log", base=2)
    b.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    b.set_yticks([1, 2, 4, 8, 16], ["1×", "2×", "4×", "8×", "16×"])
    b.set_xlabel("threads (physical cores, NUMA1)")
    b.set_ylabel("speedup over 1 thread")
    b.set_title("Compute scales ~linearly; DRAM bends after 8 threads", fontsize=9)
    b.legend(fontsize=7)
    fig.suptitle("Xeon Platinum 8470Q (Sapphire Rapids) host roofline", fontsize=10)
    save(fig, lab / "figures/xeon8470q/thread_scaling.png")


# --------------------------------------------------------------------------- int8: AMX vs K, epilogue share

def amx_vs_k() -> None:
    lab = SOL / "edge-runtime/int8-kernels/artifacts"
    r = rows(lab / "raw/xeon8470q/gemm_bench.csv")
    g = defaultdict(dict)
    shape_k = {}
    for x in r:
        g[x["shape"]][x["method"]] = float(x["gops"])
        shape_k[x["shape"]] = int(x["K"])
    shapes = sorted(g, key=lambda s: shape_k[s])
    k = np.array([shape_k[s] for s in shapes])
    ratio = np.array([g[s]["int8_amx_tile"] / g[s]["int8_vnni_packed"] for s in shapes])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.scatter(k, ratio, color=C[3], zorder=3, s=36, label="measured AMX / VNNI-packed")
    offsets = {"square_512": (-58, 6), "mbv2_pw_project_14": (6, -10), "llm_prefill_256x4096x4096": (-92, -4)}
    for kk, rr, s in zip(k, ratio, shapes):
        a.annotate(s.replace("mbv2_", "").replace("llm_prefill_256x4096x4096", "prefill 256×4096²"),
                   (kk, rr), textcoords="offset points", xytext=offsets.get(s, (5, 4)), fontsize=7)
    ref = ratio[list(shapes).index("square_512")] if "square_512" in shapes else ratio.max()
    kk = np.unique(np.concatenate([np.arange(16, 4097, 4), k]))
    a.plot(kk, ref * kk / (64 * np.ceil(kk / 64)), color="0.45", ls="--", lw=1,
           label="tile-fill bound: ratio(512³) × K / (64·⌈K/64⌉)")
    a.axhline(1, color="0.6", lw=0.8)
    a.set_xscale("log", base=2)
    a.set_xlim(16, 6000)
    a.set_xlabel("reduction length K")
    a.set_ylabel("AMX speedup over VNNI packed (1 thread)")
    a.set_title("AMX only wins once K fills the 64-byte tile", fontsize=9)
    a.legend(fontsize=7, loc="lower right")

    x = np.arange(len(shapes))
    loss_v = [1 - g[s]["int8_vnni_packed+simd_requant"] / g[s]["int8_vnni_packed"] for s in shapes]
    loss_a = [1 - g[s]["int8_amx_tile+simd_requant"] / g[s]["int8_amx_tile"] for s in shapes]
    b.bar(x - 0.2, loss_v, 0.4, label="VNNI packed + SIMD requant", color=C[0])
    b.bar(x + 0.2, loss_a, 0.4, label="AMX tile + SIMD requant", color=C[3])
    b.set_xticks(x, [f"K={kk}" for kk in k], fontsize=7)
    b.set_ylabel("throughput lost to a separate requant pass")
    b.set_title("Amdahl: a faster GEMM makes a separate epilogue cost more", fontsize=9)
    b.legend(fontsize=7)
    save(fig, lab / "figures/xeon8470q/amx_vs_k.png")


# --------------------------------------------------------------------------- q4 GEMV on the GPU

LABEL = {"cublas_fp16": "cuBLAS fp16", "triton_fp16": "Triton fp16", "q8_0_triton_gemv": "Triton Q8_0",
         "q4_0_triton_gemv": "Triton Q4_0", "q4_0_tensorcore_gemm_m1": "W4A16 TC (M=1)",
         "q4_0_dequant_then_cublas": "Q4_0 dequant + cuBLAS"}


def gemv_gpu() -> None:
    lab = SOL / "edge-runtime/q4-gemv/artifacts"
    r = [x for x in rows(lab / "raw/gpu/gemv_gpu.csv") if x["l2"] == "cold"]
    bw = json.loads((lab / "raw/gpu/bandwidth.json").read_text(encoding="utf-8"))["read_GBps"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.25, 1]})
    for i, m in enumerate(LABEL):
        for la, mk, fill in (("eager", "o", "none"), ("graph", "o", None)):
            sel = [x for x in r if x["method"] == m and x["launch"] == la]
            a.scatter([float(x["weight_bytes"]) for x in sel], [float(x["us"]) for x in sel], s=26, marker=mk,
                      facecolors=fill if fill else C[i], edgecolors=C[i], label=f"{LABEL[m]}" if la == "graph" else None)
    xs = np.logspace(5.8, 9.4, 50)
    a.plot(xs, xs / (bw * 1e3), color="k", lw=1, ls="--", label=f"bytes / {bw:.0f} GB/s")
    a.axhline(22, color="0.5", lw=0.8, ls=":")
    a.text(3e8, 16, "eager launch floor ≈ 22 µs", fontsize=7, color="0.4", ha="center")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("weight bytes read per call")
    a.set_ylabel("µs per GEMV (L2 cold)")
    a.set_title("GEMV time vs bytes: filled = CUDA graph, hollow = eager")
    a.legend(fontsize=7, loc="upper left")

    d = rows(lab / "raw/gpu/decode_pred.csv")
    methods = [m for m in LABEL if any(x["method"] == m for x in d)]
    x = np.arange(len(methods))
    for j, (la, col) in enumerate((("eager", "0.65"), ("graph", C[2]))):
        v = [float(next(y["tok_s_linear_only"] for y in d if y["method"] == m and y["launch"] == la)) for m in methods]
        b.bar(x + (j - 0.5) * 0.38, v, 0.38, color=col, label=la)
        if la == "graph":
            for xi, vi in zip(x, v):
                b.text(xi + 0.19, vi * 1.05, f"{vi:.0f}", ha="center", fontsize=7)
    ub = [float(next(y["tok_s_bound_at_read_bw"] for y in d if y["method"] == m)) for m in methods]
    b.scatter(x, ub, marker="_", s=400, color="k", zorder=3, label="bytes-only bound")
    b.set_yscale("log")
    b.set_xticks(x, [LABEL[m] for m in methods], rotation=25, fontsize=7, ha="right")
    b.set_ylabel("Qwen2.5-7B tok/s (linear layers only)")
    b.set_title("Decode prediction: Q4 pays off only once launches are cheap")
    b.legend(fontsize=7)
    fig.suptitle("RTX 6000D, batch-1 GEMV over Qwen2.5-7B / Llama-7B shapes", fontsize=10)
    save(fig, lab / "figures/gemv_gpu_roofline.png")


# --------------------------------------------------------------------------- DMA: model vs measured

def dma() -> None:
    lab = SOL / "edge-runtime/dma-double-buffer/artifacts"
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for path, name, col in ((lab / "raw/dma_measured.csv", "local VM", C[0]),
                            (lab / "raw/xeon8470q/dma_measured.csv", "Xeon 8470Q", C[3])):
        r = [x for x in rows(path) if int(x["tile_bytes"]) == 65536 and int(x["work"]) > 0]
        r.sort(key=lambda x: float(x["ratio_c_over_m"]))
        cm = [float(x["ratio_c_over_m"]) for x in r]
        ax.plot(cm, [float(x["model_speedup"]) for x in r], color=col, ls="--", lw=1, label=f"{name}: model")
        ax.plot(cm, [float(x["speedup"]) for x in r], color=col, marker="o", label=f"{name}: measured")
    ax.axhline(1, color="0.5", lw=0.8)
    c = np.logspace(-0.5, 1.3, 100)
    ax.plot(c, (1 + c) / np.maximum(1, c), color="k", lw=0.8, alpha=0.5, label="ideal (1+c/m)/max(1,c/m)")
    ax.set_xscale("log")
    ax.set_xlabel("compute / copy time per tile (c/m)")
    ax.set_ylabel("double-buffer speedup")
    ax.set_title("A copy thread is not a DMA engine: the server never wins")
    ax.legend(fontsize=7)
    save(fig, lab / "figures/xeon8470q/model_vs_measured.png")


def main() -> None:
    roofline_scaling()
    amx_vs_k()
    gemv_gpu()
    dma()


if __name__ == "__main__":
    main()
