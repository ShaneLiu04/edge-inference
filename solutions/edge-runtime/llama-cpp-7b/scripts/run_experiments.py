"""llama.cpp on a real 7B: does the per-layer GEMV model from q4-gemv predict whole-model decode, and what
does each GGUF format cost in perplexity and KL divergence?

Stages (paths come from env, defaults are the AutoDL layout):
  python scripts/run_experiments.py prep      # CPU: HF -> f16 GGUF, wikitext files, Q8_0 / Q4_0 for the CPU sweep
  python scripts/run_experiments.py cpu       # CPU: llama-bench decode / prefill over 1..16 threads
  python scripts/run_experiments.py gpu       # GPU: imatrix, KLD base, then per format quantize -> bench -> PPL -> KLD
  python scripts/run_experiments.py batched   # GPU: llama-batched-bench, aggregate tok/s vs parallel sequences
  python scripts/run_experiments.py summarize # tables + figures from artifacts/raw only (runs anywhere)

Protocol: wikitext-2 raw test, llama-perplexity defaults (n_ctx 512, score the second half of each chunk),
all chunks for PPL, the first KLD_CHUNKS chunks for KL divergence against the f16 logits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]
sys.path[:0] = [str(LAB / "code"), str(ROOT)]

from lcpp.parse import bench_rows, final_ppl, kld_stats, quantize_sizes, wikitext_raw  # noqa: E402
from edge_harness.writer import provenance, write_csv, write_json  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"
LLAMA = Path(os.environ.get("LLAMA_CPP", "/root/autodl-tmp/llama.cpp"))
HF_MODEL = Path(os.environ.get("HF_MODEL", "/root/autodl-tmp/hf/models/Qwen2.5-7B"))
WIKI = Path(os.environ.get("WIKI_PARQUET", "/root/autodl-tmp/hf/data/wikitext2"))
WORK = Path(os.environ.get("GGUF_WORK", "/root/autodl-tmp/gguf"))        # f16 GGUF, imatrix, text files
QDIR = Path(os.environ.get("GGUF_QDIR", "/root/gguf_q"))                # quantized GGUFs (system disk)
KLD_BASE = Path(os.environ.get("KLD_BASE", "/root/gguf_q/f16.kld"))
CPU_CORES = os.environ.get("CPU_CORES", "16-31")
KLD_CHUNKS = int(os.environ.get("KLD_CHUNKS", "80"))
NAME = os.environ.get("GGUF_NAME", "qwen2.5-7b")

# (label, llama-quantize type, use imatrix). Kept files stay for the CPU sweep; the rest are deleted after use.
FORMATS = [("Q8_0", "Q8_0", False), ("Q6_K", "Q6_K", False), ("Q5_K_M", "Q5_K_M", False),
           ("Q4_K_M", "Q4_K_M", False), ("Q4_K_M+imatrix", "Q4_K_M", True), ("Q4_1", "Q4_1", False),
           ("Q4_0", "Q4_0", False), ("Q3_K_M", "Q3_K_M", False), ("Q2_K", "Q2_K", False),
           ("Q2_K+imatrix", "Q2_K", True)]
KEEP = {"Q8_0", "Q4_0", "Q4_K_M"}
EMBD_PARAMS = 152064 * 3584  # Qwen2.5-7B token_embd (untied from lm_head)


def _bin(build: str, tool: str) -> str:
    return str(LLAMA / build / "bin" / tool)


def _run(cmd: list[str], log: Path, timeout: float = 7200, pin: str | None = None) -> str:
    if pin:
        cmd = ["taskset", "-c", pin, *cmd]
    print("$", " ".join(map(str, cmd)), flush=True)
    t0 = time.time()
    p = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, timeout=timeout, errors="replace")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"$ {' '.join(map(str, cmd))}\n# rc={p.returncode} {time.time() - t0:.1f}s\n{p.stdout}\n{p.stderr}",
                   encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} rc={p.returncode}, see {log}")
    return p.stdout + "\n" + p.stderr


def _gguf(label: str) -> Path:
    return WORK / f"{NAME}-f16.gguf" if label == "F16" else QDIR / f"{NAME}-{label.replace('+', '-')}.gguf"


def _logs() -> Path:
    return RAW / "logs"


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(json.dumps(provenance(ROOT)) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prep() -> None:
    import pandas as pd

    WORK.mkdir(parents=True, exist_ok=True)
    QDIR.mkdir(parents=True, exist_ok=True)
    for split in ("test", "train"):
        out = WORK / f"wiki.{split}.raw"
        if not out.exists():
            out.write_text(wikitext_raw(pd.read_parquet(WIKI / f"{split}.parquet")["text"].tolist()), encoding="utf-8")
    f16 = _gguf("F16")
    if not f16.exists():
        _run([sys.executable, LLAMA / "convert_hf_to_gguf.py", HF_MODEL, "--outtype", "f16", "--outfile", f16],
             _logs() / "convert.log", pin=CPU_CORES)
    for label in ("Q8_0", "Q4_0"):
        quantize(label, label, False)


def quantize(label: str, qtype: str, imatrix: bool) -> dict:
    out = _gguf(label)
    log = _logs() / f"quantize_{label}.log"
    if not out.exists():
        cmd = [_bin("build-cpu", "llama-quantize")]
        if imatrix:
            cmd += ["--imatrix", WORK / "imatrix.gguf"]
        _run([*cmd, _gguf("F16"), out, qtype, os.environ.get("QUANT_THREADS", "16")], log, pin=CPU_CORES)
    row = {"format": label, "qtype": qtype, "imatrix": imatrix, "file_bytes": out.stat().st_size,
           **quantize_sizes(log.read_text(encoding="utf-8") if log.exists() else "")}
    rows = [r for r in _read_csv(RAW / "quant_sizes.csv") if r["format"] != label] + [row]
    write_csv(RAW / "quant_sizes.csv", rows)
    return row


def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []


def cpu() -> None:
    threads = os.environ.get("CPU_THREADS", "1,2,4,8,16")
    for label in ("Q4_0", "Q8_0", "Q4_K_M", "F16"):
        if not _gguf(label).exists():
            continue
        out = _run([_bin("build-cpu", "llama-bench"), "-m", _gguf(label), "-ngl", "0", "-t", threads,
                    "-p", "64", "-n", "32", "-r", "3", "-o", "jsonl"], _logs() / f"bench_cpu_{label}.log",
                   timeout=14400, pin=CPU_CORES)
        _append_jsonl(RAW / "bench_cpu.jsonl", [{**r, "format": label, "cores": CPU_CORES} for r in bench_rows(out)])


def gpu() -> None:
    f16 = _gguf("F16")
    imat = WORK / "imatrix.gguf"
    if not imat.exists():
        _run([_bin("build-cuda", "llama-imatrix"), "-m", f16, "-f", WORK / "wiki.train.raw", "-ngl", "99",
              "--chunks", "200", "-o", imat], _logs() / "imatrix.log")
    if not KLD_BASE.exists():
        out = _run([_bin("build-cuda", "llama-perplexity"), "-m", f16, "-f", WORK / "wiki.test.raw", "-ngl", "99",
                    "--chunks", str(KLD_CHUNKS), "--kl-divergence-base", KLD_BASE], _logs() / "kld_base.log")
    done = {r["format"] for r in _read_csv(RAW / "quality.csv")}
    for label, qtype, use_imat in [("F16", "F16", False), *FORMATS]:
        if label in done:
            continue
        size = quantize(label, qtype, use_imat) if label != "F16" else {"file_bytes": f16.stat().st_size}
        m = _gguf(label)
        out = _run([_bin("build-cuda", "llama-bench"), "-m", m, "-ngl", "99", "-fa", "off,on",
                    "-p", "512", "-n", "128", "-r", "5", "-o", "jsonl"], _logs() / f"bench_gpu_{label}.log")
        _append_jsonl(RAW / "bench_gpu.jsonl", [{**r, "format": label} for r in bench_rows(out)])
        out = _run([_bin("build-cuda", "llama-perplexity"), "-m", m, "-f", WORK / "wiki.test.raw", "-ngl", "99"],
                   _logs() / f"ppl_{label}.log")
        ppl, err = final_ppl(out)
        row = {"format": label, "file_GB": size["file_bytes"] / 1e9, "bpw": size.get("quant_bpw", 16.0),
               "ppl": ppl, "ppl_err": err}
        if label != "F16":
            out = _run([_bin("build-cuda", "llama-perplexity"), "-m", m, "-ngl", "99",
                        "--kl-divergence-base", KLD_BASE, "--kl-divergence"], _logs() / f"kld_{label}.log")
            k = kld_stats(out)
            row.update({"kld_mean": k.get("mean kld"), "kld_mean_err": k.get("mean kld err"),
                        "kld_median": k.get("median kld"), "kld_p99": k.get("99.0% kld"),
                        "kld_p999": k.get("99.9% kld"), "kld_max": k.get("maximum kld"),
                        "same_top_p": k.get("same top p"), "same_top_p_err": k.get("same top p err"),
                        "rms_dp": k.get("rms δp"), "ppl_ratio_kld_chunks": k.get("mean ppl(q)/ppl(base)")})
        rows = _read_csv(RAW / "quality.csv") + [row]
        write_csv(RAW / "quality.csv", rows)
        print(row, flush=True)
        if label not in KEEP and label != "F16":
            m.unlink()


def batched() -> None:
    for label in ("F16", "Q8_0", "Q4_0", "Q4_K_M"):
        if not _gguf(label).exists():
            continue
        out = _run([_bin("build-cuda", "llama-batched-bench"), "-m", _gguf(label), "-ngl", "99", "-fa", "on",
                    "-c", "131072", "-b", "2048", "-ub", "512", "-npp", "512", "-ntg", "128",
                    "-npl", "1,2,4,8,16,32,64,128", "--output-format", "jsonl"], _logs() / f"batched_{label}.log")
        rows = [{**json.loads(line), "format": label} for line in out.splitlines() if line.startswith("{")]
        _append_jsonl(RAW / "batched.jsonl", rows)


def summarize() -> None:
    from edge_harness.writer import read_rows

    out: dict = {}
    for r in _read_csv(RAW / "quality.csv"):
        f = r["format"]
        out[f"ppl_{f}"] = float(r["ppl"])
        out[f"ppl_err_{f}"] = float(r["ppl_err"])
        out[f"file_GB_{f}"] = float(r["file_GB"])
        if r.get("kld_mean"):
            out[f"kld_mean_{f}"] = float(r["kld_mean"])
            out[f"same_top_p_{f}"] = float(r["same_top_p"])
    if (RAW / "bench_gpu.jsonl").exists():
        for r in read_rows(RAW / "bench_gpu.jsonl"):
            out[f"gpu_{r['format']}_{r['test']}_fa{r.get('flash_attn')}_tok_s"] = r["avg_ts"]
    for r in _cpu_rows():
        out[f"cpu_{r['format']}_{r['test']}_t{r['n_threads']}_tok_s"] = r["avg_ts"]
    write_json(RAW / "summary.json", out)
    plot()


def _host_bw() -> tuple[dict, float]:
    """Measured Xeon DRAM read GB/s per thread count and the RTX 6000D read GB/s, from the sibling labs."""
    cpu, gpu = {}, 1333.0
    p = LAB.parent / "host-roofline/artifacts/raw/xeon8470q/summary.json"
    if p.exists():
        s = json.loads(p.read_text(encoding="utf-8"))
        cpu = {int(k[1:]): v["dram_read_gbps"] for k, v in s.items() if k[:1] == "t" and k[1:].isdigit()}
    p = LAB.parent / "q4-gemv/artifacts/raw/gpu/bandwidth.json"
    if p.exists():
        gpu = json.loads(p.read_text(encoding="utf-8"))["read_GBps"]
    return cpu, gpu


def _gpu_rows() -> list[dict]:
    from edge_harness.writer import read_rows

    return read_rows(RAW / "bench_gpu.jsonl") if (RAW / "bench_gpu.jsonl").exists() else []


def _cpu_rows() -> list[dict]:
    """bench_cpu.jsonl plus the per-format runs in raw/cpu_extra/<format>.jsonl (plain llama-bench output)."""
    from edge_harness.writer import read_rows

    rows = read_rows(RAW / "bench_cpu.jsonl") if (RAW / "bench_cpu.jsonl").exists() else []
    have = {r["format"] for r in rows}
    for p in sorted((RAW / "cpu_extra").glob("*.jsonl")):
        if p.stem in have:
            continue
        rows += [{**r, "format": p.stem} for r in bench_rows(p.read_text(encoding="utf-8"))]
    return rows


def _numbered(ax, pts: list[tuple[float, float, str]], loc: tuple[float, float]) -> None:
    """Number crowded points and list the labels in a key box at axes coords `loc` (top-left corner)."""
    for i, (x, y, _) in enumerate(pts, 1):
        ax.annotate(str(i), (x, y), fontsize=6.5, xytext=(3, 3), textcoords="offset points", color="#333")
    ax.text(*loc, "\n".join(f"{i}  {t}" for i, (_, _, t) in enumerate(pts, 1)), transform=ax.transAxes,
            fontsize=6.5, va="top", family="monospace", bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9))


def plot() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from edge_harness.writer import read_rows

    FIG.mkdir(parents=True, exist_ok=True)
    cpu_bw, gpu_bw = _host_bw()
    q = {r["format"]: r for r in _read_csv(RAW / "quality.csv")}
    if q:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        a, b = axes
        for f, r in q.items():
            if not r.get("kld_mean"):
                continue
            x, y = float(r["bpw"]), float(r["kld_mean"])
            im = "imatrix" in f
            a.errorbar(x, y, yerr=float(r["kld_mean_err"]), fmt="s" if im else "o", ms=5, capsize=2,
                       color="#E45756" if im else "#4C78A8", mfc="white" if im else None)
            a.annotate(f, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
            base = q.get(f.replace("+imatrix", ""))
            if im and base and base.get("kld_mean"):
                a.annotate("", (x, y), (float(base["bpw"]), float(base["kld_mean"])),
                           arrowprops=dict(arrowstyle="->", color="#E45756", lw=1))
        a.set_yscale("log")
        a.set_xlabel("bits per weight (quantized tensors)")
        a.set_ylabel("mean KL(f16 ‖ quant) per token")
        a.set_title("Quality falls ~3× per bit removed; arrows = what imatrix buys", fontsize=9)
        handles = [plt.Line2D([], [], marker="o", ls="", color="#4C78A8"),
                   plt.Line2D([], [], marker="s", ls="", color="#E45756", mfc="white")]
        a.legend(handles, ["plain", "+ imatrix"], fontsize=7)
        f16 = float(q["F16"]["ppl"]) if "F16" in q else None
        pts = []
        for f, r in sorted(q.items(), key=lambda kv: float(kv[1]["file_GB"])):
            x, y = float(r["file_GB"]), float(r["ppl"])
            b.errorbar(x, y, yerr=float(r["ppl_err"]), fmt="o", ms=4, capsize=2,
                       color="#E45756" if "imatrix" in f else "#4C78A8")
            if r.get("same_top_p"):
                pts.append((x, y, f"{f:<15s}{y:6.3f}  {float(r['same_top_p']):4.1f}%"))
        _numbered(b, pts, (0.42, 0.97))
        if f16:
            b.axhline(f16, color="k", ls="--", lw=0.8, label=f"F16 {f16:.3f}")
            b.legend(fontsize=7)
        b.set_xscale("log")
        b.set_xticks([4, 5, 6, 8, 10, 15], ["4", "5", "6", "8", "10", "15"])
        b.set_xlabel("GGUF file size (GB, log)")
        b.set_ylabel("wikitext-2 PPL (ctx 512, ±1 s.e.)")
        b.set_title("PPL barely moves above 5 bpw; key = format, PPL, same-top-1 vs F16", fontsize=9)
        for ax in axes:
            ax.grid(True, which="both", alpha=0.3)
        fig.suptitle("Qwen2.5-7B GGUF formats: size vs fidelity", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / "quality.png", dpi=150)
        plt.close(fig)

    sizes = {r["format"]: float(r["file_bytes"]) / 1e9 for r in _read_csv(RAW / "quant_sizes.csv")}
    sizes.setdefault("F16", float(q["F16"]["file_GB"]) if "F16" in q else 15.24)
    bpw = {r["format"]: float(r["quant_bpw"]) for r in _read_csv(RAW / "quant_sizes.csv")}
    bpw.setdefault("F16", 16.0)
    # Decode reads one row of token_embd per token, not the table: drop it, at the format's average bpw.
    read = {f: gb - EMBD_PARAMS * bpw.get(f, 16.0) / 8 / 1e9 for f, gb in sizes.items()}
    g = _gpu_rows()
    if g:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        a, b = axes
        fmts = [f for f in dict.fromkeys(r["format"] for r in g) if f in read]
        tg = {f: max((r for r in g if r["format"] == f and r["test"].startswith("tg")), key=lambda r: r["avg_ts"])
              for f in fmts}
        pp = {f: max((r for r in g if r["format"] == f and r["test"].startswith("pp")), key=lambda r: r["avg_ts"])
              for f in fmts}
        xs = sorted(read[f] for f in fmts)
        grid = [xs[0] * 0.85, xs[-1] * 1.15]
        a.plot(grid, [gpu_bw / x for x in grid], "k--", lw=0.9, label=f"weights / {gpu_bw:.0f} GB/s")
        pts = []
        for f in sorted(fmts, key=lambda f: read[f]):
            x, y = read[f], tg[f]["avg_ts"]
            a.scatter(x, y, s=22, color="#E45756" if "imatrix" in f else "#4C78A8", zorder=3)
            pts.append((x, y, f"{f:<15s}{y:5.0f} tok/s {100 * y * x / gpu_bw:3.0f}%"))
        _numbered(a, pts, (0.56, 0.86))
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("weight bytes read per token (GB, log; file minus token_embd)")
        a.set_xticks([4, 6, 8, 10, 14], ["4", "6", "8", "10", "14"])
        a.set_yticks([100, 150, 200, 300], ["100", "150", "200", "300"])
        a.set_ylabel("tg128 tok/s (batch 1, best of FA on/off)")
        a.set_title("Decode: every format sits under the bandwidth line (label = % of it)", fontsize=9)
        a.legend(fontsize=7)
        order = sorted(fmts, key=lambda f: -sizes[f])
        b.bar(range(len(order)), [pp[f]["avg_ts"] for f in order], color="#72B7B2")
        b.set_xticks(range(len(order)), order, rotation=30, ha="right", fontsize=7)
        b.set_ylabel("pp512 tok/s")
        b.set_title("Prefill is compute-bound: tok/s tracks the dequant kernel, not bytes", fontsize=9)
        for ax in axes:
            ax.grid(True, which="both", alpha=0.3)
        fig.suptitle(f"llama.cpp on {g[0].get('gpu_info', 'GPU')}, Qwen2.5-7B, all layers offloaded", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / "gpu_formats.png", dpi=150)
        plt.close(fig)

    cpu_rows = [r for r in _cpu_rows() if r["test"].startswith("tg")]
    if cpu_rows:
        rows = [r for r in cpu_rows if len({x["n_threads"] for x in cpu_rows if x["format"] == r["format"]}) > 2]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        a, b = axes
        for f in dict.fromkeys(r["format"] for r in rows):
            sel = sorted((r for r in rows if r["format"] == f), key=lambda r: r["n_threads"])
            ts = [r["n_threads"] for r in sel]
            a.plot(ts, [r["avg_ts"] for r in sel], marker="o", label=f)
            if f in read:
                b.plot(ts, [r["avg_ts"] * read[f] for r in sel], marker="o", label=f)
        if cpu_bw:
            ts = sorted(cpu_bw)
            b.plot(ts, [cpu_bw[t] for t in ts], "k--", lw=1, label="DRAM read (host-roofline)")
        a.set_ylabel("decode tok/s (tg32)")
        a.set_yscale("log")
        b.set_ylabel("tok/s × weight GB read per token = GB/s")
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
            ax.set_xlabel("threads (physical cores, one NUMA node)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        a.set_title("Fewer bytes per token = more tokens, at every thread count", fontsize=9)
        b.set_title("Every format rides the measured DRAM line", fontsize=9)
        fig.suptitle("Qwen2.5-7B decode on the Xeon 8470Q CPU", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / "cpu_decode.png", dpi=150)
        plt.close(fig)

    if cpu_rows and g and cpu_bw:
        t16 = {r["format"]: r["avg_ts"] for r in cpu_rows if r["n_threads"] == 16}
        tg_gpu = {f: max(r["avg_ts"] for r in g if r["format"] == f and r["test"].startswith("tg"))
                  for f in dict.fromkeys(r["format"] for r in g)}
        fmts = sorted((f for f in read if f in t16 and f in tg_gpu), key=lambda f: -read[f])
        if fmts:
            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            x = np.arange(len(fmts))
            cpu_e = [100 * t16[f] * read[f] / cpu_bw[16] for f in fmts]
            gpu_e = [100 * tg_gpu[f] * read[f] / gpu_bw for f in fmts]
            for off, ys, c, lab in [(-0.2, cpu_e, "#F58518", f"Xeon 16 threads ({cpu_bw[16]:.0f} GB/s)"),
                                    (0.2, gpu_e, "#4C78A8", f"RTX 6000D ({gpu_bw:.0f} GB/s)")]:
                bars = ax.bar(x + off, ys, 0.38, color=c, label=lab)
                for bb, y in zip(bars, ys):
                    ax.text(bb.get_x() + bb.get_width() / 2, y + 1, f"{y:.0f}", ha="center", fontsize=7)
            ax.set_xticks(x, [f"{f}\n{read[f]:.1f} GB" for f in fmts], fontsize=7)
            ax.set_ylabel("decode tok/s × bytes per token / measured read BW (%)")
            ax.axhline(100, color="k", lw=0.8, ls="--")
            ax.set_ylim(0, 115)
            ax.legend(fontsize=8, loc="lower left")
            ax.set_title("Fraction of the bandwidth line each format reaches (higher = less decode overhead)", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIG / "bw_efficiency.png", dpi=150)
            plt.close(fig)

    if (RAW / "batched.jsonl").exists():
        rows = read_rows(RAW / "batched.jsonl")
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        a, b = axes
        for f in dict.fromkeys(r["format"] for r in rows):
            sel = sorted((r for r in rows if r["format"] == f), key=lambda r: r["pl"])
            pl = [r["pl"] for r in sel]
            a.plot(pl, [r["speed_tg"] for r in sel], marker="o", label=f)
            b.plot(pl, [r["speed_tg"] / r["pl"] for r in sel], marker="o", label=f)
        a.set_ylabel("aggregate decode tok/s")
        b.set_ylabel("per-sequence decode tok/s")
        a.set_title("Batching multiplies throughput until compute binds", fontsize=9)
        b.set_title("...and each user's speed falls; 4-bit's lead shrinks", fontsize=9)
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("parallel sequences (prompt 512, generate 128)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "batched.png", dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["prep", "cpu", "gpu", "batched", "summarize"])
    args = ap.parse_args()
    for s in args.steps:
        globals()[s]()


if __name__ == "__main__":
    main()
