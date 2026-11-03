"""Block-quantized GEMV: speed on this CPU, error on real GPT-2 weights, and the decode-rate bound.

  python scripts/run_experiments.py --only bench     # timing, idle machine, about 2 min
  python scripts/run_experiments.py --only gpt2      # needs .data/hf/models/gpt2 and wikitext-2 (hf-mirror), ~20 min

GPT-2 protocol matches mlsys-infra compiler-compression (`gpu/llm_utils.py`): wikitext-2 raw test joined
with blank lines, all 280 non-overlapping 1024-token windows, token-level cross-entropy. With every window
the sample order does not matter, so the fp32 number is directly comparable to that lab's fp16 29.96.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]
sys.path[:0] = [str(LAB / "code"), str(ROOT)]

from blockq.pack import QUANTIZERS, bits_per_weight, roundtrip, sqnr_db  # noqa: E402
from edge_harness import devices  # noqa: E402
from edge_harness.cpp import build, run_json  # noqa: E402
from edge_harness.writer import JsonlWriter, read_rows, write_csv, write_json  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"
HF = Path(os.environ.get("EDGE_HF_HOME", ROOT / ".data/hf"))


def bench(repeats: int) -> None:
    exe = build(LAB / "cpp/gemv.cpp", LAB / "cpp/bin", includes=(ROOT / "edge_harness/include",))
    rows = []
    for t in (1, 4):
        rows += run_json(exe, ["--mode=bench", f"--threads={t}", f"--repeats={repeats}"], timeout=1800)
    with JsonlWriter(RAW / "gemv_bench.jsonl", cwd=ROOT) as w:
        for r in rows:
            w.write(r)
    write_csv(RAW / "gemv_bench.csv", [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows])
    import matplotlib.pyplot as plt

    shape = "llama7b_4096x4096"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, label in ((axes[0], "us", "time per GEMV (us)"), (axes[1], "weight_GBps", "weight bytes / time (GB/s)")):
        methods = list(dict.fromkeys(r["method"] for r in rows))
        for ti, t in enumerate((1, 4)):
            vals = [next(r[key] for r in rows if r["shape"] == shape and r["method"] == m and r["threads"] == t)
                    for m in methods]
            ax.bar(np.arange(len(methods)) + ti * 0.4, vals, 0.4, label=f"{t} thread")
        ax.set_xticks(np.arange(len(methods)) + 0.2, methods, rotation=25, fontsize=7)
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{shape}: fewer weight bytes only help if the unpack keeps up")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "gemv_bench.png", dpi=150)
    plt.close(fig)


def _gpt2():
    from blockq.gpt2 import load_gpt2, load_tokenizer

    path = HF / "models/gpt2"
    return load_gpt2(path), load_tokenizer(path)


def _windows(tok, seq: int, n: int, seed: int):
    import pandas as pd
    import torch

    texts = pd.read_parquet(HF / "data/wikitext2/test.parquet")["text"].tolist()
    ids = torch.tensor(tok.encode("\n\n".join(texts)).ids, dtype=torch.long)
    k = ids.numel() // seq
    w = ids[: k * seq].view(k, seq)
    if not n or n >= k:
        return w
    g = torch.Generator().manual_seed(seed)
    return w[torch.randperm(k, generator=g)][:n]


def _ppl(model, wins, bs: int = 4) -> float:
    import torch
    from blockq.gpt2 import perplexity_from_logits

    nll, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(wins), bs):
            s, n = perplexity_from_logits(model(wins[i:i + bs]), wins[i:i + bs])
            nll, cnt = nll + s, cnt + n
    return math.exp(nll / cnt)


def _linears(model):
    return {n: m for n, m in model.named_modules() if type(m).__name__ == "Conv1D" and ".h." in f".{n}"}


def gpt2(windows: int, threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    model, tok = _gpt2()
    wins = _windows(tok, 1024, windows, seed=0)
    lin = _linears(model)
    orig = {n: m.weight.detach().clone() for n, m in lin.items()}  # Conv1D stores [in, out]

    def apply(fmt: str | None):
        for n, m in lin.items():
            w_oi = orig[n].t().contiguous().numpy()   # blocks run along the input dimension, as in GGUF
            q = w_oi if fmt is None else roundtrip(w_oi, fmt)
            m.weight.data.copy_(torch.from_numpy(np.ascontiguousarray(q.T)))

    rows, sq = [], []
    for fmt in (None, "q8_0", "q4_1", "q4_0"):
        apply(fmt)
        t0 = time.perf_counter()
        ppl = _ppl(model, wins)
        label = fmt or "fp32"
        if fmt:
            for n in lin:
                w_oi = orig[n].t().numpy()
                sq.append({"format": fmt, "layer": n, "sqnr_db": sqnr_db(w_oi, roundtrip(w_oi, fmt))})
        rows.append({"format": label, "bits_per_weight": 32.0 if fmt is None else bits_per_weight(fmt),
                     "ppl": ppl, "windows": len(wins), "tokens": int(wins[:, 1:].numel()),
                     "eval_s": time.perf_counter() - t0,
                     "mean_layer_sqnr_db": float(np.mean([r["sqnr_db"] for r in sq if r["format"] == fmt])) if fmt else None})
        print(rows[-1])
    apply(None)
    write_csv(RAW / "gpt2_ppl.csv", rows)
    write_csv(RAW / "gpt2_layer_sqnr.csv", sq)


def decode_bound() -> None:
    """Token/s upper bound = bandwidth / weight bytes per token, for the measured host and two edge boards."""
    rows = read_rows(RAW / "gemv_bench.jsonl") if (RAW / "gemv_bench.jsonl").exists() else []
    host = {}
    for r in rows:
        if r["shape"] == "llama7b_4096x4096":
            host[(r["method"], r["threads"])] = r["weight_GBps"]
    models = {"llama-7b": 6.74e9, "llama-1b": 1.24e9, "gpt2-124m": 124e6}
    out = []
    for name, params in models.items():
        for fmt, bpw in (("fp16", 16.0), ("q8_0", 8.5), ("q4_0", 4.5)):
            wbytes = params * bpw / 8
            row = {"model": name, "format": fmt, "weight_GB": wbytes / 1e9}
            for d in devices.all_devices():
                if d.dram_bw:
                    row[f"{d.slug}_tok_s_bound"] = d.dram_bw / wbytes
            meth = {"fp16": "fp16", "q8_0": "q8_0_x_q8_vnni", "q4_0": "q4_0_x_q8_vnni"}[fmt]
            for t in (1, 4):
                if (meth, t) in host:
                    row[f"host_{t}t_tok_s_at_measured_GBps"] = host[(meth, t)] * 1e9 / wbytes
            out.append(row)
    write_csv(RAW / "decode_bound.csv", out)


def summarize() -> None:
    out = {}
    if (RAW / "gemv_bench.jsonl").exists():
        for r in read_rows(RAW / "gemv_bench.jsonl"):
            out[f"{r['shape']}:{r['method']}:t{r['threads']}:GBps"] = r["weight_GBps"]
            out[f"{r['shape']}:{r['method']}:t{r['threads']}:us"] = r["us"]
    if (RAW / "gpt2_ppl.csv").exists():
        import csv

        for r in csv.DictReader(open(RAW / "gpt2_ppl.csv", encoding="utf-8")):
            out[f"gpt2_ppl_{r['format']}"] = float(r["ppl"])
            out["gpt2_windows"] = int(r["windows"])
            if r["mean_layer_sqnr_db"]:
                out[f"gpt2_mean_layer_sqnr_db_{r['format']}"] = float(r["mean_layer_sqnr_db"])
    write_json(RAW / "summary.json", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["bench", "gpt2"])
    ap.add_argument("--repeats", type=int, default=15)
    # 0 = every window of the test split (280), the set mlsys-infra evaluates; ~5 min per format on 4 cores
    ap.add_argument("--windows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    for step in args.only:
        {"bench": lambda: bench(args.repeats), "gpt2": lambda: gpt2(args.windows, args.threads)}[step]()
    decode_bound()
    summarize()


if __name__ == "__main__":
    _ = QUANTIZERS
    main()
