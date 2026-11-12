"""MIT 6.5940 Lab 4 on OPT-1.3B: FP16 vs RTN vs 1% salient fp16 vs AWQ scale search (+ clip). One GPU.

  AWQ_MODEL=.../opt-1.3b AWQ_PILE=.../val.jsonl.zst AWQ_WIKI=.../test.parquet python scripts/run_experiments.py

Perplexity: wikitext-2-raw test, 40 x 2048 tokens (the lab's protocol). Calibration: 128 pile-val texts, 512-token blocks.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[2]
sys.path[:0] = [str(LAB / "code"), str(REPO)]

import torch  # noqa: E402
from awqlab.opt import (  # noqa: E402
    apply_awq,
    apply_rtn,
    apply_salient,
    capture_inputs,
    load_opt,
    model_size_bits,
    perplexity,
    pile_calib,
    wikitext_test_ids,
)
from edge_harness.writer import JsonlWriter, write_json  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"
MODEL = os.environ.get("AWQ_MODEL", "/root/autodl-tmp/dl/opt-1.3b")
PILE = os.environ.get("AWQ_PILE", "/root/autodl-tmp/dl/pile/val.jsonl.zst")
WIKI = os.environ.get("AWQ_WIKI", "/root/autodl-tmp/dl/wikitext/test.parquet")
GROUP = 128
MiB = 8 * 1024 * 1024


def plot(rows: list[dict], alphas: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w3 = [r for r in rows if r["bits"] in (3, 16)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].barh([r["method"] for r in w3], [r["ppl"] for r in w3], color="#2c5f8a")
    for i, r in enumerate(w3):
        axes[0].text(r["ppl"], i, f" {r['ppl']:.2f}", va="center", fontsize=7)
    axes[0].set_xscale("log")
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_xticks([10, 20, 50, 100], ["10", "20", "50", "100"])
    axes[0].set_xlabel("wikitext-2 perplexity (log), W3 g128")
    axes[0].invert_yaxis()
    tags = ["qkv", "out", "fc1", "fc2"]
    for t in tags:
        ks = sorted((int(k.split(".")[0]), v) for k, v in alphas.items() if k.endswith(t))
        axes[1].plot([k for k, _ in ks], [v for _, v in ks], "o-", ms=3, label=t)
    axes[1].set_xlabel("decoder layer")
    axes[1].set_ylabel("searched alpha (W3)")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "awq_opt1.3b.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model, tok = load_opt(MODEL)
    orig = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ids = wikitext_test_ids(WIKI, tok)
    calib = pile_calib(PILE, tok)
    summary = {"model": Path(MODEL).name, "device": torch.cuda.get_device_name(), "torch": torch.__version__,
               "eval_tokens": 40 * 2048, "calib_blocks": len(calib), "group": GROUP}
    rows = []

    def reset():
        model.load_state_dict(orig)

    def record(method, bits, extra=None, size_frac=0.0):
        t0 = time.perf_counter()
        ppl = perplexity(model, ids)
        row = {"method": method, "bits": bits, "ppl": ppl, "eval_s": time.perf_counter() - t0,
               "size_mib": model_size_bits(model, None if bits == 16 else bits, GROUP, size_frac) / MiB}
        row.update(extra or {})
        rows.append(row)
        w.write(row)
        print(row, flush=True)

    with JsonlWriter(RAW / "runs.jsonl") as w:
        record("FP16", 16)
        feats = capture_inputs(model, calib)
        for bits in (3, 4):
            reset()
            apply_rtn(model, bits, GROUP)
            record(f"RTN W{bits}", bits)
        for mode in ("activation", "weight", "random"):
            reset()
            apply_salient(model, feats, 0.01, 3, GROUP, mode)
            record(f"1% fp16 ({mode})", 3, {"frac": 0.01}, 0.01)
        for frac in (0.001, 0.03):
            reset()
            apply_salient(model, feats, frac, 3, GROUP, "activation")
            record(f"{frac:.1%} fp16 (activation)", 3, {"frac": frac}, frac)
        del feats
        logs = {}
        for bits in (3, 4):
            for clip in (False, True):
                reset()
                feats = capture_inputs(model, calib)
                t0 = time.perf_counter()
                log = apply_awq(model, feats, bits, GROUP, clip=clip)
                search_s = time.perf_counter() - t0
                del feats
                name = f"AWQ W{bits}" + (" + clip" if clip else "")
                logs[name] = log
                record(name, bits, {"search_s": search_s})
    summary["rows"] = rows
    summary["alpha_w3"] = logs["AWQ W3"]["alpha"]
    write_json(RAW / "summary.json", summary)
    plot(rows, logs["AWQ W3"]["alpha"])


if __name__ == "__main__":
    if "--replot" in sys.argv:
        import json
        s = json.loads((RAW / "summary.json").read_text(encoding="utf-8"))
        plot(s["rows"], s["alpha_w3"])
    else:
        main()
