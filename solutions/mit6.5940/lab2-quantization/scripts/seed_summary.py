"""Aggregate the quantization suite over the five GPU-trained seeds.

Usage: python scripts/seed_summary.py
Reads artifacts/seeds/s*/raw/summary.json (written by run_experiments.py --ckpt ... --out artifacts/seeds/sN)
and artifacts/raw/seeds/train_log_s*.jsonl. Writes artifacts/raw/seeds/seed_summary.csv and two figures.

Every accuracy is also reported as a paired drop, acc(method) - acc(fp32) of the same seed: the
seed-to-seed spread of the fp32 model (about 0.2 points) cancels, so differences between methods
that are smaller than that spread can still be resolved.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
SEEDS = LAB / "artifacts" / "seeds"
OUT = LAB / "artifacts" / "raw" / "seeds"
FIG = LAB / "artifacts" / "figures"
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}  # two-sided t quantile, n-1 dof

GROUPS = [
    ("weight only", ["w8_per_tensor", "w8_per_channel", "w6_per_tensor", "w6_per_channel",
                     "w6_per_channel_but_depthwise_per_tensor", "w4_per_tensor", "w4_per_channel",
                     "w4_per_channel_but_depthwise_per_tensor"]),
    ("W8A8 calibrators", ["w8a8_minmax", "w8a8_percentile", "w8a8_mse", "w8a8_kl", "w8a8_kl_nonzero"]),
    ("W8A6 calibrators", ["w8a6_minmax", "w8a6_percentile", "w8a6_mse", "w8a6_kl", "w8a6_kl_nonzero"]),
    ("QAT W4A8", ["qat_w4a8_epoch0", "qat_w4a8_final"]),
]


def load() -> list[dict]:
    runs = []
    for d in sorted(SEEDS.glob("s*")):
        p = d / "raw" / "summary.json"
        if p.is_file():
            s = json.loads(p.read_text(encoding="utf-8"))
            s["seed"] = int(d.name[1:])
            runs.append(s)
    if not runs:
        raise SystemExit(f"no summaries under {SEEDS}")
    return runs


def stats(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    n = len(a)
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    return dict(n=n, mean=float(a.mean()), std=sd, min=float(a.min()), max=float(a.max()),
                ci95=T95.get(n, 1.96) * sd / math.sqrt(n) if n > 1 else 0.0)


def main() -> None:
    runs = load()
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k != "seed" and all(k in r for r in runs)]
    rows = []
    for k in keys:
        vals = [r[k] for r in runs]
        row = dict(key=k, **stats(vals))
        if k.endswith("_acc") or k.startswith(("w8", "w6", "w4", "qat_")):
            d = stats([r[k] - r["fp32_acc"] for r in runs])
            row.update(drop_mean=d["mean"], drop_std=d["std"], drop_ci95=d["ci95"])
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["key", "n", "mean", "std", "ci95", "min", "max", "drop_mean", "drop_std", "drop_ci95"]
    with open(OUT / "seed_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    sens = Counter(r.get("most_sensitive_w4_layer") for r in runs)
    meta = dict(seeds=[r["seed"] for r in runs], most_sensitive_w4_layer=dict(sens),
                train=[json.loads(l) for p in sorted(OUT.glob("train_log_s*.jsonl"))
                       for l in p.read_text(encoding="utf-8").splitlines() if '"final"' in l])
    (OUT / "seed_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot(runs)
    by = {r["key"]: r for r in rows}
    print(f"{len(runs)} seeds; fp32 {by['fp32_acc']['mean']:.4f} ± {by['fp32_acc']['std']:.4f}; wrote {OUT / 'seed_summary.csv'}")


def plot(runs: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    labels, means, sds, pts, cols = [], [], [], [], []
    for gi, (_g, ks) in enumerate(GROUPS):
        for k in ks:
            if not all(k in r for r in runs):
                continue
            d = [100 * (r[k] - r["fp32_acc"]) for r in runs]
            labels.append(k.replace("_but_depthwise_per_tensor", " (dw per-tensor)").replace("_", " "))
            means.append(np.mean(d))
            sds.append(np.std(d, ddof=1))
            pts.append(d)
            cols.append(colors[gi])
    fig, ax = plt.subplots(figsize=(8.4, 0.3 * len(labels) + 1.6))
    ys = np.arange(len(labels))[::-1]
    ax.barh(ys, means, xerr=sds, color=cols, height=0.65, alpha=0.85, error_kw=dict(lw=1, capsize=2))
    for y, p in zip(ys, pts):
        ax.scatter(p, np.full(len(p), y), s=8, color="k", zorder=3)
    ax.set_yticks(ys, labels, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xscale("symlog", linthresh=1.0)
    ticks = [t for t in (-50, -20, -10, -5, -2, -1, 0, 1) if t >= min(min(p) for p in pts) * 1.3]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(min(min(p) for p in pts) * 1.3, 2.5)
    ax.set_xlabel("accuracy change vs. the same seed's fp32 (points; symlog)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, [g for g, _ in GROUPS], fontsize=7, loc="lower left")
    ax.set_title(f"Paired accuracy drop over {len(runs)} seeds (bar = mean, line = ±1 sd, dots = seeds)", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "seeds_quant.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for policy, c in (("by_sensitivity", "#E45756"), ("by_size", "#4C78A8")):
        ns = sorted({int(k.split("_")[-2]) for k in runs[0] if k.startswith(f"mixed_{policy}_") and k.endswith("_acc")})
        kib = [np.mean([r[f"mixed_{policy}_{n}_kib"] for r in runs]) for n in ns]
        acc = np.array([[100 * r[f"mixed_{policy}_{n}_acc"] for r in runs] for n in ns])
        ax.plot(kib, acc.mean(1), marker="o", color=c, label=policy.replace("_", " "))
        ax.fill_between(kib, acc.min(1), acc.max(1), color=c, alpha=0.18)
    fp32 = 100 * np.mean([r["fp32_acc"] for r in runs])
    ax.axhline(fp32, color="k", ls="--", lw=0.8, label=f"fp32 mean {fp32:.2f}%")
    ax.set_xscale("log")
    ax.set_xticks([100, 120, 140, 160, 200], ["100", "120", "140", "160", "200"])
    ax.minorticks_off()
    ax.set_xlabel("weight storage (KiB, log)")
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title("W4 base, promote layers to W8: which layers first? (band = seed min–max)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "seeds_mixed.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
