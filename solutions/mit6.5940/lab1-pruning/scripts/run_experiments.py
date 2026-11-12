"""MIT 6.5940 Lab 1 on the course VGG: fine-grained pruning to 25% size, then channel pruning.

  python scripts/run_experiments.py            # everything; GPU recommended (about 5 minutes on one card)
  python scripts/run_experiments.py --latency  # CPU latency of the saved models on this host only

Needs the course checkpoint at $LAB1_CKPT or artifacts/work/vgg.cifar.pretrained.pth
(https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth) and .data/cifar10.npz.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[2]
sys.path[:0] = [str(LAB / "code"), str(LAB.parent / "lab2-quantization" / "code"), str(REPO)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from edge_harness.timing import time_calls  # noqa: E402
from edge_harness.writer import JsonlWriter, write_csv, write_json  # noqa: E402
from edgequant.data import load_cifar10  # noqa: E402
from prune.channel import apply_channel_sorting, channel_prune, count_macs  # noqa: E402
from prune.fine import (  # noqa: E402
    FineGrainedPruner,
    choose_sparsities,
    fine_grained_prune,
    get_model_size,
    get_num_parameters,
    get_sparsity,
    kept_fraction,
    prunable,
    sensitivity_scan,
)
from prune.train import evaluate, finetune, to_tensors  # noqa: E402
from prune.vgg import load_pretrained  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"
WORK = LAB / "artifacts/work"
CKPT = Path(os.environ.get("LAB1_CKPT", WORK / "vgg.cifar.pretrained.pth"))
EPOCHS = 5
SPARSITIES = np.arange(0.4, 1.0, 0.1)
CHANNEL_RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5)
MiB = 8 * 1024 * 1024


def latency_ms(model, threads: int, repeats: int = 200) -> float:
    model = model.cpu().eval()
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        r = time_calls(lambda: model(x), warmup=20, repeats=repeats)
    return r.p50_s * 1e3


def plot_hist(model, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers = prunable(model)
    fig, axes = plt.subplots(3, 3, figsize=(10, 7))
    for ax, (name, p) in zip(axes.ravel(), layers):
        w = p.detach().cpu().reshape(-1).numpy()
        ax.hist(w, bins=80, color="#2c5f8a")
        ax.set_title(f"{name.replace('.weight', '')}  ({w.size:,})", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sensitivity(sparsities, accs, numels, dense_acc, choice, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for name, row in accs.items():
        a.plot(sparsities, row, marker="o", ms=3, label=name.replace(".weight", ""))
    a.axhline(dense_acc, color="k", ls="--", lw=0.8)
    a.set_xlabel("sparsity of one layer (others dense)")
    a.set_ylabel("top-1 (%)")
    a.legend(fontsize=7, ncol=2)
    a.grid(alpha=0.3)
    names = list(numels)
    b.barh([n.replace(".weight", "") for n in names], [numels[n] for n in names], color="#bbbbbb", label="dense")
    b.barh([n.replace(".weight", "") for n in names], [numels[n] * (1 - choice[n]) for n in names],
           color="#2c5f8a", label="kept")
    b.set_xlabel("parameters")
    b.invert_yaxis()
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_channel(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    r = [x["ratio"] for x in rows]
    a.plot(r, [x["acc_naive"] for x in rows], marker="o", label="prune, no sorting")
    a.plot(r, [x["acc_sorted"] for x in rows], marker="o", label="sort, then prune")
    a.plot(r, [x["acc_finetuned"] for x in rows], marker="o", label="sort, prune, fine-tune 5 ep")
    a.set_xlabel("channel prune ratio")
    a.set_ylabel("top-1 (%)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    b.plot(r, [x["macs_ratio"] for x in rows], marker="o", label="MACs / dense")
    b.plot(r, [x["latency_ratio_1t"] for x in rows], marker="o", label="CPU latency / dense (1 thread)")
    b.plot(r, [(1 - x) ** 2 for x in r], ls="--", color="k", lw=0.8, label="(1 - ratio)^2")
    b.set_xlabel("channel prune ratio")
    b.legend(fontsize=8)
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def host() -> str:
    try:
        name = [ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")][0]
    except OSError:
        name = platform.processor()
    return name


def run_latency_only() -> None:
    """Re-time the saved models on this machine's CPU (no data or GPU needed)."""
    models = {"dense": torch.load(WORK / "dense.pt", weights_only=False),
              "fine_grained": torch.load(WORK / "fine_grained.pt", weights_only=False)}
    for r in CHANNEL_RATIOS:
        models[f"channel_{r}"] = torch.load(WORK / f"channel_{r}.pt", weights_only=False)
    out = {"host": host(), "torch": torch.__version__}
    for name, m in models.items():
        for t in (1, 4):
            out[f"{name}_ms_{t}t"] = latency_ms(m, t)
    tag = os.environ.get("LAB1_HOST_TAG", platform.node().lower())
    write_json(RAW / f"latency_{tag}.json", out)
    print(json.dumps(out, indent=2))


def main() -> None:
    if "--latency" in sys.argv:
        run_latency_only()
        return
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    xtr_u8, ytr_np, xte_u8, yte_np = load_cifar10()
    xtr, ytr = to_tensors(xtr_u8, ytr_np, dev)
    xte, yte = to_tensors(xte_u8, yte_np, dev)
    ev = lambda m: evaluate(m, xte, yte)  # noqa: E731

    dense = load_pretrained(CKPT, dev)
    dense_acc = ev(dense)
    dense_numel = get_num_parameters(dense)
    numels = {n: p.numel() for n, p in prunable(dense)}
    torch.save(dense.cpu(), WORK / "dense.pt")
    dense.to(dev)
    summary = {"host": host(), "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
               "torch": torch.__version__, "dense_acc": dense_acc, "dense_params": dense_numel,
               "dense_size_mib": get_model_size(dense) / MiB}
    plot_hist(dense, FIG / "weight_hist.png")

    # Q3: a 5x5 tensor keeps exactly 10 nonzeros at sparsity 15/25.
    sign = torch.where(torch.arange(25) % 2 == 0, 1.0, -1.0)
    t = (torch.arange(1, 26, dtype=torch.float32) * sign).reshape(5, 5)
    fine_grained_prune(t, 15 / 25)
    summary["q3_target_sparsity"] = 15 / 25
    summary["q3_nonzeros"] = int(t.count_nonzero())

    # Q4: one-layer sensitivity scan.
    sparsities, accs = sensitivity_scan(dense, ev, SPARSITIES)
    write_csv(RAW / "sensitivity.csv", [{"layer": n, "numel": numels[n], **{f"s{s}": a for s, a in zip(sparsities, row)}}
                                         for n, row in accs.items()])

    # Q5: per-layer sparsities under a 25% size budget, then fine-tune with masks.
    choice, tol = choose_sparsities(sparsities, accs, numels, dense_numel, dense_acc, budget=0.25)
    plot_sensitivity(sparsities, accs, numels, dense_acc, choice, FIG / "sensitivity.png")
    fg = load_pretrained(CKPT, dev)
    pruner = FineGrainedPruner(fg, choice)
    acc_pruned = ev(fg)
    log = finetune(fg, xtr, ytr, xte, yte, epochs=EPOCHS, after_step=pruner.apply)
    best = max(r["acc"] for r in log)
    sparse_bits = get_model_size(fg, count_nonzero_only=True)
    summary.update({
        "fg_tolerance_points": tol, "fg_sparsity": choice,
        "fg_kept_fraction": kept_fraction(choice, numels, dense_numel),
        "fg_size_ratio": sparse_bits / get_model_size(dense),
        "fg_size_mib": sparse_bits / MiB,
        "fg_layer_sparsity_measured": {n: get_sparsity(p) for n, p in prunable(fg)},
        "fg_acc_before_finetune": acc_pruned, "fg_acc_finetuned_last": log[-1]["acc"], "fg_acc_finetuned_best": best,
        "fg_finetune_log": log,
    })
    torch.save(fg.cpu(), WORK / "fine_grained.pt")

    # Q6-Q8: uniform channel pruning, sorting by Frobenius importance, fine-tuning, MACs and latency.
    x1 = xte[:1]
    dense_macs = count_macs(dense, x1)
    dense_lat = {t: latency_ms(dense, t) for t in (1, 4)}
    dense.to(dev)
    summary.update({"dense_macs": dense_macs, "dense_ms_1t": dense_lat[1], "dense_ms_4t": dense_lat[4],
                    "fg_ms_1t": latency_ms(fg, 1), "fg_ms_4t": latency_ms(fg, 4),
                    "fg_macs_dense_kernel": count_macs(fg.cpu(), x1.cpu())})
    sorted_model = apply_channel_sorting(dense)
    summary["sorting_preserves_acc"] = ev(sorted_model)
    rows = []
    with JsonlWriter(RAW / "runs.jsonl") as w:
        for r in CHANNEL_RATIOS:
            naive = channel_prune(dense, r)
            pruned = channel_prune(sorted_model, r)
            row = {"ratio": r, "acc_naive": ev(naive), "acc_sorted": ev(pruned)}
            ft = finetune(pruned, xtr, ytr, xte, yte, epochs=EPOCHS)
            row["acc_finetuned"] = max(e["acc"] for e in ft)
            row["params"] = get_num_parameters(pruned)
            row["macs"] = count_macs(pruned, x1)
            row["macs_ratio"] = row["macs"] / dense_macs
            for t in (1, 4):
                row[f"ms_{t}t"] = latency_ms(pruned, t)
                row[f"latency_ratio_{t}t"] = row[f"ms_{t}t"] / dense_lat[t]
            torch.save(pruned.cpu(), WORK / f"channel_{r}.pt")
            rows.append(row)
            w.write({"kind": "channel", **row, "finetune_log": ft})
            print(row, flush=True)
        w.write({"kind": "fine_grained", **{k: v for k, v in summary.items() if k.startswith("fg_")}})
    write_csv(RAW / "channel.csv", rows)
    plot_channel(rows, FIG / "channel_pruning.png")
    for row in rows:
        tag = str(row["ratio"]).replace(".", "")
        for k in ("acc_naive", "acc_sorted", "acc_finetuned", "macs_ratio", "ms_1t", "ms_4t",
                  "latency_ratio_1t", "latency_ratio_4t"):
            summary[f"ch{tag}_{k}"] = row[k]
    write_json(RAW / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, (dict, list))}, indent=2))


if __name__ == "__main__":
    main()
