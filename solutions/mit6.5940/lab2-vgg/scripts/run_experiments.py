"""MIT 6.5940 Lab 2 (official questions) on the course VGG / CIFAR-10. GPU recommended.

  python scripts/run_experiments.py

Part 1: k-means weight sharing at 8/4/2 bits; QAT when the drop exceeds 0.5 points (at most 5 epochs).
Part 2: BN fusion, min/max calibration on 512 training images, integer-only int8 VGG; plus a 8/6/5/4-bit sweep.
Needs the course checkpoint at $LAB2_CKPT (default: Lab 1's artifacts/work copy) and .data/cifar10.npz.
"""

from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[2]
sys.path[:0] = [str(LAB / "code"), str(LAB.parent / "lab1-pruning" / "code"),
                str(LAB.parent / "lab2-quantization" / "code"), str(REPO)]

import torch  # noqa: E402
from edge_harness.writer import JsonlWriter, write_json  # noqa: E402
from edgequant.data import load_cifar10  # noqa: E402
from prune.fine import get_model_size  # noqa: E402
from prune.train import evaluate, finetune, to_tensors  # noqa: E402
from prune.vgg import load_pretrained  # noqa: E402
from vggquant.kmeans import KMeansQuantizer, kmeans_model_size_bits  # noqa: E402
from vggquant.linear import QuantizedVGG, calibrate, fuse_vgg, quantize_input  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"
CKPT = Path(os.environ.get("LAB2_CKPT", LAB.parent / "lab1-pruning/artifacts/work/vgg.cifar.pretrained.pth"))
MiB = 8 * 1024 * 1024
QAT_EPOCHS = 5
QAT_TOL = 0.5
LR = 0.01


@torch.no_grad()
def evaluate_int(qm: QuantizedVGG, x, y, batch: int = 500) -> float:
    correct = 0
    for i in range(0, len(x), batch):
        correct += int((qm(quantize_input(x[i:i + batch])).argmax(1) == y[i:i + batch]).sum())
    return 100.0 * correct / len(x)


def kmeans_part(dense, xtr, ytr, xte, yte, dense_acc, w) -> dict:
    out = {}
    for bits in (8, 4, 2):
        model = copy.deepcopy(dense)
        q = KMeansQuantizer(model, bits)
        acc0 = evaluate(model, xte, yte)
        log = []
        acc = acc0
        for ep in range(QAT_EPOCHS):
            if dense_acc - acc <= QAT_TOL:
                break
            lr = LR * 0.5 * (1 + math.cos(math.pi * ep / QAT_EPOCHS))
            row = finetune(model, xtr, ytr, xte, yte, epochs=1, lr=lr, seed=ep, cosine=False,
                           after_step=lambda m, q=q: q.apply(m, update=True))[0]
            acc = row["acc"]
            log.append({"epoch": ep + 1, "lr": lr, "acc": acc})
        size = kmeans_model_size_bits(model, bits) / MiB
        out[f"km{bits}"] = {"acc_before_qat": acc0, "acc_after_qat": acc, "qat_epochs": len(log),
                            "size_mib": size, "qat_log": log}
        w.write({"part": "kmeans", "bits": bits, "acc_before_qat": acc0, "acc_after_qat": acc,
                 "qat_epochs": len(log), "size_mib": size})
        print(f"kmeans {bits}b: {acc0:.2f} -> {acc:.2f} after {len(log)} QAT epochs, {size:.2f} MiB", flush=True)
    return out


def linear_part(dense, xtr, xte, yte, w) -> dict:
    fused = fuse_vgg(dense)
    out = {"fused_acc": evaluate(fused, xte, yte)}
    ranges = calibrate(fused, xtr[:512])
    out["calib_ranges"] = {k: list(v) for k, v in ranges.items()}
    for bits in (8, 6, 5, 4):
        qm = QuantizedVGG(fused, ranges, bits)
        acc = evaluate_int(qm, xte, yte)
        size = qm.size_bits() / MiB
        out[f"lin{bits}_acc"] = acc
        out[f"lin{bits}_size_mib"] = size
        w.write({"part": "linear", "bits": bits, "acc": acc, "size_mib": size})
        print(f"linear int{bits}: {acc:.2f}%  {size:.2f} MiB", flush=True)
    qm = QuantizedVGG(fused, ranges, 8)
    with torch.no_grad():
        xq = quantize_input(xte[:1])
        out["int8_dtypes_between_layers"] = str(qm(xq).dtype)
    return out


def plot(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    bits = [8, 4, 2]
    b0 = [summary["kmeans"][f"km{b}"]["acc_before_qat"] for b in bits]
    b1 = [summary["kmeans"][f"km{b}"]["acc_after_qat"] for b in bits]
    xs = range(len(bits))
    axes[0].bar([i - 0.2 for i in xs], b0, 0.4, label="k-means only", color="#9bb7d4")
    axes[0].bar([i + 0.2 for i in xs], b1, 0.4, label="+ QAT", color="#2c5f8a")
    axes[0].axhline(summary["dense_acc"], color="k", ls="--", lw=0.8, label="fp32")
    axes[0].set_xticks(list(xs), [f"{b}-bit" for b in bits])
    axes[0].set_ylabel("CIFAR-10 top-1 (%)")
    axes[0].set_title("k-means weight sharing")
    axes[0].legend(fontsize=7)
    lb = [8, 6, 5, 4]
    axes[1].plot(lb, [summary["linear"][f"lin{b}_acc"] for b in lb], "o-", color="#2c5f8a")
    axes[1].axhline(summary["dense_acc"], color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("weight and activation bits (integer-only)")
    axes[1].set_title("linear quantization, PTQ")
    axes[1].invert_xaxis()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "lab2_vgg.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xtr_u8, ytr_np, xte_u8, yte_np = load_cifar10()
    xtr, ytr = to_tensors(xtr_u8, ytr_np, dev)
    xte, yte = to_tensors(xte_u8, yte_np, dev)
    dense = load_pretrained(CKPT, dev)
    dense_acc = evaluate(dense, xte, yte)
    summary = {"device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", "torch": torch.__version__,
               "dense_acc": dense_acc, "dense_size_mib": get_model_size(dense) / MiB}
    with JsonlWriter(RAW / "runs.jsonl") as w:
        summary["kmeans"] = kmeans_part(dense, xtr, ytr, xte, yte, dense_acc, w)
        summary["linear"] = linear_part(dense, xtr, xte, yte, w)
    write_json(RAW / "summary.json", summary)
    plot(summary)
    print({k: v for k, v in summary.items() if k not in ("kmeans", "linear")})


if __name__ == "__main__":
    main()
