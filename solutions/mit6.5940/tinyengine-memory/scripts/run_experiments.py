"""Activation-arena experiments for int8 MobileNetV2-class models. CPU only, about a minute.

  python scripts/run_experiments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[2]
sys.path[:0] = [str(LAB / "code"), str(LAB.parent / "lab2-quantization" / "code"), str(REPO)]

import torch  # noqa: E402
import torchvision  # noqa: E402
from edge_harness import devices  # noqa: E402
from edge_harness.writer import write_csv, write_json  # noqa: E402
from edgequant.model import MicroMobileNetV2  # noqa: E402
from memplan.graph import inplace_depthwise, live_profile, trace  # noqa: E402
from memplan.patch import patch_plan  # noqa: E402
from memplan.planners import PLANNERS, arena_size, lower_bound, validate  # noqa: E402
from torch import nn  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"

# SRAM budgets in bytes. 320 KB is the STM32F746 that MCUNet reports on.
BUDGETS = {"nrf52840 (256 KB)": devices.get("nrf52840").sram_bytes, "stm32f746 (320 KB)": 320 * 1024,
           "stm32h747 (1 MB)": devices.get("stm32h747").sram_bytes}


def models():
    """(label, model, features, head, resolutions)."""
    micro = MicroMobileNetV2().eval()
    yield "micro-mbv2 (this repo)", micro, micro.features, nn.Sequential(micro.pool, nn.Flatten(), micro.fc), (32, 64, 96)
    for w in (0.35, 1.0):
        tv = torchvision.models.mobilenet_v2(weights=None, width_mult=w).eval()
        head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), tv.classifier)
        yield f"mobilenet_v2 w{w}", tv, tv.features, head, (96, 128, 160, 224)


def flash_bytes(model: nn.Module) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if "bn" not in n and p.dim() > 1) + \
        4 * sum(p.numel() for p in model.parameters() if p.dim() == 1)


def step_planners() -> None:
    rows = []
    for label, model, *_rest, resolutions in models():
        for res in resolutions:
            ts = trace(model, (1, 3, res, res))
            row = {"model": label, "resolution": res, "buffers": len(ts), "lower_bound": lower_bound(ts),
                   "flash_bytes_int8": flash_bytes(model)}
            for name, fn in PLANNERS.items():
                off = fn(ts)
                validate(ts, off)
                row[name] = arena_size(ts, off)
            ip = inplace_depthwise(ts)
            off = PLANNERS["greedy_by_size"](ip)
            validate(ip, off)
            row["greedy_inplace_dw"] = arena_size(ip, off)
            row["lower_bound_inplace_dw"] = lower_bound(ip)
            row["first_fit_gap"] = row["first_fit"] / row["lower_bound"] - 1
            row["greedy_gap"] = row["greedy_by_size"] / row["lower_bound"] - 1
            row["inplace_saving"] = 1 - row["greedy_inplace_dw"] / row["greedy_by_size"]
            for b, cap in BUDGETS.items():
                row[f"fits {b}"] = row["greedy_inplace_dw"] <= cap
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if "fits" not in k})
    write_csv(RAW / "planners.csv", rows)


def step_profile() -> None:
    import matplotlib.pyplot as plt

    tv = torchvision.models.mobilenet_v2(weights=None, width_mult=0.35).eval()
    ts = trace(tv, (1, 3, 160, 160))
    prof = live_profile(ts)
    ip = live_profile(inplace_depthwise(ts))
    rows = [{"step": i, "live_bytes": a, "live_bytes_inplace_dw": b} for i, (a, b) in enumerate(zip(prof, ip))]
    write_csv(RAW / "profile_mbv2_w035_160.csv", rows)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot([r["live_bytes"] / 1024 for r in rows], label="live activations")
    ax.plot([r["live_bytes_inplace_dw"] / 1024 for r in rows], label="with in-place depthwise")
    for b, cap in BUDGETS.items():
        ax.axhline(cap / 1024, ls="--", lw=0.8, color="grey")
        ax.text(len(rows) * 0.7, cap / 1024 * 1.03, b, fontsize=7, color="grey")
    ax.set_xlabel("op index")
    ax.set_ylabel("KiB live (int8)")
    ax.set_title("MobileNetV2-0.35 @160: the peak is in the first stages")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "memory_profile.png", dpi=150)
    plt.close(fig)

    # Arena layout of the small model: each buffer as a rectangle in (time, offset).
    micro = MicroMobileNetV2().eval()
    ts = trace(micro, (1, 3, 32, 32))
    off = PLANNERS["greedy_by_size"](ts)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for t in ts:
        ax.add_patch(plt.Rectangle((t.birth - 0.4, off[t.name] / 1024), t.death - t.birth + 0.8, t.size / 1024,
                                   alpha=0.6, ec="k", lw=0.3))
    ax.set_xlim(-1, max(t.death for t in ts) + 1)
    ax.set_ylim(0, arena_size(ts, off) / 1024 * 1.05)
    ax.axhline(lower_bound(ts) / 1024, color="C3", ls="--", lw=0.8, label="lower bound (max live)")
    ax.set_xlabel("op index")
    ax.set_ylabel("arena offset (KiB)")
    ax.set_title("Greedy-by-size arena for micro-MobileNetV2 @32, int8")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "arena_layout.png", dpi=150)
    plt.close(fig)


def step_patch() -> None:
    rows = []
    for label, _model, features, head, resolutions in models():
        if label.startswith("micro"):
            continue
        res = 160 if "0.35" in label else 224
        for cut in (2, 4, 6, 8):
            for p in (1, 2, 3, 4):
                r = patch_plan(features, head, res, cut, p)
                rows.append({"model": label, "resolution": res, **r.__dict__})
                print(rows[-1])
    write_csv(RAW / "patch.csv", rows)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label in dict.fromkeys(r["model"] for r in rows):
        for cut in (4, 6):
            sel = [r for r in rows if r["model"] == label and r["cut"] == cut]
            ax.plot([r["net_mac_overhead"] for r in sel], [r["peak"] / 1024 for r in sel], marker="o",
                    label=f"{label} @{sel[0]['resolution']}, cut {cut}")
            for r in sel:
                ax.annotate(f"{r['patches']}x{r['patches']}", (r["net_mac_overhead"], r["peak"] / 1024), fontsize=6)
    for b, cap in BUDGETS.items():
        ax.axhline(cap / 1024, ls="--", lw=0.8, color="grey")
        ax.text(1.0, cap / 1024 * 1.03, b, fontsize=7, color="grey")
    ax.set_yscale("log")
    ax.set_xlabel("whole-network MACs relative to whole-image execution")
    ax.set_ylabel("peak activation memory (KiB, int8)")
    ax.set_title("Patch-based inference: memory for recomputation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "patch_tradeoff.png", dpi=150)
    plt.close(fig)


def summarize() -> None:
    import csv

    out = {}
    for r in csv.DictReader(open(RAW / "planners.csv", encoding="utf-8")):
        key = f"{r['model']}@{r['resolution']}"
        for k in ("lower_bound", "naive", "first_fit", "greedy_by_size", "greedy_inplace_dw"):
            out[f"{key}:{k}"] = int(r[k])
    for r in csv.DictReader(open(RAW / "patch.csv", encoding="utf-8")):
        out[f"{r['model']}@{r['resolution']}:cut{r['cut']}:p{r['patches']}:peak"] = int(r["peak"])
        out[f"{r['model']}@{r['resolution']}:cut{r['cut']}:p{r['patches']}:mac_overhead"] = float(r["mac_overhead"])
        out[f"{r['model']}@{r['resolution']}:cut{r['cut']}:p{r['patches']}:net_mac_overhead"] = float(r["net_mac_overhead"])
    write_json(RAW / "summary.json", out)


if __name__ == "__main__":
    torch.set_num_threads(2)
    step_planners()
    step_profile()
    step_patch()
    summarize()
