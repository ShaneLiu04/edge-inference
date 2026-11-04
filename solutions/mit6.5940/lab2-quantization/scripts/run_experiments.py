"""Post-training quantization and QAT of the trained MicroMobileNetV2 on CIFAR-10. CPU only.

  python scripts/train.py                      # once, about 45 min on 4 cores
  python scripts/run_experiments.py            # every step below
  python scripts/run_experiments.py --only calib qat

Every step writes one file under artifacts/raw; figures go to artifacts/figures.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[2]
sys.path[:0] = [str(LAB / "code"), str(REPO)]

from edge_harness.writer import write_csv, write_json  # noqa: E402
from edgequant.data import load_cifar10, normalize  # noqa: E402
from edgequant.model import MicroMobileNetV2, count_macs, count_params  # noqa: E402
from edgequant.qsim import (  # noqa: E402
    QuantConfig,
    QuantSim,
    accuracy,
    act_points,
    execution_order,
    fold_model,
    is_depthwise,
    point_outputs,
    weight_layers,
)
from edgequant.quant import fake_quant_weight, sqnr_db  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"
CKPT = LAB / "checkpoints/micronet.pt"
CALIBRATORS = ("minmax", "percentile", "mse", "kl", "kl_nonzero")


class Ctx:
    def __init__(self, calib_n: int, eval_n: int, ckpt: Path = CKPT):
        xtr, ytr, xte, yte = load_cifar10()
        perm = np.random.default_rng(0).permutation(len(xtr))
        self.xtrain = torch.from_numpy(normalize(xtr))
        self.ytrain = torch.from_numpy(ytr)
        self.calib_pool = self.xtrain[perm[:8192]]
        self.calib = self.calib_pool[:calib_n]
        self.xte = torch.from_numpy(normalize(xte[:eval_n]))
        self.yte = torch.from_numpy(yte[:eval_n])
        self.model = MicroMobileNetV2()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        self.model.eval()
        self.folded = fold_model(self.model)
        self.layers = list(weight_layers(self.folded))
        self.points = ["input", *execution_order(self.folded, self.xte[:1], list(act_points(self.folded)))]

    def acc(self, model) -> float:
        return accuracy(model, self.xte, self.yte)

    def ptq(self, cfg: QuantConfig, calib: torch.Tensor | None = None) -> QuantSim:
        sim = QuantSim(self.folded, cfg)
        if cfg.a_bits is not None or any(v is not None for v in cfg.a_overrides.values()):
            sim.calibrate(self.calib if calib is None else calib)
        return sim


def weight_bytes(ctx: Ctx, bits_of: dict[str, int | None]) -> float:
    total = 0.0
    for name, m in weight_layers(ctx.folded).items():
        b = bits_of.get(name, 32)
        total += m.weight.numel() * (32 if b is None else b) / 8
    return total


# ---------------------------------------------------------------- steps

def step_baseline(ctx: Ctx) -> None:
    fp32 = ctx.acc(ctx.model)
    folded = ctx.acc(ctx.folded)
    with torch.no_grad():
        a = ctx.model(ctx.xte[:500])
        b = ctx.folded(ctx.xte[:500])
    out = {"fp32_acc": fp32, "folded_acc": folded, "fold_logit_max_abs_diff": float((a - b).abs().max()),
           "params": count_params(ctx.model), "macs": count_macs(ctx.model), "eval_images": len(ctx.xte),
           "weight_layers": len(ctx.layers), "act_points": len(ctx.points)}
    print(out)
    write_json(RAW / "baseline.json", out)


def step_ranges(ctx: Ctx) -> None:
    """Per-layer spread of per-channel weight ranges after the fold, and what per-tensor int8 does to it."""
    rows = []
    for name, m in weight_layers(ctx.folded).items():
        w = m.weight.detach()
        amax = w.reshape(w.shape[0], -1).abs().amax(1)
        q = fake_quant_weight(w, 8, per_channel=False)
        err = ((q - w).reshape(w.shape[0], -1).norm(dim=1) / w.reshape(w.shape[0], -1).norm(dim=1).clamp_min(1e-12))
        rows.append({"layer": name, "kind": "depthwise" if is_depthwise(m) else type(m).__name__.lower(),
                     "out_channels": int(w.shape[0]), "amax_ratio": float(amax.max() / amax.min().clamp_min(1e-12)),
                     "per_tensor_w8_worst_channel_rel_err": float(err.max()),
                     "per_tensor_w8_channels_above_10pct": int((err > 0.10).sum()),
                     "per_tensor_w8_channels_zeroed": int((q.reshape(w.shape[0], -1).abs().amax(1) == 0).sum())})
    write_csv(RAW / "weight_ranges.csv", rows)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = ["C3" if r["kind"] == "depthwise" else "C0" for r in rows]
    ax.bar(range(len(rows)), [r["amax_ratio"] for r in rows], color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)), [r["layer"] for r in rows], rotation=90, fontsize=6)
    ax.set_ylabel("max / min channel |w| after fold")
    ax.set_title("Per-channel weight range spread (red: depthwise)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "weight_ranges.png", dpi=150)
    plt.close(fig)


def step_weight_only(ctx: Ctx) -> None:
    rows = []
    for bits in (8, 6, 4):
        for pc in (True, False):
            acc = ctx.acc(ctx.ptq(QuantConfig(w_bits=bits, w_per_channel=pc, a_bits=None)))
            rows.append({"w_bits": bits, "granularity": "per_channel" if pc else "per_tensor", "acc": acc})
        acc = ctx.acc(ctx.ptq(QuantConfig(w_bits=bits, w_per_channel=True, a_bits=None, dw_per_channel=False)))
        rows.append({"w_bits": bits, "granularity": "per_channel_but_depthwise_per_tensor", "acc": acc})
        print(rows[-3:])
    write_csv(RAW / "weight_only.csv", rows)


def step_calib(ctx: Ctx) -> None:
    rows = []
    for a_bits in (8, 6):
        for calib in CALIBRATORS:
            t0 = time.perf_counter()
            sim = ctx.ptq(QuantConfig(w_bits=8, a_bits=a_bits, calib=calib))
            rows.append({"a_bits": a_bits, "calib": calib, "acc": ctx.acc(sim), "calib_s": time.perf_counter() - t0})
            print(rows[-1])
    write_csv(RAW / "calibrators.csv", rows)


def step_calib_size(ctx: Ctx) -> None:
    rows = []
    for calib in ("minmax", "mse"):
        for n in (8, 32, 128, 512, 2048):
            accs = []
            for seed in range(3):
                idx = torch.from_numpy(np.random.default_rng(100 + seed).choice(len(ctx.calib_pool), n, replace=False))
                accs.append(ctx.acc(ctx.ptq(QuantConfig(w_bits=8, a_bits=6, calib=calib), ctx.calib_pool[idx])))
            rows.append({"calib": calib, "a_bits": 6, "n": n, "acc_mean": float(np.mean(accs)),
                         "acc_std": float(np.std(accs)), "accs": accs})
            print(rows[-1])
    write_csv(RAW / "calib_size.csv", [{k: v for k, v in r.items() if k != "accs"} for r in rows])


def step_sensitivity(ctx: Ctx) -> None:
    """One layer at W4 (or one point at A4), everything else in float."""
    base = ctx.acc(ctx.folded)
    rows = []
    for name, m in weight_layers(ctx.folded).items():
        overrides = {n: None for n in ctx.layers}
        overrides[name] = 4
        acc = ctx.acc(ctx.ptq(QuantConfig(w_bits=None, a_bits=None, w_overrides=overrides)))
        w = m.weight.detach()
        rows.append({"kind": "weight_w4", "name": name, "depthwise": is_depthwise(m), "params": int(w.numel()),
                     "acc": acc, "drop": base - acc,
                     "w_sqnr_db": sqnr_db(w.numpy(), fake_quant_weight(w, 4, True).numpy())})
    for point in ctx.points:
        overrides = {p: None for p in ctx.points}
        overrides[point] = 4
        acc = ctx.acc(ctx.ptq(QuantConfig(w_bits=None, a_bits=None, a_overrides=overrides, calib="mse")))
        rows.append({"kind": "act_a4", "name": point, "depthwise": None, "params": None, "acc": acc,
                     "drop": base - acc, "w_sqnr_db": None})
    write_csv(RAW / "sensitivity.csv", rows)
    for r in sorted(rows, key=lambda r: -r["drop"])[:8]:
        print(r)


def step_kl_points(ctx: Ctx) -> None:
    """Why KL fails here: its threshold next to MSE's at every point, and the accuracy with only that point at A8."""
    sims = {}
    for calib in ("mse", "kl", "kl_nonzero"):
        sims[calib] = QuantSim(ctx.folded, QuantConfig(w_bits=None, a_bits=8, calib=calib))
        sims[calib].calibrate(ctx.calib)
    base = ctx.acc(ctx.folded)
    rows = []
    for point in ctx.points:
        sample = sims["kl"].ranges[point]
        overrides = {p: None for p in ctx.points}
        overrides[point] = 8
        only = QuantSim(ctx.folded, QuantConfig(w_bits=None, a_bits=None, a_overrides=overrides, calib="kl"))
        only.ranges = {point: sample}
        only.install()
        acc = ctx.acc(only)
        hi = {c: float(s.ranges[point][1]) for c, s in sims.items()}
        rows.append({"point": point, "nonneg": float(sims["mse"].ranges[point][0]) >= 0,
                     "mse_hi": hi["mse"], "kl_hi": hi["kl"], "kl_nonzero_hi": hi["kl_nonzero"],
                     "kl_over_mse": hi["kl"] / max(hi["mse"], 1e-9),
                     "kl_nonzero_over_mse": hi["kl_nonzero"] / max(hi["mse"], 1e-9),
                     "only_this_point_kl_acc": acc, "only_this_point_kl_drop": base - acc})
        print(rows[-1])
    write_csv(RAW / "kl_points.csv", rows)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.6))
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r["kl_over_mse"] for r in rows], 0.4, label="KL on |x| (TensorRT style)",
           color=["tab:red" if r["nonneg"] else "tab:grey" for r in rows])
    ax.bar(xs + 0.2, [r["kl_nonzero_over_mse"] for r in rows], 0.4, label="KL on non-zero values", color="tab:blue")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(xs, [r["point"] for r in rows], rotation=90, fontsize=6)
    ax.set_ylabel("upper threshold / MSE threshold")
    ax.set_title("KL clips ReLU6 outputs (red) at about a quarter of the MSE range")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "kl_points.png", dpi=150)
    plt.close(fig)


def step_localize(ctx: Ctx) -> None:
    """SQNR of every activation point against the folded fp32 model, along the depth of the network."""
    x = ctx.xte[:256]
    names = [p for p in ctx.points if p != "input"]
    ref = point_outputs(ctx.folded, x, names)
    configs = {
        "W8A8 per-channel minmax": QuantConfig(w_bits=8, a_bits=8),
        "W8A8 per-tensor minmax": QuantConfig(w_bits=8, w_per_channel=False, a_bits=8),
        "W4A8 per-channel mse": QuantConfig(w_bits=4, a_bits=8, calib="mse"),
        "W8A4 per-channel mse": QuantConfig(w_bits=8, a_bits=4, calib="mse"),
    }
    rows = []
    for label, cfg in configs.items():
        sim = ctx.ptq(cfg)
        got = point_outputs(sim.model, x, names)
        for i, n in enumerate([*names, "logits"]):
            rows.append({"config": label, "depth": i, "point": n,
                         "sqnr_db": sqnr_db(ref[n].numpy(), got[n].numpy())})
    write_csv(RAW / "localize.csv", rows)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.8))
    for label in configs:
        sel = [r for r in rows if r["config"] == label]
        ax.plot([r["depth"] for r in sel], [r["sqnr_db"] for r in sel], marker=".", label=label)
    pts = [r["point"] for r in rows if r["config"] == next(iter(configs))]
    ax.set_xticks(range(len(pts)), pts, rotation=90, fontsize=6)
    ax.set_ylabel("SQNR vs fp32 (dB)")
    ax.set_title("Where the error enters: per-point SQNR along the network")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "localize.png", dpi=150)
    plt.close(fig)


def step_mixed(ctx: Ctx) -> None:
    """Start at W4A8, promote layers to W8 in order of W4 sensitivity; accuracy against weight bytes."""
    import csv

    sens = [r for r in csv.DictReader(open(RAW / "sensitivity.csv", encoding="utf-8")) if r["kind"] == "weight_w4"]
    order = [r["name"] for r in sorted(sens, key=lambda r: -float(r["drop"]))]
    by_size = [n for n, _ in sorted(weight_layers(ctx.folded).items(), key=lambda kv: -kv[1].weight.numel())]
    rows = []
    for policy, seq in (("by_sensitivity", order), ("by_size", by_size)):
        bits = {n: 4 for n in ctx.layers}
        for k in range(len(seq) + 1):
            if k:
                bits[seq[k - 1]] = 8
            if k > 8 and k < len(seq):
                continue  # the curve is flat after the first few promotions; keep the end point
            acc = ctx.acc(ctx.ptq(QuantConfig(w_bits=4, a_bits=8, calib="mse", w_overrides=dict(bits))))
            rows.append({"policy": policy, "promoted": k, "weight_kib": weight_bytes(ctx, bits) / 1024, "acc": acc})
            print(rows[-1])
    write_csv(RAW / "mixed_precision.csv", rows)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for policy in ("by_sensitivity", "by_size"):
        sel = [r for r in rows if r["policy"] == policy]
        ax.plot([r["weight_kib"] for r in sel], [r["acc"] for r in sel], marker="o", label=policy)
    ax.axhline(ctx.acc(ctx.folded), color="k", ls="--", lw=0.8, label="fp32")
    ax.set_xlabel("weight storage (KiB)")
    ax.set_ylabel("test accuracy")
    ax.set_title("W4 -> W8 promotion, A8 throughout")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "mixed_precision.png", dpi=150)
    plt.close(fig)


def step_qat(ctx: Ctx, epochs: int, w_bits: int = 4) -> None:
    torch.manual_seed(0)
    cfg = QuantConfig(w_bits=w_bits, a_bits=8, calib="mse")
    ptq_acc = ctx.acc(ctx.ptq(cfg))
    sim = QuantSim(ctx.folded, cfg, ste=True)
    sim.calibrate(ctx.calib)
    params = [p for p in sim.model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=2e-3, momentum=0.9, weight_decay=0)
    batch = 128
    steps = len(ctx.xtrain) // batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    gen = torch.Generator().manual_seed(0)
    sys.path.insert(0, str(LAB / "scripts"))
    from train import augment

    rows = [{"epoch": 0, "acc": ptq_acc, "loss": None, "epoch_s": 0.0}]
    print(rows[-1])
    for epoch in range(epochs):
        sim.model.train()
        perm = torch.randperm(len(ctx.xtrain), generator=gen)
        total, t0 = 0.0, time.perf_counter()
        for s in range(steps):
            idx = perm[s * batch:(s + 1) * batch]
            loss = F.cross_entropy(sim.model(augment(ctx.xtrain[idx], gen)), ctx.ytrain[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        rows.append({"epoch": epoch + 1, "acc": ctx.acc(sim), "loss": total / steps, "epoch_s": time.perf_counter() - t0})
        print(rows[-1])
    write_csv(RAW / f"qat_w{w_bits}a8.csv", rows)


def step_ort(ctx: Ctx) -> None:
    """Export the folded fp32 graph, quantize with onnxruntime.quantization, evaluate and time both."""
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    work = RAW.parent / "work"  # per --out, so parallel runs on different checkpoints do not share models
    work.mkdir(parents=True, exist_ok=True)
    fp32_path = work / "micronet_folded.onnx"
    torch.onnx.export(ctx.folded, ctx.xte[:1], str(fp32_path), input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    pre_path = work / "micronet_folded_pre.onnx"
    quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=True)

    class Reader(CalibrationDataReader):
        def __init__(self, x: torch.Tensor, batch: int = 64):
            self.batches = [x[i:i + batch].numpy() for i in range(0, len(x), batch)]
            self.i = 0

        def get_next(self):
            if self.i >= len(self.batches):
                return None
            self.i += 1
            return {"x": self.batches[self.i - 1]}

        def rewind(self):
            self.i = 0

    def evaluate(path: Path) -> tuple[float, float]:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        correct = 0
        xs = ctx.xte.numpy()
        for i in range(0, len(xs), 500):
            correct += int((sess.run(None, {"x": xs[i:i + 500]})[0].argmax(1) == ctx.yte.numpy()[i:i + 500]).sum())
        so1 = ort.SessionOptions()
        so1.intra_op_num_threads = 1
        s1 = ort.InferenceSession(str(path), so1, providers=["CPUExecutionProvider"])
        one = xs[:1]
        for _ in range(50):
            s1.run(None, {"x": one})
        # Median of 7 block medians: one slow block (a background task, a frequency dip) does not move it.
        blocks = []
        for _ in range(7):
            ts = []
            for _ in range(200):
                t0 = time.perf_counter()
                s1.run(None, {"x": one})
                ts.append(time.perf_counter() - t0)
            blocks.append(float(np.median(ts)))
        return correct / len(xs), float(np.median(blocks))

    rows = []
    acc, lat = evaluate(fp32_path)
    rows.append({"model": "fp32 folded", "acc": acc, "latency_us_b1_t1": lat * 1e6,
                 "file_kib": fp32_path.stat().st_size / 1024})
    print(rows[-1])
    # ORT's Entropy defaults to num_bins == num_quantized_bins == 128, which leaves a single candidate
    # threshold (the full range): it is MinMax under another name. quantize_static does not forward
    # num_bins, so the 2048-bin row runs the calibrator directly and hands its ranges over as a cache.
    from onnxruntime.quantization.calibrate import create_calibrator, save_tensors_data
    from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry

    methods = {"MinMax": (CalibrationMethod.MinMax, None), "Entropy": (CalibrationMethod.Entropy, None),
               "Entropy 2048 bins": (CalibrationMethod.Entropy, {"num_bins": 2048}),
               "Percentile": (CalibrationMethod.Percentile, None)}
    for label, (method, calib_opts) in methods.items():
        out = work / f"micronet_int8_{label.lower().replace(' ', '_')}.onnx"
        cache = None
        if calib_opts:
            cache = work / f"calib_{label.lower().replace(' ', '_')}.json"
            cache.unlink(missing_ok=True)
            calibrator = create_calibrator(pre_path, list(set(QLinearOpsRegistry) | set(QDQRegistry)),
                                           augmented_model_path=str(work / "augmented.onnx"),
                                           calibrate_method=method, extra_options=calib_opts)
            calibrator.collect_data(Reader(ctx.calib[:512]))
            save_tensors_data(calibrator.compute_data(), cache)
        quantize_static(str(pre_path), str(out), None if cache else Reader(ctx.calib[:512]),
                        quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8, calibrate_method=method,
                        calibration_cache_path=str(cache) if cache else None)
        acc, lat = evaluate(out)
        rows.append({"model": f"int8 QDQ {label}", "acc": acc, "latency_us_b1_t1": lat * 1e6,
                     "file_kib": out.stat().st_size / 1024})
        print(rows[-1])
    write_csv(RAW / "ort_ptq.csv", rows)


def summarize() -> None:
    """Flatten the headline numbers for claims.json."""
    import csv
    import json

    out: dict = {}
    if (RAW / "baseline.json").exists():
        out.update(json.loads((RAW / "baseline.json").read_text(encoding="utf-8")))

    def table(name):
        p = RAW / name
        return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []

    for r in table("weight_only.csv"):
        out[f"w{r['w_bits']}_{r['granularity']}"] = float(r["acc"])
    for r in table("calibrators.csv"):
        out[f"w8a{r['a_bits']}_{r['calib']}"] = float(r["acc"])
    for r in table("ort_ptq.csv"):
        key = r["model"].replace(" ", "_").lower()
        out[f"ort_{key}_acc"] = float(r["acc"])
        out[f"ort_{key}_latency_us"] = float(r["latency_us_b1_t1"])
        out[f"ort_{key}_kib"] = float(r["file_kib"])
    qat = table("qat_w4a8.csv")
    if qat:
        out["qat_w4a8_epoch0"] = float(qat[0]["acc"])
        out["qat_w4a8_final"] = float(qat[-1]["acc"])
    klp = table("kl_points.csv")
    if klp:
        relu = [float(r["kl_over_mse"]) for r in klp if r["nonneg"] == "True" and r["point"] not in ("pool",)]
        out["kl_over_mse_relu_median"] = float(np.median(relu))
        out["kl_nonzero_over_mse_relu_median"] = float(np.median(
            [float(r["kl_nonzero_over_mse"]) for r in klp if r["nonneg"] == "True" and r["point"] != "pool"]))
        out["kl_worst_single_point_drop"] = max(float(r["only_this_point_kl_drop"]) for r in klp)
    for r in table("mixed_precision.csv"):
        out[f"mixed_{r['policy']}_{r['promoted']}_acc"] = float(r["acc"])
        out[f"mixed_{r['policy']}_{r['promoted']}_kib"] = float(r["weight_kib"])
    sens = [r for r in table("sensitivity.csv") if r["kind"] == "weight_w4"]
    if sens:
        worst = max(sens, key=lambda r: float(r["drop"]))
        out["most_sensitive_w4_layer"] = worst["name"]
        out["most_sensitive_w4_drop"] = float(worst["drop"])
    write_json(RAW / "summary.json", out)


STEPS = ["baseline", "ranges", "weight_only", "calib", "kl_points", "calib_size", "sensitivity", "localize", "mixed",
         "qat", "ort"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=STEPS)
    ap.add_argument("--calib-n", type=int, default=512)
    ap.add_argument("--eval-n", type=int, default=10000)
    ap.add_argument("--qat-epochs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--out", type=Path, default=None, help="artifacts directory (default: the lab's)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    global RAW, FIG
    if args.out is not None:
        RAW, FIG = args.out / "raw", args.out / "figures"
    if not args.ckpt.exists():
        raise SystemExit(f"missing {args.ckpt}; run scripts/train.py first")
    ctx = Ctx(args.calib_n, args.eval_n, args.ckpt)
    fns = {"baseline": step_baseline, "ranges": step_ranges, "weight_only": step_weight_only, "calib": step_calib,
           "kl_points": step_kl_points,
           "calib_size": step_calib_size, "sensitivity": step_sensitivity, "localize": step_localize,
           "mixed": step_mixed, "qat": lambda c: step_qat(c, args.qat_epochs), "ort": step_ort}
    for name in args.only:
        t0 = time.perf_counter()
        fns[name](ctx)
        print(f"== {name} {time.perf_counter() - t0:.0f}s", flush=True)
    summarize()


if __name__ == "__main__":
    main()
