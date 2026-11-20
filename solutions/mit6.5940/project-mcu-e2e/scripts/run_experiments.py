"""Final project: Lab 3 subnets and the MCUNet VWW models in the four MLPerf Tiny columns.

  python scripts/run_experiments.py --stage prepare accuracy memory latency table

Inputs: $MCUNET_ROOT (course package, for the network classes and the supernet), $VWW_ROOT
(vww-s256/val), $ZOO_ROOT (mcunet-*_vww.{json,pth,tflite} from the MCUNet release) and the
subnets Lab 3 exported to lab3-nas/artifacts/work. Optional: $LITERT_PATH, a directory holding
ai_edge_litert, for the official int8 .tflite files.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
REPO = PROJ.parents[2]
LAB3 = PROJ.parent / "lab3-nas"
MCUNET_ROOT = Path(os.environ.get("MCUNET_ROOT", LAB3 / "artifacts/work/mcunetv2-dev-main"))
VWW_ROOT = Path(os.environ.get("VWW_ROOT", LAB3 / "artifacts/work/data/vww-s256/val"))
ZOO_ROOT = Path(os.environ.get("ZOO_ROOT", PROJ / "artifacts/work/zoo"))
LITERT_PATH = os.environ.get("LITERT_PATH")
sys.path[:0] = [str(PROJ / "code"), str(LAB3 / "code"), str(PROJ.parent / "tinyengine-memory" / "code"), str(REPO),
                str(MCUNET_ROOT)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from edge_harness.writer import JsonlWriter, write_csv, write_json  # noqa: E402
from mcue2e import costs, quant  # noqa: E402
from mcue2e.fold import fold_bn, remaining_bn  # noqa: E402
from torch import nn  # noqa: E402

RAW = PROJ / "artifacts/raw"
FIG = PROJ / "artifacts/figures"
WORK = PROJ / "artifacts/work"
ZOO = {"mcunet-vww0": "mcunet-10fps_vww", "mcunet-vww1": "mcunet-5fps_vww", "mcunet-vww2": "mcunet-320kb-1mb_vww"}
SUBNETS = {"ours-250KB-60M": "subnet_250KB_60M_pred.pt", "ours-200KB-30M": "subnet_200KB_30M_pred.pt"}
QUANT_VARIANTS = [("minmax", True, "uint8"), ("minmax", False, "uint8"), ("entropy", True, "uint8"),
                  ("entropy2048", True, "uint8"), ("percentile", True, "uint8"), ("minmax", True, "int8")]
DEPLOY = ("minmax", True, "uint8")
CALIB_N = 512


def summary_update(**kv):
    path = RAW / "summary.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(kv)
    write_json(path, data)


def vtag(v):
    return f"{v[0]}-{'pc' if v[1] else 'pt'}-{v[2]}"


# ---------------------------------------------------------------- models

def load_models() -> dict[str, tuple[nn.Module, int, dict]]:
    """name -> (fp32 model with BN, resolution, info)."""
    # The release configs are ProxylessNASNets configs; MCUNets reads the same four keys.
    from mcunet.tinynas.nn.networks.mcunets import MCUNets
    out = {}
    for name, stem in ZOO.items():
        cfg = json.load(open(ZOO_ROOT / f"{stem}.json"))
        m = MCUNets.build_from_config(cfg)
        m.load_state_dict(torch.load(ZOO_ROOT / f"{stem}.pth", map_location="cpu")["state_dict"])
        out[name] = (m.eval(), cfg["resolution"], {"source": f"MCUNet release {stem}", "tflite": f"{stem}.tflite"})
    for name, fn in SUBNETS.items():
        path = LAB3 / "artifacts/work" / fn
        if not path.exists():
            print(f"skip {name}: {path} missing")
            continue
        # net_config does not record the fused first block, so take the subnet from the supernet again.
        ck = torch.load(path, map_location="cpu")
        lab3, ofa = _supernet()
        m = lab3.subnet(ofa, ck["cfg"]).cpu()
        m.load_state_dict(ck["state_dict"])
        out[name] = (m.eval(), ck["cfg"]["image_size"], {"source": f"Lab 3 search, {fn}", "cfg": ck["cfg"]})
    return out


class Head(nn.Module):
    """What MCUNets / ProxylessNASNets run after the blocks."""

    def __init__(self, mix, classifier):
        super().__init__()
        self.mix, self.classifier = mix, classifier

    def forward(self, x):
        if self.mix is not None:
            x = self.mix(x)
        return self.classifier(x.mean(3).mean(2))


def split_features(model):
    return nn.Sequential(model.first_conv, *model.blocks), Head(model.feature_mix_layer, model.classifier)


# ---------------------------------------------------------------- data

def pixels(data, r, split):
    """uint8 N x 3 x r x r of a split, cached as .npy so worker processes can memory-map it."""
    path = WORK / f"vww_r{r}_s{split}.npy"
    if not path.exists():
        x, y = data.split(r, split)
        np.save(path, x.cpu().numpy())
        np.save(WORK / f"vww_r{r}_s{split}_y.npy", y.cpu().numpy())
    return path


def norm(u8):
    return ((u8.astype(np.float32) / 255.0) - 0.5) / 0.5


# ---------------------------------------------------------------- prepare

def stage_prepare(log):
    """Fold BN, check the fold, export fp32 ONNX, cache the evaluation pixels."""
    from nas.vww import VWW
    data = VWW(str(VWW_ROOT))
    rows = []
    for name, (m, r, info) in load_models().items():
        f = fold_bn(m)
        x = torch.from_numpy(norm(data.split(r, 0)[0][:256].cpu().numpy()))
        with torch.no_grad():
            d = (m(x) - f(x)).abs().max().item()
        quant.export_onnx(f, r, WORK / f"{name}_fp32.onnx")
        torch.save(f, WORK / f"{name}_folded.pt")
        for s in (0, 1):
            pixels(data, r, s)
        rows.append({"model": name, "resolution": r, "folded_pairs": f.folded_pairs, "bn_left": remaining_bn(f),
                     "fold_max_abs_logit_diff": d, **{k: v for k, v in info.items() if k != "cfg"}})
        log.write({"stage": "prepare", **rows[-1], "cfg": info.get("cfg")})
        print(rows[-1], flush=True)
    write_csv(RAW / "models.csv", rows)
    summary_update(models=[r["model"] for r in rows],
                   fold_max_abs_logit_diff=max(r["fold_max_abs_logit_diff"] for r in rows),
                   resolutions={r["model"]: r["resolution"] for r in rows})


# ---------------------------------------------------------------- accuracy

def _int8_job(args):
    name, r, v = args
    x0 = np.load(WORK / f"vww_r{r}_s0.npy", mmap_mode="r")
    y0 = np.load(WORK / f"vww_r{r}_s0_y.npy")
    x1 = np.load(WORK / f"vww_r{r}_s1.npy", mmap_mode="r")
    calib = norm(np.asarray(x1[:CALIB_N]))
    out = WORK / f"{name}_int8_{vtag(v)}.onnx"
    t0 = time.perf_counter()
    quant.quantize(WORK / f"{name}_fp32.onnx", out, calib, method=v[0], per_channel=v[1], activation=v[2])
    tq = time.perf_counter() - t0
    acc, _ = quant.ort_accuracy(out, norm(np.asarray(x0)), y0, threads=4)
    return {"model": name, "variant": vtag(v), "acc_split0": round(acc, 3), "onnx_bytes": out.stat().st_size,
            "quantize_s": round(tq, 1)}


def _fp32_ort_job(args):
    name, r = args
    x0 = np.load(WORK / f"vww_r{r}_s0.npy", mmap_mode="r")
    y0 = np.load(WORK / f"vww_r{r}_s0_y.npy")
    acc, _ = quant.ort_accuracy(WORK / f"{name}_fp32.onnx", norm(np.asarray(x0)), y0, threads=4)
    return {"model": name, "variant": "fp32-ort", "acc_split0": round(acc, 3),
            "onnx_bytes": (WORK / f"{name}_fp32.onnx").stat().st_size}


def _tflite_job(args):
    name, r, stem, lo, hi = args
    if LITERT_PATH:
        sys.path.insert(0, LITERT_PATH)
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(ZOO_ROOT / f"{stem}.tflite"), num_threads=1)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    x0 = np.load(WORK / f"vww_r{r}_s0.npy", mmap_mode="r")
    y0 = np.load(WORK / f"vww_r{r}_s0_y.npy")
    correct = 0
    for i in range(lo, hi):
        img = np.asarray(x0[i]).transpose(1, 2, 0).astype(np.int16) - 128  # the release's eval_tflite.py input
        it.set_tensor(inp["index"], img.astype(inp["dtype"])[None])
        it.invoke()
        correct += int(it.get_tensor(outp["index"]).reshape(-1).argmax() == y0[i])
    return name, correct, hi - lo, str(inp["dtype"]), [float(q) for q in inp["quantization"]]


def stage_accuracy(log):
    from nas.vww import VWW, accuracy
    data = VWW(str(VWW_ROOT))
    models = load_models()
    rows = []
    for name, (m, r, _info) in models.items():
        a0 = accuracy(m.cuda(), *data.split(r, 0))
        rows.append({"model": name, "variant": "fp32-torch", "acc_split0": round(a0, 3)})
        m.cpu()
    jobs = [(name, r, v) for name, (_m, r, _i) in models.items() for v in QUANT_VARIANTS]
    with ProcessPoolExecutor(min(len(jobs) + len(models), 48)) as ex:
        fp = list(ex.map(_fp32_ort_job, [(n, r) for n, (_m, r, _i) in models.items()]))
        q = list(ex.map(_int8_job, jobs))
    rows += fp + q
    if LITERT_PATH or _has_litert():
        tjobs = []
        for name, stem in ZOO.items():
            r = models[name][1]
            n = len(np.load(WORK / f"vww_r{r}_s0_y.npy"))
            step = -(-n // 16)
            tjobs += [(name, r, stem, lo, min(n, lo + step)) for lo in range(0, n, step)]
        with ProcessPoolExecutor(48) as ex:
            res = list(ex.map(_tflite_job, tjobs))
        for name in ZOO:
            c = sum(t[1] for t in res if t[0] == name)
            n = sum(t[2] for t in res if t[0] == name)
            first = next(t for t in res if t[0] == name)
            rows.append({"model": name, "variant": "official-tflite-int8", "acc_split0": round(100 * c / n, 3),
                         "onnx_bytes": (ZOO_ROOT / f"{ZOO[name]}.tflite").stat().st_size,
                         "note": f"input {first[3]} q{first[4]}"})
    for r_ in rows:
        log.write({"stage": "accuracy", **r_})
    write_csv(RAW / "accuracy.csv", rows)
    out = {}
    for r_ in rows:
        out[f"acc_{r_['model']}_{r_['variant']}"] = r_["acc_split0"]
    import filecmp
    for name in models:
        out[f"entropy_default_equals_minmax_{name}"] = filecmp.cmp(
            WORK / f"{name}_int8_entropy-pc-uint8.onnx", WORK / f"{name}_int8_minmax-pc-uint8.onnx", shallow=False)
        f32 = out[f"acc_{name}_fp32-torch"]
        out[f"int8_drop_{name}"] = round(f32 - out[f"acc_{name}_{vtag(DEPLOY)}"], 3)
    summary_update(**out)
    print(json.dumps(out, indent=1))


def _has_litert():
    try:
        import ai_edge_litert  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- memory

def stage_memory(log):
    """Peak SRAM by four accountings, patch-based inference, flash, MACs and the MCU latency estimate."""
    from mcunet.utils.pytorch_utils import count_net_flops, count_peak_activation_size
    from memplan.graph import inplace_depthwise, trace
    from memplan.patch import patch_plan
    from memplan.planners import arena_size, greedy_by_size, lower_bound
    fit = costs.fit_latency()
    fit0 = costs.fit_latency(through_origin=True)
    models = load_models()
    models.update(_extra_subnets())
    rows, prow = [], []
    for name, (m, r, _info) in models.items():
        f = fold_bn(m)
        shape = (1, 3, r, r)
        ts = trace(f, shape)
        ip = inplace_depthwise(ts)
        try:
            lab = int(count_peak_activation_size(m, shape))
        except Exception as e:  # noqa: BLE001
            print("lab peak failed", name, e)
            lab = None
        feats, head = split_features(f)
        best = {False: None, True: None}
        for cut in range(2, min(11, len(feats))):
            for p in (2, 3, 4):
                for inplace in (False, True):
                    pr = patch_plan(feats, head, r, cut, p, inplace=inplace)
                    prow.append({"model": name, "inplace": inplace, **pr.__dict__})
                    b = best[inplace]
                    if pr.net_mac_overhead <= 1.3 and (b is None or pr.peak < b.peak):
                        best[inplace] = pr
        bi, best = best[True], best[False]
        mm = costs.conv_macs(f, shape) / 1e6
        fl = costs.flash_bytes(f)
        rows.append({"model": name, "resolution": r, "mmacs_conv": round(mm, 3),
                     "mmacs_lab_counter": round(count_net_flops(m, shape) / 1e6, 3),
                     "params": sum(p.numel() for p in f.parameters()),
                     "flash_bytes": fl["total"], "flash_weights": fl["weights"],
                     "kb_lab_rule": None if lab is None else round(lab / 1024, 2),
                     "kb_lower_bound": round(lower_bound(ts) / 1024, 2),
                     "kb_greedy": round(arena_size(ts, greedy_by_size(ts)) / 1024, 2),
                     "kb_greedy_inplace": round(arena_size(ip, greedy_by_size(ip)) / 1024, 2),
                     "kb_patch": None if best is None else round(best.peak / 1024, 2),
                     "patch_cut": None if best is None else best.cut,
                     "patch_p": None if best is None else best.patches,
                     "patch_mac_overhead": None if best is None else round(best.net_mac_overhead, 3),
                     "kb_patch_inplace": None if bi is None else round(bi.peak / 1024, 2),
                     "patch_inplace_cut": None if bi is None else bi.cut,
                     "patch_inplace_p": None if bi is None else bi.patches,
                     "patch_inplace_mac_overhead": None if bi is None else round(bi.net_mac_overhead, 3),
                     "mcu_ms_est": round(fit(mm), 1), "mcu_ms_est_origin": round(fit0(mm), 1)})
        log.write({"stage": "memory", **rows[-1]})
        print(rows[-1], flush=True)
    write_csv(RAW / "memory.csv", rows)
    write_csv(RAW / "patch.csv", prow)
    out = {"latency_fit_ms_per_mmac": round(fit.ms_per_mmac, 4), "latency_fit_intercept_ms": round(fit.intercept_ms, 3),
           "latency_fit_max_rel_residual": round(fit.max_rel_residual, 4),
           "latency_fit0_ms_per_mmac": round(fit0.ms_per_mmac, 4),
           "latency_fit0_max_rel_residual": round(fit0.max_rel_residual, 4)}
    for r_ in rows:
        for k in ("mmacs_conv", "mmacs_lab_counter", "flash_bytes", "kb_lab_rule", "kb_greedy", "kb_greedy_inplace",
                  "kb_patch", "patch_mac_overhead", "kb_patch_inplace", "patch_inplace_mac_overhead", "mcu_ms_est"):
            out[f"{k}_{r_['model']}"] = r_[k]
    summary_update(**out)
    plot_patch(prow)


_SUPERNET = []


def _supernet():
    if not _SUPERNET:
        import importlib.util
        os.environ.setdefault("MCUNET_ROOT", str(MCUNET_ROOT))
        spec = importlib.util.spec_from_file_location("lab3_experiments", LAB3 / "scripts" / "run_experiments.py")
        lab3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lab3)
        lab3.DEV = "cpu"
        _SUPERNET.extend((lab3, lab3.load_supernet()))
    return _SUPERNET


def _extra_subnets():
    """Memory-only subnets (supernet weights, no accuracy claimed): the all-minimum one, for Q10's 64 KB."""
    from nas import space
    lab3, ofa = _supernet()
    out = {}
    for tag, cfg in (("smallest-subnet", space.smallest()),):
        sub = lab3.subnet(ofa, cfg).cpu().eval()
        out[tag] = (sub, cfg["image_size"], {"cfg": cfg})
    return out


def plot_patch(prow):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for name in dict.fromkeys(r["model"] for r in prow):
        rs = [r for r in prow if r["model"] == name and r["inplace"]]
        front = {}
        for r in rs:
            k = round(r["net_mac_overhead"], 2)
            front[k] = min(front.get(k, 1e18), r["peak"])
        xs = sorted(front)
        ys, best = [], 1e18
        for x in xs:
            best = min(best, front[x])
            ys.append(best / 1024)
        ax.step(xs, ys, where="post", label=name)
    for kb in (64, 128, 256, 320):
        ax.axhline(kb, color="grey", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("whole-network MACs relative to whole-image execution")
    ax.set_ylabel("peak SRAM (KB, int8, greedy arena, in-place dw)")
    ax.set_title("patch-based inference: best peak at each compute overhead", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "patch_frontier.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- latency

def _time(fn, warmup=30, repeats=300):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    ts.sort()
    return {"median_ms": round(statistics.median(ts), 4), "p10_ms": round(ts[len(ts) // 10], 4),
            "p90_ms": round(ts[9 * len(ts) // 10], 4)}


def stage_latency(log):
    """Batch-1, one thread, pinned to one core: torch fp32, ORT fp32, ORT int8 and the official .tflite."""
    try:
        os.sched_setaffinity(0, {int(os.environ.get("PIN_CORE", 3))})
    except (AttributeError, OSError):
        pass
    torch.set_num_threads(1)
    rows = []
    for name, (m, r, _info) in load_models().items():
        x = np.load(WORK / f"vww_r{r}_s0.npy", mmap_mode="r")
        one = norm(np.asarray(x[:1]))
        f = torch.load(WORK / f"{name}_folded.pt", weights_only=False).eval()
        t = torch.from_numpy(one)
        with torch.inference_mode():
            rows.append({"model": name, "backend": "torch-fp32-folded", **_time(lambda: f(t))})
        for tag, path in (("ort-fp32", WORK / f"{name}_fp32.onnx"), ("ort-int8", WORK / f"{name}_int8_{vtag(DEPLOY)}.onnx")):
            s = quant.session(path, 1)
            rows.append({"model": name, "backend": tag, **_time(lambda s=s: s.run(None, {"x": one}))})
        if name in ZOO and (LITERT_PATH or _has_litert()):
            if LITERT_PATH:
                sys.path.insert(0, LITERT_PATH)
            from ai_edge_litert.interpreter import Interpreter
            it = Interpreter(model_path=str(ZOO_ROOT / f"{ZOO[name]}.tflite"), num_threads=1)
            it.allocate_tensors()
            inp = it.get_input_details()[0]
            img = (np.asarray(x[0]).transpose(1, 2, 0).astype(np.int16) - 128).astype(inp["dtype"])[None]
            it.set_tensor(inp["index"], img)
            rows.append({"model": name, "backend": "litert-int8-official", **_time(it.invoke)})
        for r_ in rows[-4:]:
            print(r_, flush=True)
    for r_ in rows:
        log.write({"stage": "latency", **r_})
    write_csv(RAW / "latency.csv", rows)
    summary_update(**{f"lat_{r_['model']}_{r_['backend']}_ms": r_["median_ms"] for r_ in rows})


# ---------------------------------------------------------------- table

def stage_table(log):
    import csv
    acc = {(r["model"], r["variant"]): float(r["acc_split0"]) for r in csv.DictReader(open(RAW / "accuracy.csv"))}
    mem = {r["model"]: r for r in csv.DictReader(open(RAW / "memory.csv"))}
    lat = {(r["model"], r["backend"]): float(r["median_ms"]) for r in csv.DictReader(open(RAW / "latency.csv"))}
    rows = []
    for name in list(ZOO) + list(SUBNETS):
        if name not in mem:
            continue
        m = mem[name]
        rows.append({"model": name, "resolution": int(m["resolution"]), "mmacs": float(m["mmacs_conv"]),
                     "acc_fp32": acc.get((name, "fp32-torch")), "acc_int8": acc.get((name, vtag(DEPLOY))),
                     "acc_official_tflite": acc.get((name, "official-tflite-int8")),
                     "flash_kb": round(int(m["flash_bytes"]) / 1024, 1),
                     "sram_kb_greedy_inplace": float(m["kb_greedy_inplace"]),
                     "sram_kb_patch": float(m["kb_patch_inplace"]) if m["kb_patch_inplace"] else None,
                     "patch_mac_overhead": float(m["patch_inplace_mac_overhead"]) if m["patch_inplace_mac_overhead"] else None,
                     "mcu_ms_est": float(m["mcu_ms_est"]),
                     "x86_ort_int8_ms": lat.get((name, "ort-int8")), "x86_ort_fp32_ms": lat.get((name, "ort-fp32"))})
    write_csv(RAW / "final_table.csv", rows)
    log.write({"stage": "table", "rows": rows})
    plot_table(rows)
    for r in rows:
        print(r)


def plot_table(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for r in rows:
        mk = "o" if r["model"].startswith("mcunet") else "*"
        for ax, x in ((axes[0], "mcu_ms_est"), (axes[1], "sram_kb_greedy_inplace")):
            ax.scatter(r[x], r["acc_int8"], marker=mk, s=60 if mk == "*" else 30)
            ax.annotate(r["model"], (r[x], r["acc_int8"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set_xlabel("estimated STM32F746 latency (ms, fitted to TinyEngine)")
    axes[1].set_xlabel("peak SRAM (KB, int8 greedy arena, in-place depthwise)")
    for ax in axes:
        ax.set_ylabel("int8 VWW top-1 (%)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "final_tradeoff.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", default=["prepare", "accuracy", "memory", "latency", "table"])
    args = ap.parse_args()
    for d in (RAW, FIG, WORK):
        d.mkdir(parents=True, exist_ok=True)
    log = JsonlWriter(RAW / f"runs_{'_'.join(args.stage)}.jsonl", cwd=REPO, fresh=True)
    stages = {"prepare": stage_prepare, "accuracy": stage_accuracy, "memory": stage_memory,
              "latency": stage_latency, "table": stage_table}
    for st in args.stage:
        t0 = time.perf_counter()
        stages[st](log)
        log.write({"stage": st, "seconds": round(time.perf_counter() - t0, 1)})
        print(f"== {st} done in {time.perf_counter() - t0:.0f}s", flush=True)
    log.close()


if __name__ == "__main__":
    main()
