"""MIT 6.5940 Lab 3 on the course OFA-MCUNet supernet and VWW.

  python scripts/run_experiments.py --stage effcheck reproduce q1 predictor ablate search pareto

Needs the course package (assignments/lab3-resources): $MCUNET_ROOT is the unpacked
mcunetv2-dev-main (supernet code, vww_supernet.pth, acc_datasets/) and $VWW_ROOT is
data/vww-s256/val. One GPU; every stage takes minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[2]
MCUNET_ROOT = Path(os.environ.get("MCUNET_ROOT", LAB / "artifacts/work/mcunetv2-dev-main"))
VWW_ROOT = Path(os.environ.get("VWW_ROOT", LAB / "artifacts/work/data/vww-s256/val"))
sys.path[:0] = [str(LAB / "code"), str(LAB.parent / "tinyengine-memory" / "code"), str(REPO), str(MCUNET_ROOT)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from edge_harness.writer import JsonlWriter, write_csv, write_json  # noqa: E402
from nas import efficiency as eff  # noqa: E402
from nas import predictor as pred  # noqa: E402
from nas import space  # noqa: E402
from nas.search import Constraint, EvoParams, evolution, random_search  # noqa: E402
from nas.vww import VWW, evaluate  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"
WORK = LAB / "artifacts/work"
DEV = "cuda"
Q9 = (Constraint(kb=250, mmacs=60), Constraint(kb=200, mmacs=30))


def summary_update(**kv):
    """Merge into summary.json; locked, because a CPU-side process and the GPU process may both write."""
    path = RAW / "summary.json"
    with open(RAW / ".summary.lock", "w") as lock:
        try:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        except ImportError:
            pass
        data = json.loads(path.read_text()) if path.exists() else {}
        data.update(kv)
        write_json(path, data)


def load_supernet():
    from mcunet.tinynas.elastic_nn.networks.ofa_mcunets import OFAMCUNets
    net = OFAMCUNets(n_classes=2, bn_param=(0.1, 1e-3), dropout_rate=0.0, base_stage_width="mcunet384",
                     width_mult_list=[0.5, 0.75, 1.0], ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6],
                     depth_list=[0, 1, 2], base_depth=[1, 2, 2, 2, 2], fuse_blk1=True,
                     se_stages=[False, [False, True, True, True], True, True, True, False])
    sd = torch.load(MCUNET_ROOT / "vww_supernet.pth", map_location="cpu")["state_dict"]
    net.load_state_dict(sd, strict=True)
    return net.to(DEV).eval()


def subnet(ofa, cfg):
    ofa.set_active_subnet(**{k: cfg[k] for k in ("wid", "ks", "e", "d")})
    return ofa.get_active_subnet(preserve_weight=True).to(DEV)


class Evaluator:
    """Real accuracy of a subnet: BN re-calibrated on the holdout split, top-1 on both splits."""

    def __init__(self, ofa, data):
        self.ofa, self.data, self.n = ofa, data, 0

    def __call__(self, cfg) -> dict:
        self.n += 1
        return evaluate(subnet(self.ofa, cfg), self.data, cfg["image_size"])


def row(cfg, **extra):
    e = eff.efficiency(cfg)
    return {"cfg": cfg, "mmacs": round(e["mmacs"], 4), "mmacs_conv": round(e["mmacs_conv"], 4), "kb": round(e["kb"], 3),
            "kb_residual": round(e["kb_residual"], 3), "peak_at": e["peak_at"], **extra}


# ---------------------------------------------------------------- effcheck

def stage_effcheck(log):
    """Closed-form MACs / peak / params against the course hook-based counters, subnet by subnet."""
    from mcunet.utils.pytorch_utils import count_net_flops, count_peak_activation_size
    from memplan.graph import inplace_depthwise, trace
    from memplan.planners import arena_size, greedy_by_size, lower_bound
    ofa = load_supernet()
    rng = random.Random(0)
    cfgs = [space.smallest()] + [space.largest(w) for w in range(3)]
    cfgs += [space.sample(rng, widths=(0, 1, 2)) for _ in range(1000)]
    mism = {"macs": 0, "peak": 0, "params": 0}
    t_lib = t_ana = 0.0
    arena_rows = []
    for i, cfg in enumerate(cfgs):
        r = cfg["image_size"]
        sub = subnet(ofa, cfg).eval()
        t0 = time.perf_counter()
        lib_macs = int(count_net_flops(sub, (1, 3, r, r)))
        lib_peak = int(count_peak_activation_size(sub, (1, 3, r, r)))
        t_lib += time.perf_counter() - t0
        t0 = time.perf_counter()
        a_macs = eff.macs(cfg)
        a_peak, _ = eff.peak(cfg, "lab")
        t_ana += time.perf_counter() - t0
        p_mod = sum(m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
                    for m in sub.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)))
        ok = {"macs": lib_macs == a_macs, "peak": lib_peak == a_peak, "params": p_mod == eff.params(cfg)}
        for k, v in ok.items():
            mism[k] += not v
        if not all(ok.values()):
            log.write({"stage": "effcheck", "match": ok, "cfg": cfg, "lib_macs": lib_macs, "macs": a_macs,
                       "lib_peak": lib_peak, "peak": a_peak, "params_mod": p_mod, "params": eff.params(cfg)})
        if i < 4 or i % 10 == 0:
            cpu = sub.cpu().eval()
            ts = trace(cpu, (1, 3, r, r))
            ip = inplace_depthwise(ts)
            e = eff.efficiency(cfg)
            arena_rows.append({"i": i, "wid": cfg["wid"], "r": r, "mmacs": round(e["mmacs"], 3),
                               "kb_lab": round(e["kb"], 3), "kb_residual": round(e["kb_residual"], 3),
                               "kb_lower_bound": round(lower_bound(ts) / 1024, 3),
                               "kb_greedy": round(arena_size(ts, greedy_by_size(ts)) / 1024, 3),
                               "kb_greedy_inplace": round(arena_size(ip, greedy_by_size(ip)) / 1024, 3)})
    write_csv(RAW / "arena_vs_lab.csv", arena_rows)
    a = np.array([[r_["kb_lab"], r_["kb_residual"], r_["kb_greedy"], r_["kb_greedy_inplace"]] for r_ in arena_rows])
    sm = space.smallest()
    e_sm = eff.efficiency(sm)
    out = {"effcheck_n": len(cfgs), "effcheck_macs_mismatch": mism["macs"], "effcheck_peak_mismatch": mism["peak"],
           "effcheck_params_mismatch": mism["params"],
           "effcheck_lib_ms_per_subnet": round(1e3 * t_lib / len(cfgs), 2),
           "effcheck_analytic_us_per_subnet": round(1e6 * t_ana / len(cfgs), 1),
           "arena_n": len(arena_rows),
           "arena_greedy_over_lab_mean": round(float((a[:, 2] / a[:, 0]).mean()), 4),
           "arena_greedy_over_lab_max": round(float((a[:, 2] / a[:, 0]).max()), 4),
           "arena_greedy_over_residual_mean": round(float((a[:, 2] / a[:, 1]).mean()), 4),
           "arena_inplace_over_lab_mean": round(float((a[:, 3] / a[:, 0]).mean()), 4),
           "arena_residual_over_lab_mean": round(float((a[:, 1] / a[:, 0]).mean()), 4),
           "smallest_mmacs": round(e_sm["mmacs"], 3), "smallest_kb": round(e_sm["kb"], 3),
           "smallest_kb_residual": round(e_sm["kb_residual"], 3), "smallest_peak_at": e_sm["peak_at"]}
    for w in range(3):
        e_l = eff.efficiency(space.largest(w))
        out[f"largest_w{w}_mmacs"] = round(e_l["mmacs"], 3)
        out[f"largest_w{w}_kb"] = round(e_l["kb"], 3)
    log.write({"stage": "effcheck", **out})
    summary_update(**out)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- reproduce

def load_acc_dict():
    return json.load(open(MCUNET_ROOT / "acc_datasets" / "acc.dict"))


def stage_reproduce(log, data, ofa):
    """Re-measure entries of the 50k accuracy dataset to check this evaluation path against the labels."""
    acc = load_acc_dict()
    idx = random.Random(0).sample(range(len(acc)), 100)
    ev = Evaluator(ofa, data)
    rows = []
    for i in idx:
        cfg, label = acc[i]
        m = ev(cfg)
        rows.append({"index": i, "label": round(label, 4), **{k: round(v, 4) for k, v in m.items()}})
        log.write({"stage": "reproduce", **rows[-1]})
    write_csv(RAW / "reproduce.csv", rows)
    d1 = np.array([r["acc_split1"] - r["label"] for r in rows])
    d0 = np.array([r["acc_split0"] - r["label"] for r in rows])
    out = {"reproduce_n": len(rows), "reproduce_split1_mean_abs_diff": round(float(np.abs(d1).mean()), 4),
           "reproduce_split1_max_abs_diff": round(float(np.abs(d1).max()), 4),
           "reproduce_split0_minus_label_mean": round(float(d0.mean()), 4),
           "reproduce_split0_label_pearson": round(pred.pearson([r["acc_split0"] for r in rows],
                                                                [r["label"] for r in rows]), 4)}
    summary_update(**out)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- q1

def stage_q1(log, data, ofa):
    """One-at-a-time sweeps around a middle subnet, plus the extremes and random draws (Q1)."""
    ev = Evaluator(ofa, data)
    center = dict(wid=1, k=5, e=4, d=1, image_size=128)
    sweeps = {"image_size": space.RESOLUTIONS, "wid": (0, 1, 2), "k": space.KERNELS, "e": space.EXPANDS,
              "d": space.DEPTHS}
    rows = []

    def run(tag, value, cfg):
        m = ev(cfg)
        rows.append(row(cfg, sweep=tag, value=value, **{k: round(v, 3) for k, v in m.items()}))
        log.write({"stage": "q1", **rows[-1]})

    for dim, values in sweeps.items():
        for v in values:
            c = dict(center, **{dim: v})
            run(dim, v, space.uniform(c["wid"], c["k"], c["e"], c["d"], c["image_size"]))
    run("extreme", "smallest", space.smallest())
    for w in range(3):
        run("extreme", f"largest_w{w}", space.largest(w))
    rng = random.Random(1)
    for i in range(30):
        run("random", i, space.sample(rng, widths=(0, 1, 2)))
    write_csv(RAW / "q1.csv", [{**{k: v for k, v in r.items() if k != "cfg"}, "cfg": json.dumps(r["cfg"])}
                               for r in rows])
    out = {}
    for dim in sweeps:
        accs = [r["acc_split0"] for r in rows if r["sweep"] == dim]
        out[f"q1_range_{dim}"] = round(max(accs) - min(accs), 3)
    for r in rows:
        if r["sweep"] == "extreme":
            out[f"q1_{r['value']}_acc"] = r["acc_split0"]
    rnd = [r["acc_split0"] for r in rows if r["sweep"] == "random"]
    out.update(q1_random_min=min(rnd), q1_random_max=max(rnd))
    summary_update(**out)
    plot_q1(rows)
    print(json.dumps(out, indent=1))


def plot_q1(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    names = {"image_size": "resolution 96-160", "wid": "width 0.5/0.75/1.0", "k": "kernel 3/5/7",
             "e": "expand 3/4/6", "d": "extra depth 0/1/2"}
    for dim, label in names.items():
        rs = [r for r in rows if r["sweep"] == dim]
        axes[0].plot([r["mmacs"] for r in rs], [r["acc_split0"] for r in rs], "o-", label=label)
    rnd = [r for r in rows if r["sweep"] == "random"]
    for ax, x, xl in ((axes[0], "mmacs", "MACs (M)"), (axes[1], "kb", "peak activation (KB, lab rule)")):
        ax.scatter([r[x] for r in rnd], [r["acc_split0"] for r in rnd], c=[r["cfg"]["wid"] for r in rnd],
                   cmap="viridis", s=14, alpha=0.7, label="random (colour = width)" if ax is axes[0] else None)
        ax.set_xlabel(xl)
        ax.set_ylabel("VWW top-1 (%)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[0].set_title("one knob at a time around w0.75 k5 e4 d1 r128", fontsize=9)
    axes[1].set_title("30 random subnets", fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "q1_sweeps.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- predictor

def acc_tensors():
    acc = load_acc_dict()
    x = torch.tensor(np.stack([space.encode(c) for c, _ in acc]))
    y = torch.tensor([v / 100.0 for _, v in acc], dtype=torch.float32)
    return acc, x, y


def train_predictor(x, y, n_train, seed, epochs=10):
    torch.manual_seed(seed)
    tr, va = pred.split(len(x), 40000, seed)
    tr = tr[:n_train]
    xt, yt, xv, yv = x[tr].to(DEV), y[tr].to(DEV), x[va].to(DEV), y[va].to(DEV)
    model = pred.AccuracyMLP(base_acc=float(yt.mean())).to(DEV)
    hist = pred.fit(model, xt, yt, xv, yv, epochs=epochs, seed=seed)
    p = pred.predict(model, xv).cpu().numpy()
    t = yv.cpu().numpy()
    m = {"mae_pp": float(np.abs(p - t).mean() * 100), "pearson": pred.pearson(p, t),
         "spearman": pred.spearman(p, t), "kendall": pred.kendall_tau(p, t),
         "top1pct_overlap": pred.top_overlap(p, t, 0.01), "baseline_mae_pp": float(np.abs(t - yt.mean().item()).mean() * 100)}
    return model, hist, m, (p, t)


def stage_predictor(log):
    from mcunet.tinynas.search.accuracy_predictor import MCUNetArchEncoder
    acc, x, y = acc_tensors()
    enc = MCUNetArchEncoder(base_depth=[1, 2, 2, 2, 2], image_size_list=list(space.RESOLUTIONS),
                            width_mult_list=list(space.WIDTHS), ks_list=list(space.KERNELS),
                            expand_list=list(space.EXPANDS), depth_list=list(space.DEPTHS))
    enc_mismatch = sum(not np.array_equal(enc.arch2feature(c).astype(np.float32), space.encode(c)) for c, _ in acc[:2000])
    rows = []
    best = None
    for seed in (0, 1, 2):
        t0 = time.perf_counter()
        model, hist, m, pt = train_predictor(x, y, 40000, seed)
        m["train_s"] = time.perf_counter() - t0
        rows.append({"n_train": 40000, "seed": seed, **{k: round(v, 5) for k, v in m.items()}})
        log.write({"stage": "predictor", "seed": seed, "hist": hist, **m})
        if seed == 0:
            best, best_pt, best_hist = model, pt, hist
    for n in (500, 2000, 5000, 10000, 20000):
        _, _, m, _ = train_predictor(x, y, n, 0)
        rows.append({"n_train": n, "seed": 0, **{k: round(v, 5) for k, v in m.items()}})
        log.write({"stage": "predictor", "n_train": n, **m})
    write_csv(RAW / "predictor.csv", rows)
    WORK.mkdir(parents=True, exist_ok=True)
    torch.save(best.state_dict(), WORK / "acc_predictor.pt")
    full = [r for r in rows if r["n_train"] == 40000]
    out = {"encoder_mismatch_of_2000": int(enc_mismatch),
           "acc_dataset_n": len(acc), "acc_dataset_mean": round(float(y.mean()) * 100, 4),
           "acc_dataset_min": round(float(y.min()) * 100, 4), "acc_dataset_max": round(float(y.max()) * 100, 4),
           "pred_val_mae_pp_mean": round(float(np.mean([r["mae_pp"] for r in full])), 4),
           "pred_val_mae_pp_seed0": full[0]["mae_pp"], "pred_baseline_mae_pp": full[0]["baseline_mae_pp"],
           "pred_val_pearson_seed0": full[0]["pearson"], "pred_val_spearman_seed0": full[0]["spearman"],
           "pred_val_kendall_seed0": full[0]["kendall"], "pred_val_top1pct_overlap_seed0": full[0]["top1pct_overlap"],
           "pred_train_s_seed0": round(full[0]["train_s"], 2), "pred_epoch1_val_l1": round(best_hist[0]["val_l1"] * 100, 4),
           "pred_epoch10_val_l1": round(best_hist[-1]["val_l1"] * 100, 4)}
    for r in rows:
        if r["n_train"] != 40000:
            out[f"pred_n{r['n_train']}_mae_pp"] = r["mae_pp"]
            out[f"pred_n{r['n_train']}_kendall"] = r["kendall"]
    summary_update(**out)
    plot_scatter(*best_pt)
    print(json.dumps(out, indent=1))


def plot_scatter(p, t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.scatter(t * 100, p * 100, s=2, alpha=0.25)
    lo, hi = 84, 94
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured holdout accuracy (%)")
    ax.set_ylabel("predicted (%)")
    ax.set_title("accuracy predictor, 10,000 held-out subnets", fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "predictor_scatter.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- search

def load_predictor():
    m = pred.AccuracyMLP().to(DEV)
    m.load_state_dict(torch.load(WORK / "acc_predictor.pt", map_location=DEV))
    m.eval()

    def score(cfgs):
        x = torch.tensor(np.stack([space.encode(c) for c in cfgs])).to(DEV)
        return (pred.predict(m, x) * 100).cpu().tolist()
    return score


def stage_search(log, data, ofa):
    score = load_predictor()
    ev = Evaluator(ofa, data)
    out = {}
    cache = {}

    def efficiency(cfg):
        k = space.key(cfg)
        if k not in cache:
            cache[k] = eff.efficiency(cfg)
        return cache[k]

    def measured(cfg):
        return {k: round(v, 3) for k, v in ev(cfg).items()}

    # Q5 / Q6: random search, four single constraints, 300 feasible samples each.
    q6 = []
    for c in (Constraint(mmacs=50), Constraint(mmacs=100), Constraint(kb=256), Constraint(kb=512)):
        res = random_search(300, c, efficiency, score, random.Random(0))
        m = measured(res.best)
        q6.append(row(res.best, constraint=c.label(), predicted=round(res.best_score, 3), **m))
        log.write({"stage": "q6", **q6[-1]})
        out[f"q6_{c.label()}_acc"] = m["acc_split0"]
        out[f"q6_{c.label()}_mmacs"] = q6[-1]["mmacs"]
        out[f"q6_{c.label()}_kb"] = q6[-1]["kb"]
    # The notebook's random-search cell spells the MACs key "millonMACs"; satisfy_constraint skips unknown
    # keys, so a MACs-only constraint becomes no constraint. Reproduce what that cell then returns.
    res = random_search(300, Constraint(), efficiency, score, random.Random(0))
    typo = row(res.best, constraint="typo (no constraint)", predicted=round(res.best_score, 3))
    log.write({"stage": "q6_typo", **typo})
    out.update(q6_typo_mmacs=typo["mmacs"], q6_typo_kb=typo["kb"])
    write_csv(RAW / "q6_random.csv", [{**{k: v for k, v in r.items() if k != "cfg"}, "cfg": json.dumps(r["cfg"])}
                                      for r in q6 + [typo]])

    # Q8: evolution hyper-parameters on 250KB / 60M, five seeds per cell.
    c = Q9[0]
    sweep = []
    for popn in (10, 50, 100):
        for gens in (10, 30, 60):
            bests, queries = [], []
            for seed in range(5):
                r_ = evolution(c, efficiency, score, random.Random(seed), EvoParams(population=popn, generations=gens))
                bests.append(r_.best_score)
                queries.append(r_.queries)
            sweep.append({"population": popn, "generations": gens, "queries_mean": float(np.mean(queries)),
                          "best_pred_mean": round(float(np.mean(bests)), 4), "best_pred_std": round(float(np.std(bests)), 4)})
            log.write({"stage": "q8", **sweep[-1]})
    write_csv(RAW / "q8_evo_sweep.csv", sweep)
    out["q8_default_best_pred"] = sweep[0]["best_pred_mean"]
    out["q8_largest_best_pred"] = sweep[-1]["best_pred_mean"]

    # Equal-budget curves: random vs evolution on 250KB / 60M, ten seeds.
    budget = 5000
    curves = {"random": [], "evolution": []}
    for seed in range(10):
        rr = random_search(budget, c, efficiency, score, random.Random(100 + seed))
        er = evolution(c, efficiency, score, random.Random(100 + seed), EvoParams(population=100, generations=66))
        curves["random"].append(_curve(rr.history, budget))
        curves["evolution"].append(_curve(er.history, budget))
    grid = [100, 300, 1000, 2000, 5000]
    crow = []
    for name, cs in curves.items():
        arr = np.array(cs)
        for g in grid:
            crow.append({"searcher": name, "queries": g, "best_pred_mean": round(float(arr[:, g - 1].mean()), 4),
                         "best_pred_std": round(float(arr[:, g - 1].std()), 4)})
    write_csv(RAW / "search_curves.csv", crow)
    for r_ in crow:
        if r_["queries"] in (300, 5000):
            out[f"curve_{r_['searcher']}_{r_['queries']}"] = r_["best_pred_mean"]
    plot_curves(curves)

    # Q9: two compound constraints. Five evolution seeds; the distinct winners are measured; pick by holdout.
    q9 = []
    for c in Q9:
        cands = {}
        for seed in range(5):
            r_ = evolution(c, efficiency, score, random.Random(seed), EvoParams(population=100, generations=60))
            cands.setdefault(space.key(r_.best), (r_.best, r_.best_score))
        rows_c = []
        for cfg, s in cands.values():
            rows_c.append(row(cfg, constraint=c.label(), predicted=round(s, 3), **measured(cfg)))
            log.write({"stage": "q9", **rows_c[-1]})
        pick = max(rows_c, key=lambda r_: r_["acc_split1"])
        pick["picked"] = True
        q9 += rows_c
        rnd = random_search(300, c, efficiency, score, random.Random(0))
        rm = measured(rnd.best)
        q9.append(row(rnd.best, constraint=c.label(), predicted=round(rnd.best_score, 3), searcher="random300", **rm))
        lab = c.label()
        out.update({f"q9_{lab}_acc": pick["acc_split0"], f"q9_{lab}_holdout": pick["acc_split1"],
                    f"q9_{lab}_pred": pick["predicted"], f"q9_{lab}_mmacs": pick["mmacs"], f"q9_{lab}_kb": pick["kb"],
                    f"q9_{lab}_kb_residual": pick["kb_residual"], f"q9_{lab}_n_candidates": len(rows_c),
                    f"q9_{lab}_random300_acc": rm["acc_split0"],
                    f"q9_{lab}_cands_acc_min": min(r_["acc_split0"] for r_ in rows_c),
                    f"q9_{lab}_cands_acc_max": max(r_["acc_split0"] for r_ in rows_c)})
        export_subnet(ofa, data, pick["cfg"], lab)
    write_csv(RAW / "q9.csv", [{**{k: v for k, v in r.items() if k != "cfg"}, "cfg": json.dumps(r["cfg"])} for r in q9])

    # Q10: feasibility. Every MACs and peak term grows with every knob, so the all-minimum subnet bounds both.
    sm = eff.efficiency(space.smallest())
    out.update(q10_min_mmacs=round(sm["mmacs"], 3), q10_min_kb=round(sm["kb"], 3), q10_min_peak_at=sm["peak_at"],
               q10_a_feasible=bool(sm["mmacs"] <= 15 and sm["kb"] <= 256), q10_b_feasible=bool(sm["kb"] <= 64))
    units = eff.units(space.smallest())
    out["q10_min_unit_kb"] = {u.name: round(eff.unit_peak(u) / 1024, 2) for u in units[:4]}

    # Extension: let evolution score with measured holdout accuracy, with and without width 1.0.
    for widths, tag in (((0, 1), "w01"), ((0, 1, 2), "w012")):
        ev_cache = {}

        def real_score(cfgs, ev_cache=ev_cache):
            res_ = []
            for cfg in cfgs:
                k = space.key(cfg)
                if k not in ev_cache:
                    ev_cache[k] = ev(cfg)
                res_.append(ev_cache[k]["acc_split1"])
            return res_

        t0 = time.perf_counter()
        r_ = evolution(Q9[0], efficiency, real_score, random.Random(0), EvoParams(population=16, generations=8),
                       widths=widths)
        m = ev_cache[space.key(r_.best)]
        out.update({f"real_evo_{tag}_acc": round(m["acc_split0"], 3), f"real_evo_{tag}_holdout": round(m["acc_split1"], 3),
                    f"real_evo_{tag}_evals": len(ev_cache), f"real_evo_{tag}_minutes": round((time.perf_counter() - t0) / 60, 2),
                    f"real_evo_{tag}_wid": r_.best["wid"], f"real_evo_{tag}_mmacs": round(efficiency(r_.best)["mmacs"], 3),
                    f"real_evo_{tag}_kb": round(efficiency(r_.best)["kb"], 3)})
        log.write({"stage": "real_evo", "tag": tag, "best": r_.best, **m})
    out["evals_total"] = ev.n
    summary_update(**out)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- q9plus

def stage_q9plus(log, data, ofa):
    """Q9 with a larger budget: does more search close the gap to 92.5% / 90%, and what does selection cost?

    A: twenty predictor-driven evolutions; the ten best distinct members of every final population
       are measured, and the one with the best holdout accuracy is picked.
    B: evolution scored directly by measured holdout accuracy, population 48 for 20 generations.
    The best validation accuracy among everything measured is reported as the oracle, which a
    holdout-based pick cannot reach without looking at the validation set.
    """
    score = load_predictor()
    ev = Evaluator(ofa, data)
    measured: dict[bytes, dict] = {}

    def measure(cfg):
        k = space.key(cfg)
        if k not in measured:
            measured[k] = {kk: round(v, 3) for kk, v in ev(cfg).items()}
        return measured[k]

    cache = {}

    def efficiency(cfg):
        k = space.key(cfg)
        if k not in cache:
            cache[k] = eff.efficiency(cfg)
        return cache[k]

    rows, out = [], {}
    for c in Q9:
        lab = c.label()
        cands = {}
        for seed in range(20):
            r_ = evolution(c, efficiency, score, random.Random(1000 + seed), EvoParams(population=100, generations=60))
            seen = set()
            for s, cfg in r_.population:
                k = space.key(cfg)
                if k in seen:
                    continue
                seen.add(k)
                cands.setdefault(k, (cfg, s))
                if len(seen) == 10:
                    break
        a_rows = [row(cfg, constraint=lab, method="predictor-evo x20 top10", predicted=round(s, 3), **measure(cfg))
                  for cfg, s in cands.values()]
        t0 = time.perf_counter()

        def real_score(cfgs):
            return [measure(cfg)["acc_split1"] for cfg in cfgs]

        n_before = len(measured)
        r_ = evolution(c, efficiency, real_score, random.Random(7), EvoParams(population=48, generations=20))
        b_rows = [row(cfg, constraint=lab, method="real-evo 48x20", predicted=round(score([cfg])[0], 3), **measure(cfg))
                  for _, cfg in r_.population[:10]]
        out[f"q9plus_{lab}_real_evals"] = len(measured) - n_before
        out[f"q9plus_{lab}_real_minutes"] = round((time.perf_counter() - t0) / 60, 2)
        for tag, rs in (("pred", a_rows), ("real", b_rows)):
            pick = max(rs, key=lambda r: r["acc_split1"])
            out[f"q9plus_{lab}_{tag}_n"] = len(rs)
            out[f"q9plus_{lab}_{tag}_acc"] = pick["acc_split0"]
            out[f"q9plus_{lab}_{tag}_holdout"] = pick["acc_split1"]
            out[f"q9plus_{lab}_{tag}_mmacs"] = pick["mmacs"]
            out[f"q9plus_{lab}_{tag}_kb"] = pick["kb"]
            out[f"q9plus_{lab}_{tag}_oracle_acc"] = max(r["acc_split0"] for r in rs)
            pick["picked"] = True
        both = a_rows + b_rows
        best = max(both, key=lambda r: r["acc_split1"])
        out[f"q9plus_{lab}_acc"] = best["acc_split0"]
        out[f"q9plus_{lab}_holdout"] = best["acc_split1"]
        out[f"q9plus_{lab}_method"] = best["method"]
        out[f"q9plus_{lab}_oracle_acc"] = max(r["acc_split0"] for r in both)
        a1 = np.array([r["acc_split1"] for r in both])
        a0 = np.array([r["acc_split0"] for r in both])
        out[f"q9plus_{lab}_holdout_val_kendall"] = round(pred.kendall_tau(a1, a0), 4)
        export_subnet(ofa, data, best["cfg"], lab + "_plus")
        rows += both
        for r in both:
            log.write({"stage": "q9plus", **r})
    _write_cfg_csv(RAW / "q9plus.csv", rows)
    out["q9plus_measured_total"] = len(measured)
    summary_update(**out)
    print(json.dumps(out, indent=1))


def stage_final(log, data, ofa):
    """The Q9 answer by the course procedure: of everything q9plus measured, the subnet the predictor ranks first.

    Picking by holdout accuracy is reported beside it. Among near-top subnets the two splits
    barely agree (q9plus_*_holdout_val_kendall), so the holdout pick mostly selects holdout noise.
    """
    import csv
    rows = list(csv.DictReader(open(RAW / "q9plus.csv")))
    out = {}
    for c in Q9:
        lab = c.label()
        rs = [dict(r, cfg=json.loads(r["cfg"]), **{k: float(r[k]) for k in ("acc_split0", "acc_split1", "predicted")})
              for r in rows if r["constraint"] == lab]
        pick = max(rs, key=lambda r: r["predicted"])
        hold = max(rs, key=lambda r: r["acc_split1"])
        n = len(data.split(pick["cfg"]["image_size"], 0)[1])
        p = pick["acc_split0"] / 100
        out.update({f"q9final_{lab}_acc": pick["acc_split0"], f"q9final_{lab}_holdout": pick["acc_split1"],
                    f"q9final_{lab}_pred": pick["predicted"], f"q9final_{lab}_mmacs": float(pick["mmacs"]),
                    f"q9final_{lab}_mmacs_conv": float(pick["mmacs_conv"]), f"q9final_{lab}_kb": float(pick["kb"]),
                    f"q9final_{lab}_n_measured": len(rs),
                    f"q9final_{lab}_n_meeting_bar": sum(r["acc_split0"] >= (92.5 if c.kb == 250 else 90.0) for r in rs),
                    f"q9final_{lab}_holdout_pick_acc": hold["acc_split0"],
                    f"q9final_{lab}_val_se_pp": round(100 * (p * (1 - p) / n) ** 0.5, 3), "val_n": n})
        export_subnet(ofa, data, pick["cfg"], lab + "_pred")
        m = evaluate(subnet(ofa, pick["cfg"]), data, pick["cfg"]["image_size"])
        out[f"q9final_{lab}_remeasured_acc"] = round(m["acc_split0"], 3)
        log.write({"stage": "final", "constraint": lab, "cfg": pick["cfg"], **m})
    summary_update(**out)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- pareto

def stage_pareto(log, data, ofa):
    """Measure many random subnets, width 1.0 included, for the real accuracy-efficiency front.

    Gives the predictor's error on fresh subnets (split 1 is its training protocol, split 0 the
    real validation set), its error out of distribution (width 1.0 never appears in its data), how
    well holdout accuracy selects for validation accuracy, and where the Q9 picks sit on the front.
    """
    n = int(os.environ.get("PARETO_N", 2000))
    score = load_predictor()
    ev = Evaluator(ofa, data)
    rng = random.Random(7)
    rows = []
    t0 = time.perf_counter()
    for i in range(n):
        cfg = space.sample(rng, widths=(0, 1, 2))
        m = ev(cfg)
        rows.append(row(cfg, predicted=round(score([cfg])[0], 3), **{k: round(v, 3) for k, v in m.items()}))
        log.write({"stage": "pareto", **rows[-1]})
        if (i + 1) % 200 == 0:
            print(f"pareto {i + 1}/{n} {time.perf_counter() - t0:.0f}s", flush=True)
            _write_cfg_csv(RAW / "pareto.csv", rows)
    _write_cfg_csv(RAW / "pareto.csv", rows)
    out = {"pareto_n": n, "pareto_s_per_subnet": round((time.perf_counter() - t0) / n, 3)}
    for w in range(3):
        rs = [r for r in rows if r["cfg"]["wid"] == w]
        p = np.array([r["predicted"] for r in rs])
        for s in (0, 1):
            t = np.array([r[f"acc_split{s}"] for r in rs])
            out[f"pareto_w{w}_split{s}_mae_pp"] = round(float(np.abs(p - t).mean()), 4)
            out[f"pareto_w{w}_split{s}_kendall"] = round(pred.kendall_tau(p, t), 4)
            out[f"pareto_w{w}_split{s}_bias_pp"] = round(float((p - t).mean()), 4)
        out[f"pareto_w{w}_n"] = len(rs)
        out[f"pareto_w{w}_acc0_mean"] = round(float(np.mean([r["acc_split0"] for r in rs])), 3)
    a0 = np.array([r["acc_split0"] for r in rows])
    a1 = np.array([r["acc_split1"] for r in rows])
    out["pareto_split0_split1_pearson"] = round(pred.pearson(a0, a1), 4)
    out["pareto_split0_split1_kendall"] = round(pred.kendall_tau(a0, a1), 4)
    out["pareto_split0_minus_split1_mean"] = round(float((a0 - a1).mean()), 4)
    front = []
    for kb in (128, 192, 256, 320, 384, 512):
        for mm in (15, 30, 45, 60, 80, 100, 150):
            c = Constraint(kb=kb, mmacs=mm)
            feas = [r for r in rows if c.ok(r)]
            if not feas:
                continue
            by1 = max(feas, key=lambda r: r["acc_split1"])
            by0 = max(feas, key=lambda r: r["acc_split0"])
            byp = max(feas, key=lambda r: r["predicted"])
            front.append({"kb": kb, "mmacs": mm, "n_feasible": len(feas), "best_acc0": by0["acc_split0"],
                          "acc0_of_best_holdout": by1["acc_split0"], "acc0_of_best_predicted": byp["acc_split0"],
                          "wid_of_best": by0["cfg"]["wid"]})
    write_csv(RAW / "pareto_front.csv", front)
    for f in front:
        if (f["kb"], f["mmacs"]) in ((256, 60), (192, 30)):
            out[f"pareto_{f['kb']}KB_{f['mmacs']}M_best_acc0"] = f["best_acc0"]
            out[f"pareto_{f['kb']}KB_{f['mmacs']}M_n"] = f["n_feasible"]
    summary_update(**out)
    plot_pareto(rows)
    print(json.dumps(out, indent=1))


def _write_cfg_csv(path, rows):
    write_csv(path, [{**{k: v for k, v in r.items() if k != "cfg"}, "cfg": json.dumps(r["cfg"])} for r in rows])


def _front(xs, ys):
    pts = sorted(zip(xs, ys))
    fx, fy, best = [], [], -1.0
    for x, y in pts:
        if y > best:
            fx.append(x)
            fy.append(y)
            best = y
    return fx, fy


def plot_pareto(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    colors = ("tab:blue", "tab:orange", "tab:green")
    for w in range(3):
        rs = [r for r in rows if r["cfg"]["wid"] == w]
        lab = f"width {space.WIDTHS[w]}"
        for ax, x in ((axes[0], "mmacs"), (axes[1], "kb")):
            xs, ys = [r[x] for r in rs], [r["acc_split0"] for r in rs]
            ax.scatter(xs, ys, s=3, alpha=0.3, color=colors[w])
            ax.step(*_front(xs, ys), where="post", color=colors[w], label=lab)
        axes[2].scatter([r["acc_split1"] for r in rs], [r["predicted"] for r in rs], s=3, alpha=0.3,
                        color=colors[w], label=lab + (" (unseen)" if w == 2 else ""))
    axes[0].set_xlabel("MACs (M, lab counter)")
    axes[1].set_xlabel("peak activation (KB, lab rule)")
    for ax in axes[:2]:
        ax.set_ylabel("validation top-1 (%)")
        ax.set_ylim(84, 95)
    axes[2].plot([84, 95], [84, 95], "k--", lw=0.8)
    axes[2].set_xlabel("measured holdout top-1 (%)")
    axes[2].set_ylabel("predicted (%)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"{len(rows)} random subnets, measured", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "pareto.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- ablate

def stage_ablate(log):
    """Predictor design choices: size, depth, loss, epochs, ensembling. Three seeds each."""
    acc, x, y = acc_tensors()
    variants = [("lab 3x400 L1 e10", dict()), ("1x400", dict(n_layers=1)), ("5x400", dict(n_layers=5)),
                ("3x100", dict(hidden=100)), ("3x1000", dict(hidden=1000)), ("MSE", dict(loss="mse")),
                ("e30", dict(epochs=30)), ("e30 lr3e-4", dict(epochs=30, lr=3e-4)), ("batch 1024", dict(batch=1024))]
    rows = []
    for name, kw in variants:
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            tr, va = pred.split(len(x), 40000, seed)
            xt, yt, xv, yv = x[tr].to(DEV), y[tr].to(DEV), x[va].to(DEV), y[va].to(DEV)
            m = pred.AccuracyMLP(hidden=kw.get("hidden", 400), n_layers=kw.get("n_layers", 3),
                                 base_acc=float(yt.mean())).to(DEV)
            t0 = time.perf_counter()
            pred.fit(m, xt, yt, xv, yv, epochs=kw.get("epochs", 10), batch=kw.get("batch", 256),
                     lr=kw.get("lr", 1e-3), seed=seed, loss=kw.get("loss", "l1"))
            p, t = pred.predict(m, xv).cpu().numpy(), yv.cpu().numpy()
            rows.append({"variant": name, "seed": seed, "mae_pp": round(float(np.abs(p - t).mean() * 100), 4),
                         "kendall": round(pred.kendall_tau(p, t), 4),
                         "top1pct_overlap": round(pred.top_overlap(p, t, 0.01), 3),
                         "train_s": round(time.perf_counter() - t0, 2)})
            log.write({"stage": "ablate", **rows[-1]})
    # Ensemble of five on one split: does averaging buy ranking quality?
    tr, va = pred.split(len(x), 40000, 0)
    xt, yt, xv, yv = x[tr].to(DEV), y[tr].to(DEV), x[va].to(DEV), y[va].to(DEV)
    ps = []
    for seed in range(5):
        torch.manual_seed(100 + seed)
        m = pred.AccuracyMLP(base_acc=float(yt.mean())).to(DEV)
        pred.fit(m, xt, yt, xv, yv, seed=100 + seed)
        ps.append(pred.predict(m, xv).cpu().numpy())
    t = yv.cpu().numpy()
    for k in (1, 3, 5):
        p = np.mean(ps[:k], axis=0)
        rows.append({"variant": f"ensemble x{k}", "seed": 0, "mae_pp": round(float(np.abs(p - t).mean() * 100), 4),
                     "kendall": round(pred.kendall_tau(p, t), 4),
                     "top1pct_overlap": round(pred.top_overlap(p, t, 0.01), 3), "train_s": None})
    write_csv(RAW / "ablate.csv", rows)
    out = {}
    for name in dict.fromkeys(r["variant"] for r in rows):
        rs = [r for r in rows if r["variant"] == name]
        key_ = name.replace(" ", "_").replace("-", "")
        out[f"ablate_{key_}_mae_pp"] = round(float(np.mean([r["mae_pp"] for r in rs])), 4)
        out[f"ablate_{key_}_kendall"] = round(float(np.mean([r["kendall"] for r in rs])), 4)
    summary_update(**out)
    print(json.dumps(out, indent=1))


def _curve(history, budget):
    best = np.full(budget, np.nan)
    for q, b in history:
        if q <= budget:
            best[q - 1] = b
    return list(np.fmax.accumulate(np.nan_to_num(best, nan=-np.inf)))


def plot_curves(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for name, cs in curves.items():
        arr = np.array(cs)
        x = np.arange(1, arr.shape[1] + 1)
        ax.plot(x, arr.mean(0), label=f"{name} (mean of {len(cs)} seeds)")
        ax.fill_between(x, arr.min(0), arr.max(0), alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("predictor queries")
    ax.set_ylabel("best predicted accuracy (%)")
    ax.set_ylim(91.5, 94.5)
    ax.set_title("250 KB / 60M MACs", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "search_curves.png", dpi=140)
    plt.close(fig)


def export_subnet(ofa, data, cfg, label):
    """BN-recalibrated weights of a chosen subnet, for the mcu-e2e project."""
    sub = subnet(ofa, cfg)
    evaluate(sub, data, cfg["image_size"])
    WORK.mkdir(parents=True, exist_ok=True)
    tag = label.replace("/", "_")
    torch.save({"cfg": cfg, "state_dict": sub.cpu().state_dict(), "net_config": sub.config}, WORK / f"subnet_{tag}.pt")
    write_json(RAW / f"subnet_{tag}.json", {"cfg": cfg, **eff.efficiency(cfg)})


GPU_STAGES = {"reproduce": stage_reproduce, "q1": stage_q1, "search": stage_search, "q9plus": stage_q9plus,
              "final": stage_final, "pareto": stage_pareto}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+",
                    default=["effcheck", "reproduce", "q1", "predictor", "ablate", "search", "q9plus", "final",
                             "pareto"])
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    log = JsonlWriter(RAW / f"runs_{'_'.join(args.stage)}.jsonl", cwd=REPO, fresh=True)
    data = ofa = None
    for st in args.stage:
        t0 = time.perf_counter()
        if st in GPU_STAGES and data is None:
            data = VWW(str(VWW_ROOT))
            for r in space.RESOLUTIONS:
                data.pixels(r)
            ofa = load_supernet()
        if st in GPU_STAGES:
            GPU_STAGES[st](log, data, ofa)
        else:
            {"effcheck": stage_effcheck, "predictor": stage_predictor, "ablate": stage_ablate}[st](log)
        log.write({"stage": st, "seconds": round(time.perf_counter() - t0, 1)})
        print(f"== {st} done in {time.perf_counter() - t0:.0f}s", flush=True)
    log.close()


if __name__ == "__main__":
    main()
