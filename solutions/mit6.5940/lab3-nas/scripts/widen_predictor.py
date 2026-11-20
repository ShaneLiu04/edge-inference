"""Does adding measured width-1.0 subnets fix the predictor's width-1.0 bias? CPU only.

  python scripts/widen_predictor.py

The course accuracy set (acc.dict) only has width 0.5 and 0.75, so the predictor under-predicts width 1.0
by about 0.9 pp (pareto_w2_split1_bias_pp). The 684 width-1.0 subnets measured in the Pareto sample are
the only width-1.0 labels we have. This script runs 5-fold cross-validation over them:

  base  : the course training set only (same 40,000 rows and seed as the predictor stage);
  w1x1  : base + the four training folds of width-1.0 rows, once;
  w1x4  : base + the same rows repeated four times (up-weighted).

Each fold's held-out width-1.0 subnets are scored; the course validation split is scored for every variant
to check that width 1.0 does not cost accuracy on the widths the course set covers.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(LAB / "scripts"), str(LAB / "code")]
RAW = LAB / "artifacts/raw"
VARIANTS = {"base": 0, "w1x1": 1, "w1x4": 4}
SEEDS = (0, 1)
FOLDS = 5


def _width1_rows():
    from nas import space

    w1 = space.WIDTHS.index(1.0)
    cfgs, accs = [], []
    with open(RAW / "pareto.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cfg = json.loads(r["cfg"])
            if cfg["wid"] == w1:
                cfgs.append(cfg)
                accs.append(float(r["acc_split1"]) / 100.0)
    return cfgs, np.array(accs, dtype=np.float32)


def run_one(args):
    """One (seed, fold, variant) training run. Everything is rebuilt inside the worker."""
    seed, fold, variant = args
    import torch

    torch.set_num_threads(2)
    import run_experiments as lab3

    lab3.DEV = "cpu"
    from nas import predictor as pred
    from nas import space

    acc, x, y = lab3.acc_tensors()
    cfgs, aw = _width1_rows()
    xw = torch.tensor(np.stack([space.encode(c) for c in cfgs]))
    yw = torch.tensor(aw)
    perm = np.random.default_rng(seed).permutation(len(yw))
    te = perm[fold::FOLDS]
    tr_w = np.setdiff1d(perm, te)

    base_tr, base_va = pred.split(len(x), 40000, seed)
    reps = VARIANTS[variant]
    xt = torch.cat([x[base_tr[:40000]]] + [xw[tr_w]] * reps)
    yt = torch.cat([y[base_tr[:40000]]] + [yw[tr_w]] * reps)
    torch.manual_seed(seed)
    model = pred.AccuracyMLP(base_acc=float(yt.mean()))
    pred.fit(model, xt, yt, x[base_va], y[base_va], epochs=10, seed=seed)
    model.eval()

    p_w = pred.predict(model, xw[te]).numpy()
    t_w = yw[te].numpy()
    p_v = pred.predict(model, x[base_va]).numpy()
    t_v = y[base_va].numpy()
    return {
        "seed": seed, "fold": fold, "variant": variant, "n_test_w1": int(len(te)),
        "w1_mae_pp": float(np.abs(p_w - t_w).mean() * 100),
        "w1_bias_pp": float((p_w - t_w).mean() * 100),
        "w1_kendall": pred.kendall_tau(p_w, t_w),
        "course_val_mae_pp": float(np.abs(p_v - t_v).mean() * 100),
        "maxrss_mb": _peak_rss_mb(),
    }


def _peak_rss_mb() -> int:
    """Peak resident memory of this process (Unix only; 0 elsewhere)."""
    try:
        import resource
    except ImportError:
        return 0
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def _job_subprocess(args, attempts: int = 3):
    """Each run gets its own interpreter (forking after torch has initialised crashed the pool).
    A run the kernel kills (the container has a small memory limit) is retried."""
    seed, fold, variant = args
    for _ in range(attempts):
        out = subprocess.run([sys.executable, __file__, "--job", str(seed), str(fold), variant],
                             capture_output=True, text=True)
        if out.returncode == 0:
            return json.loads(out.stdout.strip().splitlines()[-1])
    raise RuntimeError(f"job {args} failed {attempts} times: {out.stderr[-500:]}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--job":
        print(json.dumps(run_one((int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]))))
        return
    jobs = [(s, k, v) for s in SEEDS for k in range(FOLDS) for v in VARIANTS]
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_job_subprocess, jobs))

    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / "widen_predictor.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    summary: dict = {"widen_n_width1": int(sum(r["n_test_w1"] for r in results if r["seed"] == 0 and r["variant"] == "base"))}
    for v in VARIANTS:
        rs = [r for r in results if r["variant"] == v]
        for key in ("w1_mae_pp", "w1_bias_pp", "w1_kendall", "course_val_mae_pp"):
            vals = np.array([r[key] for r in rs])
            summary[f"widen_{v}_{key}_mean"] = round(float(vals.mean()), 4)
            summary[f"widen_{v}_{key}_sd"] = round(float(vals.std(ddof=1)), 4)
    (RAW / "widen_predictor.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
