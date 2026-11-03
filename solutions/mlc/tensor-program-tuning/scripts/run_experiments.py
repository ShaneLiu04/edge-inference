"""Schedule search for fp32 matmul on this CPU: four strategies, same budget, same measurement.

  python scripts/run_experiments.py               # idle machine; compiles with 4 g++ in parallel
  python scripts/run_experiments.py --budget 32   # quicker
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]
sys.path[:0] = [str(LAB / "code"), str(ROOT)]

from edge_harness.writer import write_csv, write_json  # noqa: E402
from tuner.runner import measure  # noqa: E402
from tuner.schedule import Problem, Schedule, Space  # noqa: E402
from tuner.search import STRATEGIES, Ridge, best_so_far, features  # noqa: E402

RAW = LAB / "artifacts/raw"
FIG = LAB / "artifacts/figures"
WORK = LAB / "cpp/bin"
INCLUDE = ROOT / "edge_harness/include"

PROBLEMS = [Problem("square_256", 256, 256, 256), Problem("mbv2_pw_expand_14", 196, 576, 96)]

NUMPY_PROBE = r"""
import json, sys, time
import numpy as np
M, N, K = map(int, sys.argv[1:4])
a = np.random.default_rng(0).random((M, K), dtype=np.float32)
b = np.random.default_rng(1).random((K, N), dtype=np.float32)
for _ in range(5):
    a @ b
ts = []
for _ in range(50):
    t0 = time.perf_counter(); a @ b; ts.append(time.perf_counter() - t0)
ts.sort()
print(json.dumps({"gflops": 2 * M * N * K / ts[len(ts) // 2] / 1e9}))
"""


def numpy_single_thread(p: Problem) -> float:
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", NUMPY_PROBE, str(p.M), str(p.N), str(p.K)], env=env,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["gflops"]


def holdout_spearman(cache: dict[str, dict], seed: int) -> float:
    """Rank quality of the search's cost model: fit on a random half of every measured schedule, rank the rest."""
    import random

    keys = sorted(cache)
    random.Random(seed).shuffle(keys)
    half = len(keys) // 2

    def xy(ks: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([features(Schedule.from_key(k)) for k in ks])
        return X, np.log([max(cache[k]["gflops"], 1e-3) for k in ks])

    model = Ridge()
    model.fit(*xy(keys[:half]))
    X, y = xy(keys[half:])
    pred = model.predict(X)
    rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    return float(np.corrcoef(rank(pred), rank(y))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="independent runs per strategy; measurements are cached")
    args = ap.parse_args()
    all_hist, summary, width_rows = [], {}, []
    for p in PROBLEMS:
        space = Space(p)
        cache: dict[str, dict] = {}

        def run(batch: list[Schedule]) -> dict[str, float]:
            todo = [s for s in batch if s.key() not in cache]
            if todo:
                cache.update(measure(p, todo, WORK, INCLUDE, batch=16, jobs=4, repeats=5))
            bad = [k for k in (s.key() for s in batch) if cache[k]["max_rel_err"] > 1e-4]
            if bad:
                raise RuntimeError(f"wrong results for {bad}")
            return {s.key(): cache[s.key()]["gflops"] for s in batch}

        baseline = run([Schedule("ijk", "ijk", 1, 16, 8, False, 1)])  # untiled-equivalent naive loop order
        # 32x64x64 where it divides the problem, otherwise the largest legal tile below it
        tile = [max(d for d in opts if d <= cap) for opts, cap in ((space.ti, 32), (space.tj, 64), (space.tk, 64))]
        textbook_s = Schedule("ijk", "ikj", *tile, False, 1)
        textbook = run([textbook_s])
        summary[p.name] = {"space_size": space.size(), "naive_ijk_gflops": next(iter(baseline.values())),
                           "textbook_ikj_gflops": next(iter(textbook.values())), "textbook_key": textbook_s.key(),
                           "numpy_1thread_gflops": numpy_single_thread(p)}
        overall = 0.0
        for name, fn in STRATEGIES.items():
            per_seed = []
            for seed in range(args.seed, args.seed + args.seeds):
                hist = fn(space, run, args.budget, seed=seed)
                for h in hist:
                    h["problem"], h["seed"] = p.name, seed
                all_hist += hist
                per_seed.append(hist)
            bests = [max(h, key=lambda r: r["gflops"]) for h in per_seed]
            top = max(bests, key=lambda r: r["gflops"])
            overall = max(overall, top["gflops"])
            s = summary[p.name]
            s[f"{name}_best_gflops"] = top["gflops"]
            s[f"{name}_best_key"] = top["key"]
            s[f"{name}_best_gflops_per_seed"] = [b["gflops"] for b in bests]
            s[f"{name}_median_best_gflops"] = float(np.median([b["gflops"] for b in bests]))
            print(p.name, name, [round(b["gflops"], 1) for b in bests], top["key"])
        # trials until a seed's run reaches 90% of the best any strategy found: same bar for every strategy
        for name in STRATEGIES:
            runs = [[h for h in all_hist if h["problem"] == p.name and h["strategy"] == name and h["seed"] == sd]
                    for sd in range(args.seed, args.seed + args.seeds)]
            hits = [next((i + 1 for i, v in enumerate(best_so_far(r)) if v >= 0.9 * overall), None) for r in runs]
            summary[p.name][f"{name}_trials_to_90pct_of_overall"] = hits
        summary[p.name]["overall_best_gflops"] = overall
        summary[p.name]["measured_unique"] = len(cache)
        summary[p.name]["cost_model_holdout_spearman"] = holdout_spearman(cache, args.seed)

        # The search ran with GCC's default 256-bit auto-vectorization; re-time the top schedules both ways.
        top = [Schedule.from_key(k) for k in sorted(cache, key=lambda k: -cache[k]["gflops"])[:8]]
        w256 = measure(p, top, WORK, INCLUDE, batch=8, jobs=1, repeats=5)
        w512 = measure(p, top, WORK, INCLUDE, batch=8, jobs=1, repeats=5, flags=("-mprefer-vector-width=512",))
        for s in top:
            if w512[s.key()]["max_rel_err"] > 1e-4:
                raise RuntimeError(f"wrong results for {s.key()} at 512-bit")
            width_rows.append({"problem": p.name, "key": s.key(), "gflops_256": w256[s.key()]["gflops"],
                               "gflops_512": w512[s.key()]["gflops"]})
        summary[p.name]["top8_best_gflops_256"] = max(w256[s.key()]["gflops"] for s in top)
        summary[p.name]["top8_best_gflops_512"] = max(w512[s.key()]["gflops"] for s in top)
        print(p.name, summary[p.name])
    write_csv(RAW / "search_history.csv", all_hist)
    write_csv(RAW / "vector_width.csv", width_rows)
    write_json(RAW / "summary.json", summary)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(11, 3.8))
    for ax, p in zip(axes, PROBLEMS):
        for name in STRATEGIES:
            curves = [best_so_far([h for h in all_hist if h["problem"] == p.name and h["strategy"] == name
                                   and h["seed"] == sd]) for sd in range(args.seed, args.seed + args.seeds)]
            n = min(len(c) for c in curves)
            arr = np.array([c[:n] for c in curves])
            x = range(1, n + 1)
            line, = ax.plot(x, np.median(arr, axis=0), label=f"{name} (median of {args.seeds} seeds)")
            ax.fill_between(x, arr.min(axis=0), arr.max(axis=0), color=line.get_color(), alpha=0.15)
        s = summary[p.name]
        ax.axhline(s["numpy_1thread_gflops"], color="k", ls="--", lw=0.8, label="numpy (BLAS, 1 thread)")
        ax.axhline(s["textbook_ikj_gflops"], color="grey", ls=":", lw=0.8, label=f"textbook {s['textbook_key']}")
        ax.set_title(f"{p.name} (M={p.M}, N={p.N}, K={p.K})", fontsize=9)
        ax.set_xlabel("measured schedules")
        ax.set_ylabel("best GFLOP/s so far")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "search_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
