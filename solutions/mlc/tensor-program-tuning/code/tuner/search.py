"""Search strategies over ``Space``: grid, random, cost-model guided (AutoTVM-style), and evolutionary.

Each strategy proposes schedules in rounds; ``measure_fn(list[Schedule]) -> {key: gflops}`` runs them.
The cost model is ridge regression on hand-written features, refit after every round. It ranks a
large random pool and the top few (plus some random exploration) are measured next, which is the
loop AutoTVM runs with XGBoost and TVM MetaSchedule runs with its own model.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import numpy as np

from .schedule import ORDERS, Schedule, Space, footprint_bytes

MeasureFn = Callable[[list[Schedule]], dict[str, float]]

L1, L2 = 48 * 1024, 1280 * 1024


def features(s: Schedule) -> np.ndarray:
    fp = footprint_bytes(s)
    tile = fp["A"] + fp["B"] + fp["C"]
    f = [math.log2(s.ti), math.log2(s.tj), math.log2(s.tk),
         float(tile <= L1), float(tile <= L2), math.log2(tile),
         float(s.inner[-1] == "j"), float(s.inner[-1] == "k"), float(s.inner[-1] == "i"),
         float(s.pack_b), float(s.unroll > 1), float(s.pack_b and s.inner[-1] == "j"),
         math.log2(s.tj) * float(s.inner[-1] == "j")]
    f += [float(s.outer == o) for o in ORDERS]
    f += [float(s.inner == o) for o in ORDERS]
    return np.array(f)


class Ridge:
    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Xb = np.hstack([X, np.ones((len(X), 1))])
        self.w = np.linalg.solve(Xb.T @ Xb + self.lam * np.eye(Xb.shape[1]), Xb.T @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([X, np.ones((len(X), 1))]) @ self.w


def _record(history: list[dict], batch: list[Schedule], res: dict[str, float], strategy: str) -> None:
    for s in batch:
        if s.key() in res:
            history.append({"strategy": strategy, "trial": len(history) + 1, "key": s.key(), "gflops": res[s.key()],
                            **s.as_dict()})


def grid(space: Space, measure: MeasureFn, budget: int, seed: int = 0) -> list[dict]:
    """The textbook schedule (ikj inner, ijk outer, no packing) and a sweep of its three tile sizes."""
    pts = [Schedule("ijk", "ikj", a, b, c, False, 1) for a in space.ti for b in space.tj for c in space.tk]
    random.Random(seed).shuffle(pts)
    pts = pts[:budget]
    hist: list[dict] = []
    _record(hist, pts, measure(pts), "grid_tiles_only")
    return hist


def random_search(space: Space, measure: MeasureFn, budget: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    seen, pts = set(), []
    while len(pts) < budget:
        s = space.sample(rng)
        if s.key() not in seen:
            seen.add(s.key())
            pts.append(s)
    hist: list[dict] = []
    _record(hist, pts, measure(pts), "random")
    return hist


def model_guided(space: Space, measure: MeasureFn, budget: int, seed: int = 0, rounds: int = 4,
                 pool: int = 4000, explore: float = 0.25) -> list[dict]:
    rng = random.Random(seed)
    per_round = budget // rounds
    hist: list[dict] = []
    seen: set[str] = set()

    def fresh(n: int) -> list[Schedule]:
        out = []
        while len(out) < n:
            s = space.sample(rng)
            if s.key() not in seen:
                seen.add(s.key())
                out.append(s)
        return out

    batch = fresh(per_round)
    _record(hist, batch, measure(batch), "cost_model")
    model = Ridge()
    for _ in range(rounds - 1):
        X = np.array([features(Schedule(**{k: h[k] for k in Schedule.__dataclass_fields__})) for h in hist])
        y = np.log([max(h["gflops"], 1e-3) for h in hist])
        model.fit(X, y)
        cands: dict[str, Schedule] = {}
        target = min(pool, space.size() - len(seen))  # small spaces have fewer unseen points than the pool
        while len(cands) < target:
            s = space.sample(rng)
            if s.key() not in seen:
                cands[s.key()] = s
        cl = list(cands.values())
        pred = model.predict(np.array([features(s) for s in cl]))
        n_explore = int(per_round * explore)
        best = [cl[i] for i in np.argsort(-pred)[: per_round - n_explore]]
        batch = best + rng.sample([s for s in cl if s not in best], n_explore)
        for s in batch:
            seen.add(s.key())
        _record(hist, batch, measure(batch), "cost_model")
    return hist


def evolutionary(space: Space, measure: MeasureFn, budget: int, seed: int = 0, rounds: int = 4) -> list[dict]:
    rng = random.Random(seed)
    per_round = budget // rounds
    hist: list[dict] = []
    seen: set[str] = set()
    batch = []
    while len(batch) < per_round:
        s = space.sample(rng)
        if s.key() not in seen:
            seen.add(s.key())
            batch.append(s)
    _record(hist, batch, measure(batch), "evolutionary")
    for _ in range(rounds - 1):
        elite = sorted(hist, key=lambda h: -h["gflops"])[:4]
        batch = []
        tries = 0
        while len(batch) < per_round and tries < 10000:
            tries += 1
            parent = Schedule(**{k: rng.choice(elite)[k] for k in Schedule.__dataclass_fields__}) if rng.random() < 0.2 \
                else Schedule(**{k: elite[rng.randrange(len(elite))][k] for k in Schedule.__dataclass_fields__})
            for child in space.neighbors(parent, rng, 1):
                if child.key() not in seen:
                    seen.add(child.key())
                    batch.append(child)
        _record(hist, batch, measure(batch), "evolutionary")
    return hist


STRATEGIES = {"grid_tiles_only": grid, "random": random_search, "cost_model": model_guided,
              "evolutionary": evolutionary}


def best_so_far(hist: list[dict]) -> list[float]:
    out, best = [], 0.0
    for h in hist:
        best = max(best, h["gflops"])
        out.append(best)
    return out
