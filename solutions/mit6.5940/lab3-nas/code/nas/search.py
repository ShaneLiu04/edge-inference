"""Random search (Q5) and evolutionary search (Q7) under MACs / peak-memory constraints.

Both searchers take ``efficiency(cfg) -> dict`` and a batched ``score(cfgs) -> list[float]``.
They count every score query, so random and evolution can be compared at equal budget.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import space


@dataclass(frozen=True)
class Constraint:
    kb: float | None = None
    mmacs: float | None = None

    def ok(self, eff: dict) -> bool:
        return (self.kb is None or eff["kb"] <= self.kb) and (self.mmacs is None or eff["mmacs"] <= self.mmacs)

    def label(self) -> str:
        parts = []
        if self.kb is not None:
            parts.append(f"{self.kb:g}KB")
        if self.mmacs is not None:
            parts.append(f"{self.mmacs:g}M")
        return "/".join(parts) or "none"


class Infeasible(RuntimeError):
    pass


@dataclass
class Result:
    best: dict
    best_score: float
    queries: int
    history: list[tuple[int, float]] = field(default_factory=list)  # (score queries so far, best so far)
    population: list[tuple[float, dict]] = field(default_factory=list)  # final generation, best first


class _Scorer:
    def __init__(self, score):
        self._score = score
        self.queries = 0
        self.best_score = float("-inf")
        self.best: dict | None = None
        self.history: list[tuple[int, float]] = []

    def __call__(self, cfgs: list[dict]) -> list[float]:
        s = list(self._score(cfgs))
        for c, v in zip(cfgs, s):
            self.queries += 1
            if v > self.best_score:
                self.best_score, self.best = v, c
            self.history.append((self.queries, self.best_score))
        return s


def _draw(make, constraint: Constraint, efficiency, max_tries: int) -> dict:
    for _ in range(max_tries):
        cfg = make()
        if constraint.ok(efficiency(cfg)):
            return cfg
    raise Infeasible(f"no subnet satisfies {constraint.label()} in {max_tries} draws")


def random_search(n_subnets: int, constraint: Constraint, efficiency, score, rng: random.Random,
                  widths=space.PREDICTOR_WIDTHS, max_tries: int = 100_000) -> Result:
    """Sample until ``n_subnets`` subnets satisfy the constraint, then keep the best-scoring one."""
    scorer = _Scorer(score)
    cands = [_draw(lambda: space.sample(rng, widths), constraint, efficiency, max_tries) for _ in range(n_subnets)]
    scorer(cands)
    return Result(scorer.best, scorer.best_score, scorer.queries, scorer.history)


@dataclass(frozen=True)
class EvoParams:
    population: int = 100
    generations: int = 30
    parent_ratio: float = 0.25
    mutation_ratio: float = 0.5
    arch_mutate_prob: float = 0.1
    resolution_mutate_prob: float = 0.5


def evolution(constraint: Constraint, efficiency, score, rng: random.Random, p: EvoParams = EvoParams(),
              widths=space.PREDICTOR_WIDTHS, max_tries: int = 100_000) -> Result:
    """Keep the top ``parent_ratio`` each generation; refill with mutations of one parent and crossovers of two.

    Children that violate the constraint are redrawn, so every scored subnet is feasible.
    """
    scorer = _Scorer(score)
    pop = [_draw(lambda: space.sample(rng, widths), constraint, efficiency, max_tries) for _ in range(p.population)]
    scores = scorer(pop)
    n_parents = max(1, int(round(p.population * p.parent_ratio)))
    n_mut = int(round(p.population * p.mutation_ratio))
    for _ in range(p.generations):
        ranked = sorted(zip(scores, range(len(pop))), key=lambda t: -t[0])[:n_parents]
        parents = [pop[i] for _, i in ranked]
        parent_scores = [s for s, _ in ranked]
        children = []
        for _ in range(n_mut):
            children.append(_draw(lambda: space.mutate(rng.choice(parents), rng, p.arch_mutate_prob,
                                                       p.resolution_mutate_prob, widths),
                                  constraint, efficiency, max_tries))
        while len(parents) + len(children) < p.population:
            children.append(_draw(lambda: space.crossover(rng.choice(parents), rng.choice(parents), rng),
                                  constraint, efficiency, max_tries))
        pop = parents + children
        scores = parent_scores + scorer(children)
    final = sorted(zip(scores, pop), key=lambda t: -t[0])
    return Result(scorer.best, scorer.best_score, scorer.queries, scorer.history, final)
