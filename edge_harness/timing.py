"""Timing summaries. Same contract as ``mlsys_harness.timing``.

Every result carries p10, p50 and p90 so a write-up can say which one it
quoted. Edge latency is usually reported as p50 and p99 of a single call, so
``time_calls`` also returns p99 when there are enough samples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeResult:
    mode: str
    samples: int
    p10_s: float
    p50_s: float
    p90_s: float
    p99_s: float

    def as_dict(self) -> dict:
        return {"mode": self.mode, "samples": self.samples, "p10_s": self.p10_s,
                "p50_s": self.p50_s, "p90_s": self.p90_s, "p99_s": self.p99_s}


def percentile(samples: list[float], q: float) -> float:
    """Linear percentile. q is in [0, 1]. One sample returns that sample."""
    if not samples:
        raise ValueError("no samples")
    if not 0.0 <= q <= 1.0:
        raise ValueError(q)
    xs = sorted(samples)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize(samples: list[float], mode: str) -> TimeResult:
    if mode not in ("per_call", "batched"):
        raise ValueError(mode)
    return TimeResult(mode, len(samples), percentile(samples, 0.10), percentile(samples, 0.50),
                      percentile(samples, 0.90), percentile(samples, 0.99))


def time_calls(fn, warmup: int = 5, repeats: int = 20) -> TimeResult:
    """Host-clock per-call timing."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return summarize(samples, "per_call")
