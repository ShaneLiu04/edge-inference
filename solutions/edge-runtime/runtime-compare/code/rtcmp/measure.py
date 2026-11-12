"""Quota parsing and per-call timing. No runtime dependencies."""

from __future__ import annotations

import time
from collections.abc import Callable


def cpu_quota_cores(text: str) -> float | None:
    """Cores granted by a cgroup v2 ``cpu.max`` file. ``max`` means uncapped."""
    parts = text.split()
    if len(parts) != 2 or parts[0] == "max":
        return None
    period = int(parts[1])
    if period <= 0:
        raise ValueError(text)
    return int(parts[0]) / period


def time_call(fn: Callable[[], object], repeats: int, warmup: int) -> dict:
    """Wall time includes cgroup stalls; process time does not.

    Under a quota below one core, wall time measures the quota. Process time
    measures the work, which is what a runtime comparison can report.
    """
    if repeats <= 0 or warmup < 0:
        raise ValueError((repeats, warmup))
    for _ in range(warmup):
        fn()
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    for _ in range(repeats):
        fn()
    return {
        "repeats": repeats,
        "wall_us": (time.perf_counter() - wall0) / repeats * 1e6,
        "cpu_us": (time.process_time() - cpu0) / repeats * 1e6,
    }


def agreement(reference, other) -> dict:
    """Max abs logit gap and how often the argmax matches. Both are (N, C)."""
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(other, dtype=np.float64)
    if ref.shape != got.shape:
        raise ValueError((ref.shape, got.shape))
    return {
        "logit_max_abs": float(np.max(np.abs(ref - got))),
        "argmax_agree": float(np.mean(ref.argmax(1) == got.argmax(1))),
    }


def combine(trials: list[dict], metrics: tuple[str, ...] = ("cpu_us_eval", "cpu_us", "wall_us")) -> dict:
    """First successful trial's row, with each timing replaced by the median over fresh-process trials.

    Accuracy and logits are deterministic, so they come from the first trial. Under a sub-core quota
    the same binary's CPU time moves by tens of percent between processes; min and max keep that visible.
    """
    import numpy as np

    ok = [t for t in trials if t.get("status") == "ok"]
    if not ok:
        return dict(trials[0])
    row = dict(ok[0])
    row.pop("trial", None)
    row["trials"] = len(ok)
    for metric in metrics:
        vals = sorted(t[metric] for t in ok if metric in t)
        if not vals:
            continue
        row[metric] = float(np.median(vals))
        row[f"{metric}_min"] = vals[0]
        row[f"{metric}_max"] = vals[-1]
    return row
