"""Static arena planners: assign every buffer an offset so buffers alive at the same time never overlap.

* ``naive``           every buffer gets its own range; arena = sum of sizes.
* ``first_fit``       buffers in birth order, each at the lowest offset that fits.
* ``greedy_by_size``  buffers largest first, each at the lowest offset that fits. This is the
                      order TFLite Micro's GreedyMemoryPlanner uses.
* ``lower_bound``     max over time of live bytes. No planner can beat it; the gap to it is
                      fragmentation.
"""

from __future__ import annotations

from .graph import Tensor


def lower_bound(tensors: list[Tensor]) -> int:
    events = sorted({t.birth for t in tensors})
    return max(sum(t.size for t in tensors if t.birth <= s <= t.death) for s in events)


def _place(tensors: list[Tensor], order: list[Tensor]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    placed: list[Tensor] = []
    for t in order:
        busy = sorted((offsets[p.name], offsets[p.name] + p.size) for p in placed if p.overlaps(t))
        off = 0
        for lo, hi in busy:
            if off + t.size <= lo:
                break
            off = max(off, hi)
        offsets[t.name] = off
        placed.append(t)
    return offsets


def naive(tensors: list[Tensor]) -> dict[str, int]:
    offsets, off = {}, 0
    for t in tensors:
        offsets[t.name] = off
        off += t.size
    return offsets


def first_fit(tensors: list[Tensor]) -> dict[str, int]:
    return _place(tensors, sorted(tensors, key=lambda t: (t.birth, -t.size)))


def greedy_by_size(tensors: list[Tensor]) -> dict[str, int]:
    return _place(tensors, sorted(tensors, key=lambda t: (-t.size, t.birth)))


PLANNERS = {"naive": naive, "first_fit": first_fit, "greedy_by_size": greedy_by_size}


def arena_size(tensors: list[Tensor], offsets: dict[str, int]) -> int:
    return max(offsets[t.name] + t.size for t in tensors)


def validate(tensors: list[Tensor], offsets: dict[str, int]) -> None:
    """Raise if two buffers that are alive at the same step share a byte."""
    for i, a in enumerate(tensors):
        for b in tensors[i + 1:]:
            if not a.overlaps(b):
                continue
            a0, b0 = offsets[a.name], offsets[b.name]
            if a0 < b0 + b.size and b0 < a0 + a.size:
                raise AssertionError(f"{a.name} [{a0}, {a0 + a.size}) overlaps {b.name} [{b0}, {b0 + b.size})")
