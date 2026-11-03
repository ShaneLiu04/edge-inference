"""Turn a bandwidth-versus-size sweep and two peak numbers into a roofline.

The sweep is noisy and the cache boundaries are soft (L2 is inclusive of the
working set of other cores, the last level is shared), so levels are found as
plateaus of the median over a window rather than by thresholds on single points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Level:
    name: str
    lo_bytes: int
    hi_bytes: int
    gbps: float


def plateaus(sizes: list[int], gbps: list[float], drop: float = 0.65) -> list[Level]:
    """Split the sweep wherever bandwidth falls below ``drop`` times the running plateau level.

    Returns levels from fastest (smallest working set) to slowest.
    """
    if len(sizes) != len(gbps) or not sizes:
        raise ValueError("sizes and gbps must be non-empty and equal length")
    order = np.argsort(sizes)
    s = np.asarray(sizes)[order]
    g = np.asarray(gbps, dtype=float)[order]
    groups: list[list[int]] = [[0]]
    for i in range(1, len(s)):
        level = float(np.median(g[groups[-1]]))
        if g[i] < drop * level:
            groups.append([i])
        else:
            groups[-1].append(i)
    names = ["L1", "L2", "L3", "DRAM"]
    out = []
    for k, idx in enumerate(groups):
        name = names[k] if k < len(names) - 1 else f"level{k}"
        if k == len(groups) - 1 and len(groups) > 1:
            name = "DRAM"
        out.append(Level(name, int(s[idx[0]]), int(s[idx[-1]]), float(np.median(g[idx]))))
    return out


def ridge(peak_gops: float, bw_gbps: float) -> float:
    """Arithmetic intensity (op/byte) where the compute and bandwidth roofs meet."""
    if bw_gbps <= 0:
        raise ValueError("bandwidth must be positive")
    return peak_gops / bw_gbps


def attainable(intensity: float | np.ndarray, peak_gops: float, bw_gbps: float) -> np.ndarray:
    return np.minimum(peak_gops, np.asarray(intensity, dtype=float) * bw_gbps)


def conv_intensity(macs: float, act_bytes: float, weight_bytes: float) -> float:
    """Ops per DRAM byte when activations and weights are each moved once (the best case)."""
    return 2.0 * macs / (act_bytes + weight_bytes)


def efficiency(gops: float, intensity: float, peak_gops: float, bw_gbps: float) -> float:
    """Fraction of the roofline actually reached."""
    return float(gops / attainable(intensity, peak_gops, bw_gbps))
