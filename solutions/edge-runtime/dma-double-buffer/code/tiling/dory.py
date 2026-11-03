"""Dory-style tile search for a conv layer on an MCU with a small L1 and a DMA from L2.

The layer is tiled over output rows (th), output channels (tk) and input channels (tc). One tile needs
the input slab, the weight block and the output slab in L1; with double buffering every buffer exists
twice. For each feasible tile the cost is the pipeline time of its tiles:

  n_tiles * max(t_dma, t_compute) + min(t_dma, t_compute)    (double buffered)
  n_tiles * (t_dma + t_compute)                              (single buffer)

with t_dma = bytes_moved / dma_bytes_per_cycle + setup, t_compute = MACs / macs_per_cycle + setup. Splitting
input channels forces partial sums: the int32 output tile must stay resident and is written once.
This is the same objective Dory states as "maximize tile size subject to L1, prefer channel tiles that
are multiples of the SIMD width", reduced to the two terms that decide it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Conv:
    h: int        # output rows (= cols)
    c: int        # input channels
    k: int        # output channels
    r: int = 3    # kernel
    stride: int = 1
    depthwise: bool = False  # c == k, each output channel reads one input channel

    @property
    def macs(self) -> int:
        return self.h * self.h * self.k * (1 if self.depthwise else self.c) * self.r * self.r


@dataclass(frozen=True)
class Target:
    name: str
    l1_bytes: int
    dma_bytes_per_cycle: float
    dma_setup_cycles: float
    macs_per_cycle: float
    compute_setup_cycles: float  # kernel call, pointer setup, cluster fork: paid once per tile


# Rough shapes of real classes of part: a PULP cluster (GAP8-like) fed from on-chip L2, the same
# cluster's L2 fed from off-chip HyperRAM (~1 byte/cycle), and a Cortex-M7 with DTCM.
TARGETS = {
    "pulp-cluster": Target("pulp-cluster", 64 * 1024, 8.0, 30.0, 16.0, 300.0),
    "pulp-l3-hyperram": Target("pulp-l3-hyperram", 64 * 1024, 1.0, 100.0, 16.0, 300.0),
    "cortex-m7": Target("cortex-m7", 128 * 1024, 4.0, 20.0, 2.0, 100.0),
}


@dataclass(frozen=True)
class Tile:
    th: int
    tk: int
    tc: int
    l1_bytes: int
    n_tiles: int
    dma_cycles: float
    compute_cycles: float
    total_cycles: float


def _tile_cost(layer: Conv, t: Target, th: int, tk: int, tc: int, buffers: int) -> Tile | None:
    dw = layer.depthwise
    in_rows = (th - 1) * layer.stride + layer.r
    in_ch = tk if dw else tc                                                    # depthwise: tile channels together
    in_bytes = in_rows * ((layer.h - 1) * layer.stride + layer.r) * in_ch       # int8 input slab
    w_bytes = tk * (1 if dw else tc) * layer.r * layer.r                        # int8 weights
    split = not dw and tc < layer.c
    out_bytes = th * layer.h * tk * (4 if split else 1)                         # int32 partials if split
    l1 = buffers * (in_bytes + w_bytes) + (1 if split else buffers) * out_bytes
    if l1 > t.l1_bytes:
        return None
    n = math.ceil(layer.h / th) * math.ceil(layer.k / tk) * (1 if dw else math.ceil(layer.c / tc))
    dma = (in_bytes + w_bytes) / t.dma_bytes_per_cycle + t.dma_setup_cycles
    comp = th * layer.h * tk * (1 if dw else tc) * layer.r * layer.r / t.macs_per_cycle + t.compute_setup_cycles
    if buffers >= 2:
        total = n * max(dma, comp) + min(dma, comp)
    else:
        total = n * (dma + comp)
    # Output slabs go back once per (th, tk) tile.
    total += math.ceil(layer.h / th) * math.ceil(layer.k / tk) * (th * layer.h * tk / t.dma_bytes_per_cycle)
    return Tile(th, tk, tc, l1, n, dma, comp, total)


def search(layer: Conv, t: Target, buffers: int = 2) -> list[Tile]:
    """Every feasible tile, best first."""
    out = []
    for th in range(1, layer.h + 1):
        for tk in range(1, layer.k + 1):
            if layer.k % tk:
                continue
            for tc in ([tk] if layer.depthwise else range(1, layer.c + 1)):
                if layer.c % tc:
                    continue
                tile = _tile_cost(layer, t, th, tk, tc, buffers)
                if tile is not None:
                    out.append(tile)
    return sorted(out, key=lambda x: (x.total_cycles, x.n_tiles))


def ideal_cycles(layer: Conv, t: Target) -> float:
    """Compute-only time: the floor any tiling approaches when DMA hides completely."""
    return layer.macs / t.macs_per_cycle
