"""Discrete-event model of a tiled kernel: DMA-in, compute, DMA-out, with a fixed number of buffers.

Each tile passes through stages in order; a stage is one engine (the DMA engine, the core) and handles
one tile at a time; a tile may enter stage s only when a buffer is free for it. With ``buffers=1`` the
stages serialize; with two or more, consecutive tiles overlap. The closed forms this checks:

  serial (1 buffer)   n * sum(stages)
  2 stages, 2 buffers n * max(c, m) + min(c, m)
  k stages, enough buffers, one engine per stage: sum(stages) + (n - 1) * max(stages)

When DMA-in and DMA-out share one engine (a single DMA channel), their times add on that engine.
"""

from __future__ import annotations


def simulate(n: int, stages: list[float], buffers: int, shared_dma: bool = False) -> float:
    """Finish time of the last tile. ``stages`` = [dma_in, compute] or [dma_in, compute, dma_out]."""
    if n <= 0:
        return 0.0
    k = len(stages)
    engine = list(range(k))
    if shared_dma and k == 3:
        engine = [0, 1, 0]
    engine_free = [0.0] * (max(engine) + 1)
    done = [[0.0] * k for _ in range(n)]
    # A buffer is held from the start of DMA-in until the end of the last stage of that tile.
    buffer_free = [0.0] * buffers
    if shared_dma and k == 3:
        # DMA engine order: in(0), in(1), out(0), in(2), out(1), ... the usual ping-pong schedule.
        dma_order = []
        for t in range(n):
            dma_order.append((t, 0))
            if t >= 1:
                dma_order.append((t - 1, 2))
        dma_order.append((n - 1, 2))
        comp_order = [(t, 1) for t in range(n)]
        events = []
        i = j = 0
        # Merge by dependency: compute(t) after in(t); out(t) after compute(t).
        while i < len(dma_order) or j < len(comp_order):
            if i < len(dma_order) and (dma_order[i][1] == 0 or j > dma_order[i][0]):
                events.append(dma_order[i])
                i += 1
            else:
                events.append(comp_order[j])
                j += 1
    else:
        # One engine per stage, tiles in order on every engine: tile-major order respects every dependency.
        events = [(t, s) for t in range(n) for s in range(k)]
    for t, s in events:
        ready = done[t][s - 1] if s > 0 else buffer_free[t % buffers]
        start = max(ready, engine_free[engine[s]])
        done[t][s] = start + stages[s]
        engine_free[engine[s]] = done[t][s]
        if s == k - 1:
            buffer_free[t % buffers] = done[t][s]
    return max(done[t][k - 1] for t in range(n))


def serial_time(n: int, stages: list[float]) -> float:
    return n * sum(stages)


def double_buffer_time(n: int, c: float, m: float) -> float:
    return n * max(c, m) + min(c, m)


def pipelined_time(n: int, stages: list[float]) -> float:
    return sum(stages) + (n - 1) * max(stages)
