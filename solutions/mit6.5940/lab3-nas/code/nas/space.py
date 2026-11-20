"""The Lab 3 design space: the OFA-MCUNet supernet built with ``base_stage_width="mcunet384"``.

A subnet is the dict the supernet's ``set_active_subnet`` takes plus ``image_size``:
``wid`` (index into WIDTHS), ``ks`` and ``e`` (one entry per elastic block, 20 of them)
and ``d`` (extra depth per stage, 6 stages). The fused first block is not elastic.
"""

from __future__ import annotations

import copy
import random

import numpy as np

RESOLUTIONS = (96, 112, 128, 144, 160)
WIDTHS = (0.5, 0.75, 1.0)
KERNELS = (3, 5, 7)
EXPANDS = (3, 4, 6)
DEPTHS = (0, 1, 2)
BASE_DEPTH = (1, 2, 2, 2, 2, 1)
STAGE_BLOCKS = (3, 4, 4, 4, 4, 1)
N_BLOCKS = sum(STAGE_BLOCKS)
N_STAGES = len(STAGE_BLOCKS)
# The 50k accuracy dataset was collected with the supernet's own sampler, which draws
# wid from range(len(WIDTHS) - 1): width 1.0 never appears, so the predictor cannot rank it.
PREDICTOR_WIDTHS = (0, 1)

BLOCK_STAGE = tuple(s for s, n in enumerate(STAGE_BLOCKS) for _ in range(n))
BLOCK_POS = tuple(i for n in STAGE_BLOCKS for i in range(n))

_R0 = 0
_W0 = _R0 + len(RESOLUTIONS)
_K0 = _W0 + len(WIDTHS)
_E0 = _K0 + N_BLOCKS * len(KERNELS)
N_FEATURES = _E0 + N_BLOCKS * len(EXPANDS)


def stage_depth(cfg: dict, stage: int) -> int:
    return min(STAGE_BLOCKS[stage], BASE_DEPTH[stage] + cfg["d"][stage])


def active(cfg: dict) -> list[bool]:
    return [BLOCK_POS[i] < stage_depth(cfg, BLOCK_STAGE[i]) for i in range(N_BLOCKS)]


def validate(cfg: dict) -> None:
    assert cfg["image_size"] in RESOLUTIONS, cfg["image_size"]
    assert cfg["wid"] in range(len(WIDTHS)), cfg["wid"]
    assert len(cfg["ks"]) == N_BLOCKS and set(cfg["ks"]) <= set(KERNELS)
    assert len(cfg["e"]) == N_BLOCKS and set(cfg["e"]) <= set(EXPANDS)
    assert len(cfg["d"]) == N_STAGES and set(cfg["d"]) <= set(DEPTHS)


def sample(rng: random.Random, widths=PREDICTOR_WIDTHS, resolutions=RESOLUTIONS) -> dict:
    return {
        "wid": rng.choice(widths),
        "ks": [rng.choice(KERNELS) for _ in range(N_BLOCKS)],
        "e": [rng.choice(EXPANDS) for _ in range(N_BLOCKS)],
        "d": [rng.choice(DEPTHS) for _ in range(N_STAGES)],
        "image_size": rng.choice(resolutions),
    }


def uniform(wid: int, k: int, e: int, d: int, image_size: int) -> dict:
    return {"wid": wid, "ks": [k] * N_BLOCKS, "e": [e] * N_BLOCKS, "d": [d] * N_STAGES, "image_size": image_size}


def smallest() -> dict:
    return uniform(0, min(KERNELS), min(EXPANDS), min(DEPTHS), min(RESOLUTIONS))


def largest(wid: int = len(WIDTHS) - 1) -> dict:
    return uniform(wid, max(KERNELS), max(EXPANDS), max(DEPTHS), max(RESOLUTIONS))


def mutate(cfg: dict, rng: random.Random, arch_p: float, res_p: float,
           widths=PREDICTOR_WIDTHS, resolutions=RESOLUTIONS) -> dict:
    out = copy.deepcopy(cfg)
    if rng.random() < res_p:
        out["image_size"] = rng.choice(resolutions)
    if rng.random() < arch_p:
        out["wid"] = rng.choice(widths)
    for key, choices in (("ks", KERNELS), ("e", EXPANDS), ("d", DEPTHS)):
        out[key] = [rng.choice(choices) if rng.random() < arch_p else v for v in out[key]]
    return out


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    out = {}
    for key in ("wid", "image_size"):
        out[key] = rng.choice((a[key], b[key]))
    for key in ("ks", "e", "d"):
        out[key] = [rng.choice(pair) for pair in zip(a[key], b[key])]
    return out


def encode(cfg: dict) -> np.ndarray:
    """One-hot features in the layout of the course ``MCUNetArchEncoder`` (r, w, then k and e per block).

    Kernel and expand slots of skipped blocks stay zero, so two configs that build the same network
    get the same vector.
    """
    f = np.zeros(N_FEATURES, dtype=np.float32)
    f[_R0 + RESOLUTIONS.index(cfg["image_size"])] = 1
    f[_W0 + cfg["wid"]] = 1
    for i, on in enumerate(active(cfg)):
        if on:
            f[_K0 + 3 * i + KERNELS.index(cfg["ks"][i])] = 1
            f[_E0 + 3 * i + EXPANDS.index(cfg["e"][i])] = 1
    return f


def key(cfg: dict) -> bytes:
    return encode(cfg).tobytes()
