"""Flash bytes, compute counts and an MCU latency estimate for an int8 deployment.

Latency on a microcontroller is not measured here. It is estimated from MACs with a line fitted
to the three TinyEngine latencies MCUNet publishes for its VWW models on an STM32F746
(216 MHz Cortex-M7). The fit is only as good as the assumption that TinyEngine's throughput in
MACs per second is the same for these layer shapes; the residuals of the fit are reported with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

# MCUNet README, VWW models on STM32F746 with TinyEngine: (MACs in millions, latency in ms).
TINYENGINE_VWW = {"mcunet-vww0": (6.0, 27.0), "mcunet-vww1": (11.6, 51.0), "mcunet-vww2": (55.8, 234.0)}


def flash_bytes(model: nn.Module, per_channel: bool = True) -> dict:
    """int8 weights, int32 biases and one fp32 scale per output channel (per tensor if not per_channel).

    Run on a BN-folded model: every conv then has a bias. The 2-way classifier is counted the same way.
    """
    w = b = scales = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w += m.weight.numel()
            out = m.weight.shape[0]
            b += 4 * (out if m.bias is None else m.bias.numel())
            scales += 4 * (out if per_channel else 1) + 4  # weight scales + activation scale/zero-point
    return {"weights": w, "bias": b, "scales": scales, "total": w + b + scales}


def conv_macs(model: nn.Module, shape: tuple) -> int:
    total = 0

    def hook(m, _i, out):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            total += out.numel() * m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        elif isinstance(m, nn.Linear):
            total += out.numel() * m.in_features

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model.eval()(torch.zeros(shape))
    for h in hs:
        h.remove()
    return total


@dataclass(frozen=True)
class LatencyFit:
    ms_per_mmac: float
    intercept_ms: float
    max_rel_residual: float

    def __call__(self, mmacs: float) -> float:
        return self.intercept_ms + self.ms_per_mmac * mmacs


def fit_latency(points: dict[str, tuple[float, float]] = TINYENGINE_VWW, through_origin: bool = False) -> LatencyFit:
    x = np.array([p[0] for p in points.values()])
    y = np.array([p[1] for p in points.values()])
    if through_origin:
        a, b = float((x @ y) / (x @ x)), 0.0
    else:
        a, b = (float(v) for v in np.polyfit(x, y, 1))
    rel = np.abs((a * x + b) - y) / y
    return LatencyFit(a, b, float(rel.max()))
