"""MCUNetV2 patch-based inference: run the first stages tile by tile so their big activations never exist whole.

For a cut after ``features[:k]`` and a p x p split of that stage's output:

* each output tile is ceil(H_out / p) wide; the input tile it needs grows back through every
  conv by (tile - 1) * stride + kernel (the receptive-field halo);
* peak memory of the patch stage = the full stage output buffer (tiles are written into it)
  + the arena peak of one tile running through ``features[:k]``;
* peak of the rest = the arena peak of ``features[k:]`` and the head at full resolution;
* compute overhead = p^2 * MACs(one input tile) / MACs(full input), because halos are recomputed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .graph import trace
from .planners import arena_size, greedy_by_size


@dataclass(frozen=True)
class PatchResult:
    cut: int
    patches: int
    tile_in: int
    stage_out_bytes: int
    patch_stage_peak: int
    rest_peak: int
    peak: int
    mac_overhead: float  # MACs of the patched stage / MACs of that stage run whole
    net_mac_overhead: float  # same, for the whole network


def spatial_convs(modules: list[nn.Module]) -> list[tuple[int, int]]:
    """(kernel, stride) of every conv in execution order. 1x1 stride-1 convs do not change the extent."""
    out = []
    for m in modules:
        for sub in m.modules():
            if isinstance(sub, nn.Conv2d):
                out.append((sub.kernel_size[0], sub.stride[0]))
    return out


def input_extent(convs: list[tuple[int, int]], out_tile: int) -> int:
    size = out_tile
    for k, s in reversed(convs):
        size = (size - 1) * s + k
    return size


def macs(model: nn.Module, shape: tuple) -> int:
    total = 0

    def hook(m, _i, out):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            total += out.numel() * m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        elif isinstance(m, nn.Linear):
            total += out.numel() * m.in_features

    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model.eval()(torch.zeros(shape))
    for h in hooks:
        h.remove()
    return total


class _Tail(nn.Module):
    """features[k:] followed by whatever the model does after ``features``."""

    def __init__(self, features: nn.Sequential, head: nn.Module):
        super().__init__()
        self.features = features
        self.head = head

    def forward(self, x):
        return self.head(self.features(x))


def peak(model: nn.Module, shape: tuple) -> int:
    ts = trace(model, shape)
    return arena_size(ts, greedy_by_size(ts))


def patch_plan(features: nn.Sequential, head: nn.Module, resolution: int, cut: int, patches: int,
               in_ch: int = 3) -> PatchResult:
    stage = features[:cut]
    rest = _Tail(features[cut:], head)
    with torch.no_grad():
        y = stage.eval()(torch.zeros(1, in_ch, resolution, resolution))
    _, c_out, h_out, _ = y.shape
    stage_out = c_out * h_out * h_out
    rest_peak = peak(rest, tuple(y.shape))
    if patches == 1:
        whole = trace(_Tail(features, head), (1, in_ch, resolution, resolution))
        p = arena_size(whole, greedy_by_size(whole))
        return PatchResult(cut, 1, resolution, stage_out, p, p, p, 1.0, 1.0)
    tile_out = math.ceil(h_out / patches)
    tile_in = min(resolution, input_extent(spatial_convs(list(stage)), tile_out))
    patch_peak = stage_out + peak(stage, (1, in_ch, tile_in, tile_in))
    stage_macs = macs(stage, (1, in_ch, resolution, resolution))
    rest_macs = macs(rest, tuple(y.shape))
    patched_macs = patches ** 2 * macs(stage, (1, in_ch, tile_in, tile_in))
    return PatchResult(cut, patches, tile_in, stage_out, patch_peak, rest_peak, max(patch_peak, rest_peak),
                       patched_macs / stage_macs, (patched_macs + rest_macs) / (stage_macs + rest_macs))
