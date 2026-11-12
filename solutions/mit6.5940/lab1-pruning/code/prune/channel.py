"""Uniform channel pruning of the VGG backbone, with importance-based channel sorting first."""

from __future__ import annotations

import copy

import torch
from torch import nn


def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    return int(round(channels * (1.0 - prune_ratio)))


def _convs_and_bns(model: nn.Module) -> tuple[list[nn.Conv2d], list[nn.BatchNorm2d]]:
    convs = [m for m in model.backbone if isinstance(m, nn.Conv2d)]
    bns = [m for m in model.backbone if isinstance(m, nn.BatchNorm2d)]
    assert len(convs) == len(bns)
    return convs, bns


def _keep_bn(bn: nn.BatchNorm2d, idx: torch.Tensor) -> None:
    bn.weight = nn.Parameter(bn.weight.detach()[idx].clone())
    bn.bias = nn.Parameter(bn.bias.detach()[idx].clone())
    bn.running_mean = bn.running_mean[idx].clone()
    bn.running_var = bn.running_var[idx].clone()
    bn.num_features = len(idx)


@torch.no_grad()
def channel_prune(model: nn.Module, prune_ratio) -> nn.Module:
    """Keep the first `n_keep` output channels of every conv except the last (which feeds the classifier).

    `prune_ratio` is a float (uniform) or one ratio per pruned conv. Returns a pruned copy.
    """
    model = copy.deepcopy(model)
    convs, bns = _convs_and_bns(model)
    ratios = [prune_ratio] * (len(convs) - 1) if isinstance(prune_ratio, (int, float)) else list(prune_ratio)
    assert len(ratios) == len(convs) - 1
    for i, r in enumerate(ratios):
        prev, bn, nxt = convs[i], bns[i], convs[i + 1]
        n_keep = get_num_channels_to_keep(prev.out_channels, r)
        idx = torch.arange(n_keep, device=prev.weight.device)
        prev.weight = nn.Parameter(prev.weight.detach()[idx].clone())
        prev.out_channels = n_keep
        _keep_bn(bn, idx)
        nxt.weight = nn.Parameter(nxt.weight.detach()[:, idx].clone())
        nxt.in_channels = n_keep
    return model


def get_input_channel_importance(weight: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of each input-channel slice `weight[:, c]`."""
    return weight.detach().transpose(0, 1).reshape(weight.shape[1], -1).norm(dim=1)


@torch.no_grad()
def apply_channel_sorting(model: nn.Module) -> nn.Module:
    """Reorder every intermediate channel by the next conv's input importance, descending.

    The permutation is applied consistently to conv(out), BN and next conv(in), so the network
    computes the same function; only the order changes, which makes `channel_prune` keep the
    most important channels.
    """
    model = copy.deepcopy(model)
    convs, bns = _convs_and_bns(model)
    for i in range(len(convs) - 1):
        prev, bn, nxt = convs[i], bns[i], convs[i + 1]
        order = torch.argsort(get_input_channel_importance(nxt.weight), descending=True)
        prev.weight.copy_(prev.weight[order])
        for t in (bn.weight, bn.bias, bn.running_mean, bn.running_var):
            t.copy_(t[order])
        nxt.weight.copy_(nxt.weight[:, order])
    return model


def count_macs(model: nn.Module, x: torch.Tensor) -> int:
    """Multiply-accumulates of every Conv2d and Linear for one forward pass of `x` (per sample)."""
    total = 0

    def conv_hook(m: nn.Conv2d, inp, out):
        nonlocal total
        k = m.kernel_size[0] * m.kernel_size[1]
        total += out[0].numel() * (m.in_channels // m.groups) * k

    def fc_hook(m: nn.Linear, inp, out):
        nonlocal total
        total += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(fc_hook))
    with torch.no_grad():
        model(x[:1])
    for h in hooks:
        h.remove()
    return total
