"""BatchNorm folding for the MCUNet module family (and any module that holds ``conv`` then ``bn``)."""

from __future__ import annotations

import copy

import torch
from torch import nn


@torch.no_grad()
def fold_bn(model: nn.Module) -> nn.Module:
    """Return a copy where every ``conv -> bn`` pair is one conv with bias and the bn is an Identity.

    MCUNet layers keep the pair as attributes named ``conv`` and ``bn`` and run their children in
    registration order, so replacing ``bn`` in place keeps the forward unchanged.
    """
    model = copy.deepcopy(model).eval()
    n = 0
    for m in model.modules():
        conv, bn = m._modules.get("conv"), m._modules.get("bn")
        if not (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)):
            continue
        names = list(m._modules)
        if names.index("bn") < names.index("conv"):
            continue  # bn before weight: cannot fold into this conv
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bias = bn.bias - bn.running_mean * scale
        if conv.bias is not None:
            bias = bias + conv.bias * scale
        conv.weight.mul_(scale.reshape(-1, 1, 1, 1))
        conv.bias = nn.Parameter(bias)
        m._modules["bn"] = nn.Identity()
        n += 1
    model.folded_pairs = n
    return model


def remaining_bn(model: nn.Module) -> int:
    return sum(isinstance(m, nn.BatchNorm2d) for m in model.modules())
