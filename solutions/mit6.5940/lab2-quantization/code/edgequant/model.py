"""A small MobileNetV2 for 32x32 inputs.

Same building blocks as MobileNetV2 (pointwise expand, 3x3 depthwise, linear
pointwise project, residual when shapes match, ReLU6), scaled down so it trains
on a laptop CPU. Depthwise layers followed by BN are exactly the layers where
per-tensor weight scales are known to break after folding.
"""

from __future__ import annotations

import torch
from torch import nn

# (expansion t, output channels c, repeats n, first stride s)
CONFIG = ((1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 2, 2), (6, 64, 2, 2), (6, 96, 1, 1))
STEM = 16
STEM_STRIDE = 2
HEAD = 256


def conv_bn(cin: int, cout: int, k: int, stride: int, groups: int = 1, act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
    ]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, t: int):
        super().__init__()
        hidden = cin * t
        layers: list[nn.Module] = []
        if t != 1:
            layers.append(conv_bn(cin, hidden, 1, 1))
        layers.append(conv_bn(hidden, hidden, 3, stride, groups=hidden))
        layers.append(conv_bn(hidden, cout, 1, 1, act=False))
        self.body = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        return x + y if self.residual else y


class MicroMobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        layers: list[nn.Module] = [conv_bn(3, STEM, 3, STEM_STRIDE)]
        cin = STEM
        for t, c, n, s in CONFIG:
            for i in range(n):
                layers.append(InvertedResidual(cin, c, s if i == 0 else 1, t))
                cin = c
        layers.append(conv_bn(cin, HEAD, 1, 1))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(HEAD, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.pool(self.features(x)), 1))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, shape=(1, 3, 32, 32)) -> int:
    """Multiply-accumulates of conv and linear layers, by forward hooks."""
    total = 0

    def conv_hook(m: nn.Conv2d, _inp, out):
        nonlocal total
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        total += out.numel() * k

    def fc_hook(m: nn.Linear, _inp, out):
        nonlocal total
        total += out.numel() * m.in_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(fc_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(shape, device=next(model.parameters()).device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return total
