"""Fold BN into the whole network and simulate the integer graph.

Quantization points match the tensors an integer runtime materializes:

* ``input``                    the image, before the stem
* every conv-BN-act block      its output (after ReLU6 if present)
* every residual block         the output of the add
* ``pool``                     the global-average-pool output, i.e. the FC input

Weights of every Conv2d / Linear are replaced through a parametrization, so the
same object serves PTQ evaluation (round to nearest) and QAT (straight-through
gradient). Biases stay in float: in the integer graph they are int32 with scale
``s_w * s_x``, whose rounding error is below anything measured here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from .model import InvertedResidual
from .quant import CALIBRATORS, act_qparams, fake_quant_act, fake_quant_weight, ste_act, ste_weight

SAMPLE_PER_POINT = 1_000_000


def fold_model(model: nn.Module) -> nn.Module:
    """Return a copy with every Conv2d -> BatchNorm2d pair folded into the conv."""
    model = copy.deepcopy(model).eval()
    for module in model.modules():
        if not isinstance(module, nn.Sequential):
            continue
        children = list(module.children())
        for i in range(len(children) - 1):
            conv, bn = children[i], children[i + 1]
            if not (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)):
                continue
            inv = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
            bias = conv.bias.detach() if conv.bias is not None else torch.zeros_like(bn.running_mean)
            conv.weight.data = conv.weight.detach() * inv.reshape(-1, 1, 1, 1)
            conv.bias = nn.Parameter((bias - bn.running_mean) * inv + bn.bias.detach())
            module[i + 1] = nn.Identity()
    return model


def weight_layers(model: nn.Module) -> dict[str, nn.Module]:
    return {name: m for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))}


def is_depthwise(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d) and module.groups > 1 and module.groups == module.in_channels


def act_points(model: nn.Module) -> dict[str, nn.Module]:
    """Modules whose output is a materialized tensor in the integer graph."""
    points: dict[str, nn.Module] = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Sequential) and any(isinstance(c, nn.Conv2d) for c in m.children()) \
                and not any(isinstance(c, nn.Sequential) for c in m.children()):
            points[name] = m
        elif isinstance(m, InvertedResidual) and m.residual:
            points[name] = m
    points["pool"] = model.pool
    return points


class _WeightQuant(nn.Module):
    def __init__(self, bits: int, per_channel: bool, ste: bool):
        super().__init__()
        self.bits, self.per_channel, self.ste = bits, per_channel, ste

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if self.ste:
            return ste_weight(w, self.bits, self.per_channel)
        return fake_quant_weight(w, self.bits, self.per_channel)


@dataclass
class QuantConfig:
    w_bits: int | None = 8
    w_per_channel: bool = True
    a_bits: int | None = 8
    calib: str = "minmax"
    # layer name -> weight bits; None keeps that layer in float
    w_overrides: dict[str, int | None] = field(default_factory=dict)
    # activation point name -> bits; None keeps that point in float
    a_overrides: dict[str, int | None] = field(default_factory=dict)
    # depthwise layers only: override per-channel choice (None = follow w_per_channel)
    dw_per_channel: bool | None = None


class QuantSim:
    """A folded copy of a model with fake quantization applied in place."""

    def __init__(self, folded: nn.Module, cfg: QuantConfig, ste: bool = False):
        self.model = copy.deepcopy(folded)
        self.cfg = cfg
        self.ste = ste
        self.ranges: dict[str, tuple[float, float]] = {}
        self._hooks: list = []
        for name, layer in weight_layers(self.model).items():
            bits = cfg.w_overrides.get(name, cfg.w_bits)
            if bits is None:
                continue
            per_channel = cfg.w_per_channel
            if cfg.dw_per_channel is not None and is_depthwise(layer):
                per_channel = cfg.dw_per_channel
            parametrize.register_parametrization(layer, "weight", _WeightQuant(bits, per_channel, ste))

    # -- activation points
    def _points(self) -> dict[str, nn.Module]:
        pts = {"input": self.model.features[0]}
        pts.update(act_points(self.model))
        return pts

    def _bits(self, name: str) -> int | None:
        return self.cfg.a_overrides.get(name, self.cfg.a_bits)

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor, batch: int = 256, seed: int = 0) -> None:
        """Collect activations on ``x`` with weights already quantized, then set ranges."""
        rng = np.random.default_rng(seed)
        samples: dict[str, list[np.ndarray]] = {}

        def grab(name):
            def fn(_m, args, out=None):
                t = args[0] if out is None else out
                arr = t.detach().reshape(-1).numpy()
                if arr.size > SAMPLE_PER_POINT // 8:
                    arr = rng.choice(arr, SAMPLE_PER_POINT // 8, replace=False)
                samples.setdefault(name, []).append(arr)
            return fn

        hooks = []
        for name, m in self._points().items():
            if name == "input":
                hooks.append(m.register_forward_pre_hook(grab(name)))
            else:
                hooks.append(m.register_forward_hook(grab(name)))
        self.model.eval()
        for i in range(0, len(x), batch):
            self.model(x[i:i + batch])
        for h in hooks:
            h.remove()
        calib = CALIBRATORS[self.cfg.calib]
        for name, parts in samples.items():
            bits = self._bits(name)
            if bits is None:
                continue
            values = np.concatenate(parts)
            if values.size > SAMPLE_PER_POINT:
                values = rng.choice(values, SAMPLE_PER_POINT, replace=False)
            self.ranges[name] = calib(values, bits)
        self.install()

    def install(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        quant = ste_act if self.ste else fake_quant_act
        for name, m in self._points().items():
            bits = self._bits(name)
            if bits is None or name not in self.ranges:
                continue
            scale, zp = act_qparams(*self.ranges[name], bits)
            if name == "input":
                def pre(_m, args, s=scale, z=zp, b=bits):
                    return (quant(args[0], s, z, b),)
                self._hooks.append(m.register_forward_pre_hook(pre))
            else:
                def post(_m, _args, out, s=scale, z=zp, b=bits):
                    return quant(out, s, z, b)
                self._hooks.append(m.register_forward_hook(post))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@torch.no_grad()
def accuracy(model, x: torch.Tensor, y: torch.Tensor, batch: int = 1000) -> float:
    if isinstance(model, nn.Module):
        model.eval()
    else:
        model.model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        correct += (model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
    return correct / len(x)


@torch.no_grad()
def execution_order(model: nn.Module, x: torch.Tensor, names: list[str]) -> list[str]:
    """``names`` sorted by when each module finishes in a forward pass.

    ``named_modules`` lists a residual block before its body; its output is produced after it.
    """
    modules = dict(model.named_modules())
    done: list[str] = []
    hooks = [(model.pool if n == "pool" else modules[n]).register_forward_hook(
        lambda _m, _a, _o, n=n: done.append(n)) for n in names]
    model.eval()
    model(x)
    for h in hooks:
        h.remove()
    return done


@torch.no_grad()
def point_outputs(model: nn.Module, x: torch.Tensor, names: list[str]) -> dict[str, torch.Tensor]:
    """Outputs of named activation points (after any installed fake quant)."""
    modules = dict(model.named_modules())
    out: dict[str, torch.Tensor] = {}
    hooks = []
    for name in names:
        target = model.pool if name == "pool" else modules[name]
        hooks.append(target.register_forward_hook(lambda _m, _a, o, n=name: out.__setitem__(n, o.detach().clone())))
    model.eval()
    logits = model(x)
    for h in hooks:
        h.remove()
    out["logits"] = logits
    return out
