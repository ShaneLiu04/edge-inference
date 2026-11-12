"""Linear (affine) quantization r = S (q - Z), and an integer-only VGG.

Weights: per-output-channel symmetric int8 (Z = 0). Activations: per-tensor asymmetric int8 from
calibration min/max; the ReLU disappears because the output zero point sits at the clamp floor.
Bias: int32 at scale S_in * S_w, Z = 0. The int32 accumulator is computed in float64, which is exact
for |acc| < 2^53 (here at most 255 * 127 * 4608 < 2^28).
"""

from __future__ import annotations

import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


def get_quantized_range(bits: int) -> tuple[int, int]:
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def linear_quantize(fp: torch.Tensor, bits: int, scale, zero_point, dtype=torch.int8) -> torch.Tensor:
    qmin, qmax = get_quantized_range(bits)
    return torch.clamp(torch.round(fp / scale) + zero_point, qmin, qmax).to(dtype)


def get_scale_and_zero_point(rmin: float, rmax: float, bits: int) -> tuple[float, int]:
    qmin, qmax = get_quantized_range(bits)
    rmin, rmax = min(rmin, 0.0), max(rmax, 0.0)
    scale = (rmax - rmin) / (qmax - qmin) or 1.0
    zp = int(round(qmin - rmin / scale))
    return scale, max(qmin, min(qmax, zp))


def per_channel_symmetric(weight: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (q, scale[co]) with scale = max|w| / qmax along every output channel."""
    _, qmax = get_quantized_range(bits)
    amax = weight.detach().abs().flatten(1).amax(1).clamp(min=1e-12)
    scale = amax / qmax
    shape = (-1,) + (1,) * (weight.dim() - 1)
    return linear_quantize(weight, bits, scale.view(shape), 0), scale


def quantize_bias(bias: torch.Tensor, w_scale: torch.Tensor, in_scale: float) -> torch.Tensor:
    return torch.round(bias / (w_scale * in_scale)).to(torch.int32)


def _requantize(acc: torch.Tensor, in_scale: float, w_scale: torch.Tensor, out_scale: float,
                out_zp: int, bits: int, channel_dim: int) -> torch.Tensor:
    qmin, qmax = get_quantized_range(bits)
    shape = [1] * acc.dim()
    shape[channel_dim] = -1
    m = (in_scale * w_scale.double() / out_scale).view(shape)
    return torch.clamp(torch.round(acc * m) + out_zp, qmin, qmax).to(torch.int8)


def quantized_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, in_zp: int, in_scale: float,
                     w_scale: torch.Tensor, out_scale: float, out_zp: int, bits: int = 8) -> torch.Tensor:
    acc = F.linear(x.double() - in_zp, w.double())
    if b is not None:
        acc = acc + b.double()
    return _requantize(acc, in_scale, w_scale, out_scale, out_zp, bits, channel_dim=1)


def quantized_conv2d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, in_zp: int, in_scale: float,
                     w_scale: torch.Tensor, out_scale: float, out_zp: int, stride, padding,
                     bits: int = 8) -> torch.Tensor:
    """Padding uses the input zero point (the real value 0), as an int8 runtime pads."""
    xd = x.double() - in_zp
    acc = F.conv2d(xd, w.double(), None, stride, padding)
    if b is not None:
        acc = acc + b.double().view(1, -1, 1, 1)
    return _requantize(acc, in_scale, w_scale, out_scale, out_zp, bits, channel_dim=1)


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    fused = copy.deepcopy(conv)
    std = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight / std
    fused.weight = nn.Parameter(conv.weight.detach() * gamma.view(-1, 1, 1, 1))
    b0 = conv.bias.detach() if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.bias = nn.Parameter(bn.bias.detach() + (b0 - bn.running_mean) * gamma)
    return fused


def fuse_vgg(model: nn.Module) -> nn.Module:
    """conv-bn-relu triples -> conv(with bias)-relu, same forward up to float rounding."""
    model = copy.deepcopy(model).eval()
    layers = list(model.backbone.named_children())
    out = []
    i = 0
    while i < len(layers):
        name, m = layers[i]
        if isinstance(m, nn.Conv2d) and i + 1 < len(layers) and isinstance(layers[i + 1][1], nn.BatchNorm2d):
            out.append((name, fuse_conv_bn(m, layers[i + 1][1])))
            i += 2
        else:
            out.append((name, m))
            i += 1
    model.backbone = nn.Sequential(OrderedDict(out))
    return model


# Input: x in [0, 1] -> q = round(255 x) - 128, i.e. S = 1/255, Z = -128 (exact for 8-bit images).
INPUT_SCALE, INPUT_ZP = 1.0 / 255.0, -128


@torch.no_grad()
def calibrate(fused: nn.Module, x: torch.Tensor) -> dict[str, tuple[float, float]]:
    """min/max of every conv output after its ReLU, and of the logits."""
    ranges: dict[str, tuple[float, float]] = {}
    hooks = []
    mods = list(fused.backbone.named_children())
    for j, (name, m) in enumerate(mods):
        if isinstance(m, nn.Conv2d):
            nxt = mods[j + 1][1] if j + 1 < len(mods) else None
            tap = nxt if isinstance(nxt, nn.ReLU) else m

            def hook(_m, _i, out, name=name):
                lo, hi = ranges.get(name, (float("inf"), float("-inf")))
                ranges[name] = (min(lo, out.min().item()), max(hi, out.max().item()))
            hooks.append(tap.register_forward_hook(hook))

    def logit_hook(_m, _i, out):
        lo, hi = ranges.get("classifier", (float("inf"), float("-inf")))
        ranges["classifier"] = (min(lo, out.min().item()), max(hi, out.max().item()))
    hooks.append(fused.classifier.register_forward_hook(logit_hook))
    fused(x)
    for h in hooks:
        h.remove()
    return ranges


class QuantizedVGG(nn.Module):
    """Integer-only forward: int8 in, int8 logits out; every tensor between layers is int8."""

    def __init__(self, fused: nn.Module, ranges: dict[str, tuple[float, float]], bits: int = 8):
        super().__init__()
        self.bits = bits
        self.ops: list[tuple] = []
        in_scale, in_zp = INPUT_SCALE, INPUT_ZP
        for name, m in fused.backbone.named_children():
            if isinstance(m, nn.Conv2d):
                q, ws = per_channel_symmetric(m.weight, bits)
                out_scale, out_zp = get_scale_and_zero_point(*ranges[name], bits)
                self.ops.append(("conv", q, quantize_bias(m.bias, ws, in_scale), in_zp, in_scale, ws,
                                 out_scale, out_zp, m.stride, m.padding))
                in_scale, in_zp = out_scale, out_zp
            elif isinstance(m, nn.MaxPool2d):
                self.ops.append(("maxpool", m.kernel_size, m.stride))
        q, ws = per_channel_symmetric(fused.classifier.weight, bits)
        out_scale, out_zp = get_scale_and_zero_point(*ranges["classifier"], bits)
        self.head = (q, quantize_bias(fused.classifier.bias, ws, in_scale), in_zp, in_scale, ws, out_scale, out_zp)
        self.logit_scale, self.logit_zp = out_scale, out_zp

    def forward(self, x_int8: torch.Tensor) -> torch.Tensor:
        x = x_int8
        for op in self.ops:
            if op[0] == "conv":
                _, w, b, zin, sin, ws, sout, zout, stride, pad = op
                x = quantized_conv2d(x, w, b, zin, sin, ws, sout, zout, stride, pad, self.bits)
            else:
                x = F.max_pool2d(x.float(), op[1], op[2]).to(torch.int8)
        # Global average pool on int8: mean in the integer domain (same scale and zero point), rounded.
        x = torch.round(x.double().mean(dim=(2, 3))).to(torch.int8)
        w, b, zin, sin, ws, sout, zout = self.head
        return quantized_linear(x, w, b, zin, sin, ws, sout, zout, self.bits)

    def size_bits(self) -> int:
        total = 0
        for op in self.ops:
            if op[0] == "conv":
                total += op[1].numel() * self.bits + op[2].numel() * 32 + op[5].numel() * 32
        w, b, *_rest, ws, _s, _z = self.head
        return total + w.numel() * self.bits + b.numel() * 32 + ws.numel() * 32


def quantize_input(x01: torch.Tensor) -> torch.Tensor:
    return (torch.round(x01 * 255) - 128).to(torch.int8)
