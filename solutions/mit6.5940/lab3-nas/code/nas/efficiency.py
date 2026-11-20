"""Closed-form MACs and peak activation memory of an OFA-MCUNet subnet, without building it.

The layer list follows the supernet's construction: a 3x3 stride-2 stem, a fused first block
(one 1x1 projection, no shortcut), six stages of inverted residual blocks, a 1x1 feature mix
to 384 channels and a 2-way classifier. Every count is in elements, which is bytes at int8.

Two peak rules:

* ``lab``: the course ``count_peak_activation_size``. Per layer, input + output; inside a
  block with a shortcut the block input is added only while the depthwise conv runs.
* ``residual``: the block input is also held while the projection conv runs, because the
  add needs it after the projection. This is what an arena actually has to keep.

Two MACs counts:

* ``conv_macs``: multiply-accumulates of the conv and linear layers, what a fused int8 kernel runs.
* ``macs`` (the lab number): the course ``count_net_flops``. It disables BatchNorm's forward but thop's
  hook still charges every BN 4 ops per output element, so the constraints in the lab (50M, 60M, ...)
  are 9-15% looser than they look. Both are reported; constraints use the lab number.
"""

from __future__ import annotations

from dataclasses import dataclass

from .space import STAGE_BLOCKS, WIDTHS, stage_depth

STEM_WIDTH = 32
FIRST_BLOCK_WIDTH = 16
STAGE_WIDTHS = (24, 40, 80, 96, 192, 320)
STAGE_STRIDES = (2, 2, 2, 1, 2, 1)
STAGE_SE = ((False,) * 3, (False, True, True, True), (True,) * 4, (True,) * 4, (True,) * 4, (False,))
MIX_WIDTH = 384
N_CLASSES = 2


def make_divisible(v: float, divisor: int, min_val: int | None = None) -> int:
    min_val = divisor if min_val is None else min_val
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@dataclass(frozen=True)
class Op:
    name: str
    in_elems: int
    out_elems: int
    macs: int
    params: int
    bn: bool = True  # followed by a BatchNorm (charged 4 ops per output element by the lab counter)

    @property
    def lab_macs(self) -> int:
        return self.macs + (4 * self.out_elems if self.bn else 0)


@dataclass(frozen=True)
class Unit:
    """One entry of the peak-memory rule: the stem, a block, the feature mix or the classifier."""
    name: str
    ops: tuple[Op, ...]
    residual: int = 0  # block input elements when the block has a shortcut

    def op(self, name: str) -> Op | None:
        return next((o for o in self.ops if o.name == name), None)


def _out(h: int, stride: int) -> int:
    return (h - 1) // stride + 1


def units(cfg: dict) -> list[Unit]:
    w = WIDTHS[cfg["wid"]]
    r = cfg["image_size"]
    c = make_divisible(STEM_WIDTH * w, 8)
    h = _out(r, 2)
    out = [Unit("stem", (Op("conv", 3 * r * r, c * h * h, h * h * c * 27, c * 27),))]
    c1 = make_divisible(FIRST_BLOCK_WIDTH * w, 8)
    out.append(Unit("block0", (Op("project", c * h * h, c1 * h * h, h * h * c * c1, c * c1),)))
    cin = c1
    i = 0
    for s, n in enumerate(STAGE_BLOCKS):
        cout = make_divisible(STAGE_WIDTHS[s] * w, 8)
        depth = stage_depth(cfg, s)
        for j in range(n):
            idx, i = i, i + 1
            if j >= depth:
                continue
            k, e = cfg["ks"][idx], cfg["e"][idx]
            stride = STAGE_STRIDES[s] if j == 0 else 1
            mid = make_divisible(round(cin * e), 8)
            ho = _out(h, stride)
            ops = [Op("expand", cin * h * h, mid * h * h, h * h * cin * mid, cin * mid),
                   Op("dw", mid * h * h, mid * ho * ho, ho * ho * mid * k * k, mid * k * k)]
            if STAGE_SE[s][j]:
                se = make_divisible(cin // 4, 1)
                ops += [Op("se_reduce", mid, se, mid * se, mid * se + se, bn=False),
                        Op("se_expand", se, mid, se * mid, se * mid + mid, bn=False)]
            ops.append(Op("project", mid * ho * ho, cout * ho * ho, ho * ho * mid * cout, mid * cout))
            out.append(Unit(f"s{s}b{j}", tuple(ops), residual=cin * h * h if j > 0 else 0))
            cin, h = cout, ho
    out.append(Unit("mix", (Op("conv", cin * h * h, MIX_WIDTH * h * h, h * h * cin * MIX_WIDTH, cin * MIX_WIDTH),)))
    out.append(Unit("classifier", (Op("linear", MIX_WIDTH, N_CLASSES, MIX_WIDTH * N_CLASSES,
                                      MIX_WIDTH * N_CLASSES + N_CLASSES, bn=False),)))
    return out


def _io(op: Op | None) -> int:
    return 0 if op is None else op.in_elems + op.out_elems


def unit_peak(u: Unit, rule: str = "lab") -> int:
    if u.name in ("stem", "mix", "classifier", "block0"):
        return _io(u.ops[0])
    expand, dw, project = u.op("expand"), u.op("dw"), u.op("project")
    if not u.residual:
        return max(_io(expand), _io(dw), _io(project))
    held = u.residual if rule == "residual" else 0
    return max(_io(expand), _io(dw) + u.residual, _io(project) + held)


def conv_macs(cfg: dict) -> int:
    return sum(o.macs for u in units(cfg) for o in u.ops)


def macs(cfg: dict) -> int:
    """The course counter's number, BatchNorm ops included."""
    return sum(o.lab_macs for u in units(cfg) for o in u.ops)


def params(cfg: dict) -> int:
    """Conv and linear weights plus SE and classifier biases. BatchNorm is folded and not counted."""
    return sum(o.params for u in units(cfg) for o in u.ops)


def peak(cfg: dict, rule: str = "lab") -> tuple[int, str]:
    best = max(units(cfg), key=lambda u: unit_peak(u, rule))
    return unit_peak(best, rule), best.name


def efficiency(cfg: dict) -> dict:
    """The two numbers the lab constrains: MACs in millions, peak activation in KB (1 KB = 1024 int8)."""
    us = units(cfg)
    best = max(us, key=unit_peak)
    res = max(unit_peak(u, "residual") for u in us)
    return {"mmacs": sum(o.lab_macs for u in us for o in u.ops) / 1e6,
            "mmacs_conv": sum(o.macs for u in us for o in u.ops) / 1e6,
            "kb": unit_peak(best) / 1024, "kb_residual": res / 1024, "peak_at": best.name}
