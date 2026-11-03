"""Tensor lifetimes of a PyTorch model, as an int8 runtime would allocate them.

The model is traced with torch.fx and shapes are propagated with one dummy
input. Then the graph is rewritten the way a deployment runtime sees it:

* BatchNorm, ReLU/ReLU6, Identity, Dropout and flatten/view are fused into
  (or aliased to) the tensor that feeds them. They allocate nothing.
* Every other node output is one buffer of ``numel * bytes_per_elem`` bytes.
* A buffer is live from the op that writes it to the last op that reads it.
  Graph inputs are live from step 0, graph outputs until the end.

Weights are not part of the activation arena; they sit in flash.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import fx, nn
from torch.fx.passes.shape_prop import ShapeProp

FUSED_MODULES = (nn.BatchNorm2d, nn.ReLU, nn.ReLU6, nn.Identity, nn.Dropout, nn.Hardswish, nn.Hardsigmoid)
ALIAS_FUNCTIONS = {torch.flatten, torch.reshape}
ALIAS_METHODS = {"view", "reshape", "flatten", "contiguous"}


@dataclass
class Tensor:
    name: str
    size: int
    birth: int
    death: int
    shape: tuple
    producer: str = ""
    depthwise_stride: int | None = None   # set when produced by a depthwise conv
    inputs: list[str] = field(default_factory=list)

    def overlaps(self, other: "Tensor") -> bool:
        return self.birth <= other.death and other.birth <= self.death


def _is_fused(gm: fx.GraphModule, node: fx.Node) -> bool:
    if node.op == "call_module":
        return isinstance(gm.get_submodule(node.target), FUSED_MODULES)
    if node.op == "call_function":
        return node.target in ALIAS_FUNCTIONS or getattr(node.target, "__name__", "") in ("relu", "relu6", "hardtanh")
    if node.op == "call_method":
        return node.target in ALIAS_METHODS
    return False


def trace(model: nn.Module, input_shape: tuple, bytes_per_elem: int = 1, align: int = 16) -> list[Tensor]:
    """One Tensor per materialized buffer, in execution order."""
    model = model.eval()
    gm = fx.symbolic_trace(model)
    with torch.no_grad():
        ShapeProp(gm).propagate(torch.zeros(input_shape))

    root: dict[str, str] = {}      # node name -> name of the buffer it lives in
    tensors: dict[str, Tensor] = {}
    step = 0
    order: list[str] = []
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        if node.op == "get_attr":
            continue
        args = [a for a in node.all_input_nodes]
        if node.op != "placeholder" and _is_fused(gm, node):
            root[node.name] = root[args[0].name]
            continue
        meta = node.meta.get("tensor_meta")
        shape = tuple(meta.shape) if meta is not None else ()
        numel = 1
        for d in shape:
            numel *= int(d)
        size = -(-numel * bytes_per_elem // align) * align
        if node.op != "placeholder":
            step += 1
        t = Tensor(node.name, size, step, step, shape, producer=str(node.target),
                   inputs=[root[a.name] for a in args if a.name in root])
        if node.op == "call_module":
            m = gm.get_submodule(node.target)
            if isinstance(m, nn.Conv2d) and m.groups > 1 and m.groups == m.in_channels == m.out_channels:
                t.depthwise_stride = m.stride[0]
        tensors[node.name] = t
        root[node.name] = node.name
        order.append(node.name)
        for src in t.inputs:
            tensors[src].death = max(tensors[src].death, step)

    # Graph outputs stay live until after the last op.
    out_node = next(n for n in gm.graph.nodes if n.op == "output")
    for a in out_node.all_input_nodes:
        tensors[root[a.name]].death = step + 1
    return [tensors[n] for n in order]


def inplace_depthwise(tensors: list[Tensor]) -> list[Tensor]:
    """TinyEngine in-place depthwise: when the input dies at the depthwise op, the output reuses its buffer.

    The kernel processes one channel at a time through a temporary buffer of one input channel,
    so the merged buffer is max(in, out) and a small scratch tensor is live during the op.
    """
    out: dict[str, Tensor] = {}
    host_of: dict[str, str] = {}
    for t in tensors:
        t = Tensor(t.name, t.size, t.birth, t.death, t.shape, t.producer, t.depthwise_stride,
                   [host_of.get(i, i) for i in t.inputs])
        if t.depthwise_stride is not None and len(t.inputs) == 1:
            src = out[t.inputs[0]]
            if src.death == t.birth:  # nobody reads the input after this op
                src.size = max(src.size, t.size)
                src.death = t.death
                host_of[t.name] = src.name
                h, w = int(src.shape[-2]), int(src.shape[-1])
                scratch = Tensor(f"{t.name}.scratch", -(-h * w // 16) * 16, t.birth, t.birth, (h, w), "scratch")
                out[scratch.name] = scratch
                continue
        out[t.name] = t
    return list(out.values())


def live_profile(tensors: list[Tensor]) -> list[int]:
    """Bytes live at every step."""
    last = max(t.death for t in tensors)
    prof = [0] * (last + 1)
    for t in tensors:
        for s in range(t.birth, t.death + 1):
            prof[s] += t.size
    return prof
