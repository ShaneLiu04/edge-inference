"""Magnitude-based fine-grained pruning, sensitivity scan and per-layer sparsity selection."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn


def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Zero the `round(sparsity * numel)` smallest-magnitude entries in place; return the keep mask."""
    sparsity = min(max(0.0, float(sparsity)), 1.0)
    n = tensor.numel()
    num_zeros = round(n * sparsity)
    if num_zeros == 0:
        return torch.ones_like(tensor, dtype=torch.bool)
    if num_zeros >= n:
        tensor.zero_()
        return torch.zeros_like(tensor, dtype=torch.bool)
    importance = tensor.detach().abs()
    threshold = importance.reshape(-1).kthvalue(num_zeros).values
    mask = importance > threshold
    tensor.mul_(mask)
    return mask


def prunable(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Conv and linear weights; BN and biases stay dense."""
    return [(n, p) for n, p in model.named_parameters() if p.dim() > 1]


class FineGrainedPruner:
    """Holds one keep mask per weight; `apply` re-zeroes pruned entries after each optimizer step."""

    def __init__(self, model: nn.Module, sparsity: dict[str, float]):
        self.masks: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, p in prunable(model):
                if name in sparsity:
                    self.masks[name] = fine_grained_prune(p.data, sparsity[name])

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.masks:
                p.mul_(self.masks[name])


def get_num_parameters(model: nn.Module, count_nonzero_only: bool = False) -> int:
    total = 0
    for p in model.parameters():
        total += int(p.count_nonzero()) if count_nonzero_only else p.numel()
    return total


def get_model_size(model: nn.Module, data_width: int = 32, count_nonzero_only: bool = False) -> int:
    """Bits, counting every parameter (or only the nonzero ones) at `data_width`."""
    return get_num_parameters(model, count_nonzero_only) * data_width


def get_sparsity(t: torch.Tensor) -> float:
    return 1.0 - float(t.count_nonzero()) / t.numel()


@torch.no_grad()
def sensitivity_scan(model: nn.Module, evaluate: Callable[[nn.Module], float],
                     sparsities=np.arange(0.4, 1.0, 0.1)) -> tuple[list[float], dict[str, list[float]]]:
    """Prune one layer at a time to each sparsity, evaluate, restore. Returns (sparsities, {layer: [acc]})."""
    sparsities = [round(float(s), 2) for s in sparsities]
    accs: dict[str, list[float]] = {}
    for name, p in prunable(model):
        orig = p.detach().clone()
        accs[name] = []
        for s in sparsities:
            fine_grained_prune(p.data, s)
            accs[name].append(evaluate(model))
            p.copy_(orig)
    return sparsities, accs


def pick_sparsity(sparsities: list[float], row: list[float], floor: float) -> float:
    """Largest scanned sparsity reached before the one-layer accuracy first drops below `floor`."""
    chosen = 0.0
    for s, a in zip(sparsities, row):
        if a < floor:
            break
        chosen = s
    return chosen


def kept_fraction(choice: dict[str, float], numels: dict[str, int], dense_numel: int) -> float:
    fixed = dense_numel - sum(numels.values())
    return (fixed + sum(round(numels[n] * (1 - s)) for n, s in choice.items())) / dense_numel


def choose_sparsities(sparsities: list[float], accs: dict[str, list[float]], numels: dict[str, int],
                      dense_numel: int, dense_acc: float, budget: float = 0.25) -> tuple[dict[str, float], float]:
    """Smallest uniform tolerance `d` (accuracy points) whose per-layer picks fit the size budget.

    Each layer gets `pick_sparsity(..., dense_acc - d)`. `budget` is the kept fraction of all
    parameters; BN weights and biases stay dense and count against it.
    """
    for d in np.arange(0.0, 100.0, 0.1):
        choice = {n: pick_sparsity(sparsities, row, dense_acc - d) for n, row in accs.items()}
        if kept_fraction(choice, numels, dense_numel) <= budget:
            return choice, round(float(d), 1)
    raise ValueError("budget not reachable with the scanned sparsities")
