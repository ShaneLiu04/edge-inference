"""k-means weight sharing (Deep Compression): per-layer codebook of 2^bits centroids plus integer labels.

QAT keeps labels fixed and moves each centroid to the mean of the weights currently assigned to it,
after every optimizer step; the weights are then snapped back to their centroid.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Codebook:
    centroids: torch.Tensor  # (k,)
    labels: torch.Tensor  # same shape as the weight, int64


def kmeans_1d(x: torch.Tensor, k: int, iters: int = 50) -> Codebook:
    """Lloyd's algorithm on a flattened tensor; quantile init makes it deterministic.

    Sorting the centroids lets assignment be a bucketize over midpoints, which is exact in 1-D.
    """
    flat = x.detach().flatten().float()
    k = min(k, flat.numel())
    srt = flat.sort().values
    c = srt[((torch.arange(k, device=flat.device) + 0.5) / k * flat.numel()).long()]
    for _ in range(iters):
        c = c.sort().values
        labels = torch.bucketize(flat, (c[1:] + c[:-1]) / 2)
        sums = torch.zeros(k, device=flat.device).index_add_(0, labels, flat)
        counts = torch.bincount(labels, minlength=k).float()
        new = torch.where(counts > 0, sums / counts.clamp(min=1), c)
        if torch.allclose(new, c):
            c = new
            break
        c = new
    c = c.sort().values
    labels = torch.bucketize(flat, (c[1:] + c[:-1]) / 2)
    return Codebook(c.to(x.dtype), labels.view_as(x))


def update_centroids(weight: torch.Tensor, cb: Codebook) -> None:
    k = cb.centroids.numel()
    flat = weight.detach().flatten().float()
    lab = cb.labels.flatten()
    sums = torch.zeros(k, device=flat.device).index_add_(0, lab, flat)
    counts = torch.bincount(lab, minlength=k).float()
    cb.centroids = torch.where(counts > 0, sums / counts.clamp(min=1), cb.centroids.float()).to(weight.dtype)


class KMeansQuantizer:
    """Quantize every conv / linear weight (dim > 1) to `bits`; `apply(update=True)` is the QAT step."""

    def __init__(self, model: nn.Module, bits: int):
        self.bits = bits
        self.codebooks: dict[str, Codebook] = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() > 1:
                    self.codebooks[name] = kmeans_1d(p, 1 << bits)
        self.apply(model, update=False)

    @torch.no_grad()
    def apply(self, model: nn.Module, update: bool = False) -> None:
        for name, p in model.named_parameters():
            cb = self.codebooks.get(name)
            if cb is None:
                continue
            if update:
                update_centroids(p, cb)
            p.copy_(cb.centroids[cb.labels])


def kmeans_model_size_bits(model: nn.Module, bits: int) -> int:
    """Labels at `bits` plus a 32-bit codebook per quantized tensor; other parameters at 32 bits."""
    total = 0
    for p in model.parameters():
        total += p.numel() * bits + (1 << bits) * 32 if p.dim() > 1 else p.numel() * 32
    return total
