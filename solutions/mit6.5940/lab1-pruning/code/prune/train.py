"""CIFAR-10 on device tensors: the course's crop+flip augmentation, evaluation and fine-tuning."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def to_tensors(x_uint8: np.ndarray, y: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    """NHWC uint8 -> NCHW float in [0, 1] (the course loader uses ToTensor only, no normalization)."""
    x = torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    return x.contiguous(), torch.from_numpy(y.astype(np.int64)).to(device)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """RandomCrop(32, padding=4) and RandomHorizontalFlip, per sample."""
    n, _, h, w = x.shape
    padded = F.pad(x, (4, 4, 4, 4))
    dy = torch.randint(0, 9, (n,), generator=gen, device=x.device)
    dx = torch.randint(0, 9, (n,), generator=gen, device=x.device)
    rows = (dy[:, None] + torch.arange(h, device=x.device))[:, :, None]
    cols = (dx[:, None] + torch.arange(w, device=x.device))[:, None, :]
    out = padded[torch.arange(n, device=x.device)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(n, generator=gen, device=x.device) < 0.5
    out[flip] = out[flip].flip(3)
    return out.contiguous()


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 1000) -> float:
    """Top-1 in percent."""
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        correct += int((model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum())
    return 100.0 * correct / len(x)


def finetune(model: nn.Module, xtr, ytr, xte, yte, epochs: int = 5, lr: float = 0.01,
             batch: int = 512, weight_decay: float = 1e-4, seed: int = 0,
             after_step: Callable[[nn.Module], None] | None = None, cosine: bool = True) -> list[dict]:
    """SGD + cosine, the course recipe. `after_step` re-applies pruning masks. Returns per-epoch rows.

    `cosine=False` keeps lr constant, for callers that step an epoch-level schedule themselves.
    """
    gen = torch.Generator(device=xtr.device).manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    steps = epochs * ((len(xtr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if cosine else \
        torch.optim.lr_scheduler.LambdaLR(opt, lambda _s: 1.0)
    rows = []
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(xtr), generator=gen, device=xtr.device)
        for i in range(0, len(xtr), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(model(augment(xtr[idx], gen)), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if after_step is not None:
                after_step(model)
        rows.append({"epoch": ep + 1, "acc": evaluate(model, xte, yte), "seconds": time.perf_counter() - t0})
    return rows
