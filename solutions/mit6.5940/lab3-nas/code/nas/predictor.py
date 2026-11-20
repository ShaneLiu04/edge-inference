"""MLP accuracy predictor (Q3, Q4) and the ranking metrics a searcher actually depends on."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .space import N_FEATURES


class AccuracyMLP(nn.Module):
    """``n_layers`` x (Linear(hidden) + ReLU), then Linear(hidden, 1) without bias, plus a fixed offset.

    The offset is the mean accuracy of the training set, so the layers only learn the deviation.
    """

    def __init__(self, in_dim: int = N_FEATURES, hidden: int = 400, n_layers: int = 3, base_acc: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            d = hidden
        layers.append(nn.Linear(hidden, 1, bias=False))
        self.layers = nn.Sequential(*layers)
        self.register_buffer("base_acc", torch.tensor(float(base_acc)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1) + self.base_acc


def split(n: int, n_train: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Shuffled train / validation indices; validation is always the last n - n_train of the permutation."""
    perm = np.random.default_rng(seed).permutation(n)
    return perm[:n_train], perm[n_train:]


def fit(model: AccuracyMLP, x_tr: torch.Tensor, y_tr: torch.Tensor, x_va: torch.Tensor, y_va: torch.Tensor,
        epochs: int = 10, batch: int = 256, lr: float = 1e-3, seed: int = 0, loss: str = "l1") -> list[dict]:
    """Adam, plain minibatches, L1 (the lab's choice) or MSE. Returns per-epoch train loss and validation L1."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = {"l1": nn.L1Loss(), "mse": nn.MSELoss()}[loss]
    hist = []
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(x_tr), generator=g).to(x_tr.device)
        total = 0.0
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        hist.append({"epoch": ep + 1, "train_l1": total / len(x_tr), "val_l1": evaluate_l1(model, x_va, y_va)})
    return hist


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch: int = 8192) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[s:s + batch]) for s in range(0, len(x), batch)])


def evaluate_l1(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return (predict(model, x) - y).abs().mean().item()


def _ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    # average ranks over ties
    vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals))
    np.add.at(sums, inv, r)
    return (sums / counts)[inv]


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b) -> float:
    return pearson(_ranks(np.asarray(a, float)), _ranks(np.asarray(b, float)))


def kendall_tau(a, b) -> float:
    """Tau-a over all pairs, chunked so 10k points fit in memory."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    s = 0.0
    for i in range(0, n, 512):
        da = np.sign(a[i:i + 512, None] - a[None, :])
        db = np.sign(b[i:i + 512, None] - b[None, :])
        s += float((da * db).sum())
    return s / (n * (n - 1))


def top_overlap(pred, true, frac: float = 0.01) -> float:
    """Share of the true top ``frac`` that the predictor also puts in its top ``frac``."""
    pred, true = np.asarray(pred), np.asarray(true)
    k = max(1, int(round(len(true) * frac)))
    return len(set(np.argsort(-pred)[:k]) & set(np.argsort(-true)[:k])) / k
