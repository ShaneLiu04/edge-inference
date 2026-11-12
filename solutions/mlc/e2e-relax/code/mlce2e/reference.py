"""The assignment's PyTorch model, a NumPy reference of every layer, and the Fashion-MNIST loader.

Nothing here imports TVM, so it is testable on any machine.
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np

BATCH = 4
INPUT_SHAPE = (BATCH, 1, 28, 28)
NAME_MAP = {"0.weight": "conv2d_weight", "0.bias": "conv2d_bias", "4.weight": "linear0_weight",
            "4.bias": "linear0_bias", "6.weight": "linear1_weight", "6.bias": "linear1_bias"}


def load_weights(path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        return {k: np.asarray(v, dtype=np.float32) for k, v in pickle.load(f).items()}


def pytorch_model(weight_map: dict[str, np.ndarray]):
    import torch
    from torch import nn

    model = nn.Sequential(
        nn.Conv2d(1, 32, (3, 3), bias=True), nn.ReLU(), nn.MaxPool2d((2, 2)), nn.Flatten(),
        nn.Linear(5408, 100, bias=True), nn.ReLU(), nn.Linear(100, 10, bias=True), nn.Softmax(dim=1),
    )
    for name, p in model.named_parameters():
        p.data = torch.from_numpy(weight_map[NAME_MAP[name]].copy())
    return model.eval()


def load_fashion_mnist_test(raw_dir) -> tuple[np.ndarray, np.ndarray]:
    """(10000, 1, 28, 28) float32 in [0, 1] (ToTensor, no normalization) and int64 labels."""
    raw = Path(raw_dir)

    def read(name, offset):
        p = raw / name
        data = gzip.open(p).read() if p.suffix == ".gz" else p.read_bytes()
        return np.frombuffer(data, np.uint8, offset=offset)

    imgs = _first(raw, ("t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"))
    labs = _first(raw, ("t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"))
    x = read(imgs, 16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    y = read(labs, 8).astype(np.int64)
    return x, y


def _first(raw: Path, names) -> str:
    for n in names:
        if (raw / n).exists():
            return n
    raise FileNotFoundError(f"none of {names} in {raw}")


def conv2d_nchw(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Valid (no padding), stride 1. Used as the 'external library' and as the test oracle."""
    n, c, h, wd = x.shape
    f, _, kh, kw = w.shape
    oh, ow = h - kh + 1, wd - kw + 1
    cols = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))  # n c oh ow kh kw
    return np.einsum("ncyxij,fcij->nfyx", cols, w, optimize=True).astype(np.float32).reshape(n, f, oh, ow)


def forward_numpy(x: np.ndarray, wm: dict[str, np.ndarray]) -> np.ndarray:
    y = conv2d_nchw(x, wm["conv2d_weight"]) + wm["conv2d_bias"].reshape(1, -1, 1, 1)
    y = np.maximum(y, 0)
    n, c, h, w = y.shape
    y = y.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5)).reshape(n, -1)
    y = np.maximum(y @ wm["linear0_weight"].T + wm["linear0_bias"], 0)
    y = y @ wm["linear1_weight"].T + wm["linear1_bias"]
    y = np.exp(y - y.max(axis=1, keepdims=True))
    return (y / y.sum(axis=1, keepdims=True)).astype(np.float32)
