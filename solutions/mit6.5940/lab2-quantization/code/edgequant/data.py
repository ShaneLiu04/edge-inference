"""CIFAR-10 as uint8 numpy arrays, cached once under ``<repo>/.data``.

Source: the Hugging Face copy of CIFAR-10 (``uoft-cs/cifar10``, two parquet
files of PNG bytes). ``EDGE_HF_ENDPOINT`` picks the host; the default is
hf-mirror.com because huggingface.co does not answer behind the company proxy.
Nothing under ``.data`` is committed. ``EDGE_DATA_DIR`` overrides the location.
"""

from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
DATASET = "datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def data_dir() -> Path:
    return Path(os.environ.get("EDGE_DATA_DIR", REPO / ".data"))


def _parquet(split: str) -> Path:
    path = data_dir() / f"cifar10-{split}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = os.environ.get("EDGE_HF_ENDPOINT", "https://hf-mirror.com").rstrip("/") + "/" + DATASET.format(split=split)
        tmp = path.with_suffix(".part")
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def _decode(split: str) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(_parquet(split)).to_pydict()
    images = np.stack([np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB")) for item in table["img"]])
    return images.astype(np.uint8), np.asarray(table["label"], dtype=np.int64)


def load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x_train (50000,32,32,3) uint8, y_train, x_test (10000,32,32,3), y_test."""
    cache = data_dir() / "cifar10.npz"
    if cache.exists():
        d = np.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    xtr, ytr = _decode("train")
    xte, yte = _decode("test")
    np.savez(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return xtr, ytr, xte, yte


def normalize(x_uint8: np.ndarray) -> np.ndarray:
    """NHWC uint8 to NCHW float32, per-channel standardized."""
    x = x_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))
