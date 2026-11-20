"""Visual Wake Words evaluation with the course protocol, but decoded once and kept on the GPU.

The course loader is ``ImageFolder(vww-s256/val)`` -> ``Resize((r, r))`` -> ``ToTensor`` ->
``Normalize(0.5, 0.5)``, then ``Subset(range(n))[split::2]``: split 0 is the real validation
set, split 1 the holdout used for BN re-calibration and for building the accuracy dataset.
Here the resized uint8 pixels are cached per resolution; normalizing them on the GPU gives the
same tensors the course loader produces.

BN re-calibration follows the course ``set_running_statistics``: forward the holdout split in
batches of 128 with every BatchNorm normalizing by the batch's own mean and biased variance,
and set the running statistics to the sample-weighted average of those batch statistics.
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _load(args):
    path, r = args
    from PIL import Image
    from torchvision import transforms
    img = Image.open(path).convert("RGB")
    return np.asarray(transforms.Resize((r, r))(img), dtype=np.uint8)


class VWW:
    def __init__(self, root: str, device: str = "cuda", workers: int | None = None):
        from torchvision.datasets import ImageFolder
        ds = ImageFolder(root)
        self.paths = [p for p, _ in ds.samples]
        self.labels = torch.tensor([y for _, y in ds.samples], device=device)
        self.device = device
        self.workers = workers or min(64, os.cpu_count() or 1)
        self._cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def pixels(self, r: int) -> torch.Tensor:
        """uint8 N x 3 x r x r on the device, in ImageFolder order."""
        if r not in self._cache:
            with ProcessPoolExecutor(self.workers) as ex:
                arrs = list(ex.map(_load, [(p, r) for p in self.paths], chunksize=64))
            x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()
            self._cache[r] = x.to(self.device)
        return self._cache[r]

    def split(self, r: int, split: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pixels(r)[split::2], self.labels[split::2]


def normalize(u8: torch.Tensor) -> torch.Tensor:
    return (u8.float() / 255.0 - 0.5) / 0.5


@torch.no_grad()
def calibrate_bn(model: nn.Module, images: torch.Tensor, batch: int = 128) -> None:
    sums: dict[str, list] = {}
    fwd = copy.deepcopy(model)
    for name, m in fwd.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            acc = sums.setdefault(name, [0.0, 0.0, 0])

            def bn_forward(x, m=m, acc=acc):
                mean = x.mean(dim=(0, 2, 3))
                var = ((x - mean[None, :, None, None]) ** 2).mean(dim=(0, 2, 3))
                acc[0] = acc[0] + mean * x.size(0)
                acc[1] = acc[1] + var * x.size(0)
                acc[2] += x.size(0)
                return F.batch_norm(x, mean, var, m.weight, m.bias, False, 0.0, m.eps)

            m.forward = bn_forward
    fwd.eval()
    for s in range(0, len(images), batch):
        fwd(normalize(images[s:s + batch]))
    for name, m in model.named_modules():
        if name in sums and sums[name][2] > 0:
            m.running_mean.copy_(sums[name][0] / sums[name][2])
            m.running_var.copy_(sums[name][1] / sums[name][2])


@torch.no_grad()
def accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, batch: int = 512) -> float:
    model.eval()
    correct = 0
    for s in range(0, len(images), batch):
        correct += (model(normalize(images[s:s + batch])).argmax(1) == labels[s:s + batch]).sum().item()
    return 100.0 * correct / len(images)


def evaluate(model: nn.Module, data: VWW, r: int, calib_split: int = 1, eval_splits=(0, 1)) -> dict:
    """Re-calibrate BN on ``calib_split`` and return top-1 (%) on each of ``eval_splits``."""
    x_cal, _ = data.split(r, calib_split)
    calibrate_bn(model, x_cal)
    return {f"acc_split{s}": accuracy(model, *data.split(r, s)) for s in eval_splits}
