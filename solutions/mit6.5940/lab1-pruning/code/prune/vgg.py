"""The 8-conv VGG used by the course labs; parameter names match the published checkpoint."""

from __future__ import annotations

from collections import OrderedDict, defaultdict

import torch
from torch import nn

ARCH = (64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M")


class VGG(nn.Module):
    def __init__(self, arch: tuple = ARCH, n_classes: int = 10) -> None:
        super().__init__()
        layers: list[tuple[str, nn.Module]] = []
        counts: dict[str, int] = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_ch = 3
        for x in arch:
            if x == "M":
                add("pool", nn.MaxPool2d(2))
            else:
                add("conv", nn.Conv2d(in_ch, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_ch = x
        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(in_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.classifier(x.mean([2, 3]))


def load_pretrained(path, device="cpu") -> VGG:
    model = VGG()
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    return model.to(device).eval()
