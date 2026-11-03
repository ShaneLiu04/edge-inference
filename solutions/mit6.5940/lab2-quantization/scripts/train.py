"""Train MicroMobileNetV2 on CIFAR-10 on the CPU.

Writes the checkpoint to ``checkpoints/micronet.pt`` (not committed) and the
per-epoch log to ``artifacts/raw/train_log.jsonl``. The quantization
experiments only need a converged fp32 model; the exact accuracy does not
matter as long as it is far above chance.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parents[2]
sys.path.insert(0, str(LAB / "code"))
sys.path.insert(0, str(REPO))

from edge_harness.writer import JsonlWriter  # noqa: E402
from edgequant.data import load_cifar10, normalize  # noqa: E402
from edgequant.model import MicroMobileNetV2, count_macs, count_params  # noqa: E402

CKPT = LAB / "checkpoints" / "micronet.pt"


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random crop with 4-pixel reflect padding and horizontal flip, on a batch."""
    n = x.shape[0]
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    ix = torch.randint(0, 9, (n,), generator=gen)
    iy = torch.randint(0, 9, (n,), generator=gen)
    rows = (iy[:, None] + torch.arange(32)[None, :])
    cols = (ix[:, None] + torch.arange(32)[None, :])
    out = padded[torch.arange(n)[:, None, None, None], torch.arange(3)[None, :, None, None],
                 rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(n, generator=gen) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


@torch.no_grad()
def evaluate(model, x: torch.Tensor, y: torch.Tensor, batch: int = 1000) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        correct += (model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
    return correct / len(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    # 4 physical cores; 8 threads oversubscribe the hyperthreads and run 15x slower.
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    xtr, ytr, xte, yte = load_cifar10()
    xtr_t = torch.from_numpy(normalize(xtr))
    ytr_t = torch.from_numpy(ytr)
    xte_t = torch.from_numpy(normalize(xte))
    yte_t = torch.from_numpy(yte)

    model = MicroMobileNetV2()
    steps_per_epoch = len(xtr_t) // args.batch
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=steps_per_epoch, pct_start=0.15)
    log = JsonlWriter(LAB / "artifacts" / "raw" / "train_log.jsonl", cwd=REPO)
    log.write({"kind": "model", "params": count_params(model), "macs": count_macs(model),
               "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seed": args.seed})
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr_t), generator=gen)
        total, t0 = 0.0, time.perf_counter()
        for step in range(steps_per_epoch):
            idx = perm[step * args.batch:(step + 1) * args.batch]
            xb = augment(xtr_t[idx], gen)
            loss = F.cross_entropy(model(xb), ytr_t[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        acc = evaluate(model, xte_t, yte_t)
        row = {"kind": "epoch", "epoch": epoch + 1, "loss": total / steps_per_epoch, "test_acc": acc,
               "epoch_s": time.perf_counter() - t0}
        log.write(row)
        print(row, flush=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT)
    log.write({"kind": "final", "test_acc": acc, "train_s": time.perf_counter() - t_start})
    log.close()
    print(f"saved {CKPT}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
