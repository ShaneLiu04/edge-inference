"""Group-wise asymmetric fake quantization and the three AWQ ingredients, written against plain tensors.

All functions take `w` as (out_features, in_features) and `x` as (tokens, in_features), so they are
testable without a language model.
"""

from __future__ import annotations

import torch


def pseudo_quantize(w: torch.Tensor, n_bit: int = 3, group: int = 128,
                    clip: torch.Tensor | None = None) -> torch.Tensor:
    """Round-to-nearest with per-group min/max; `clip` (out, n_groups) shrinks the max |w| per group."""
    shape = w.shape
    g = w.reshape(-1, group)
    if clip is not None:
        c = clip.reshape(-1, 1).to(g.dtype)
        g = g.clamp(-c, c)
    mx = g.amax(dim=1, keepdim=True)
    mn = g.amin(dim=1, keepdim=True)
    qmax = (1 << n_bit) - 1
    scale = ((mx - mn).clamp(min=1e-5) / qmax)
    zero = (-torch.round(mn / scale)).clamp(0, qmax)
    q = torch.clamp(torch.round(g / scale) + zero, 0, qmax)
    return ((q - zero) * scale).reshape(shape)


def channel_importance(x: torch.Tensor) -> torch.Tensor:
    """Mean |activation| per input channel, the AWQ saliency signal."""
    return x.abs().float().mean(dim=0)


def quantize_keep_salient(w: torch.Tensor, importance: torch.Tensor, frac: float,
                          n_bit: int = 3, group: int = 128) -> torch.Tensor:
    """Q1: quantize everything, then restore the top `frac` input channels (columns) to full precision."""
    k = max(1, int(round(frac * w.shape[1])))
    idx = importance.topk(k).indices
    q = pseudo_quantize(w, n_bit, group)
    q[:, idx] = w[:, idx]
    return q


@torch.no_grad()
def search_scale(ws: list[torch.Tensor], x: torch.Tensor, n_bit: int = 3, group: int = 128,
                 grid: int = 20) -> tuple[torch.Tensor, float, list[float]]:
    """Q2: s = s_x^alpha, alpha in [0, 1); minimize || Q(W diag(s)) (x / s) - W x ||^2 over the stacked ws.

    Returns (best scales, best alpha, loss per alpha).
    """
    w = torch.cat(ws, dim=0)
    ref = x @ w.t()
    sx = channel_importance(x).to(w.dtype)
    losses = []
    best, best_alpha, best_scale = float("inf"), 0.0, torch.ones_like(sx)
    for i in range(grid):
        alpha = i / grid
        s = sx.pow(alpha).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        out = (x / s) @ pseudo_quantize(w * s, n_bit, group).t()
        loss = (out - ref).float().pow(2).mean().item()
        losses.append(loss)
        if loss < best:
            best, best_alpha, best_scale = loss, alpha, s.clone()
    return best_scale, best_alpha, losses


@torch.no_grad()
def search_clip(w: torch.Tensor, x: torch.Tensor, n_bit: int = 3, group: int = 128, grid: int = 20,
                max_shrink: float = 0.5, n_tokens: int = 4096, oc_batch: int = 256) -> torch.Tensor:
    """Bonus: per-group clip threshold max|w| * r, r in [1 - max_shrink, 1], minimizing that group's output error."""
    x = x[torch.linspace(0, x.shape[0] - 1, min(n_tokens, x.shape[0]), device=x.device).long()]
    n_groups = w.shape[1] // group
    xg = x.reshape(x.shape[0], n_groups, group)  # t, G, g
    clips = []
    for o in range(0, w.shape[0], oc_batch):
        wo = w[o:o + oc_batch].reshape(-1, n_groups, group)  # O, G, g
        ref = torch.einsum("tgk,ogk->otg", xg, wo)
        amax = wo.abs().amax(dim=-1)  # O, G
        best_err = torch.full_like(amax, float("inf"), dtype=torch.float32)
        best = amax.clone()
        for i in range(int(max_shrink * grid)):
            c = amax * (1 - i / grid)
            q = pseudo_quantize(wo.reshape(wo.shape[0], -1), n_bit, group, clip=c).reshape_as(wo)
            err = (torch.einsum("tgk,ogk->otg", xg, q) - ref).float().pow(2).mean(dim=1)
            better = err < best_err
            best_err[better] = err[better]
            best[better] = c[better]
        clips.append(best)
    return torch.cat(clips, dim=0)
