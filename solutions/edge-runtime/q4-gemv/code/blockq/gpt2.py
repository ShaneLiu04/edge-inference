"""GPT-2 forward pass straight from ``model.safetensors``, no transformers dependency.

Only what perplexity needs: causal self-attention, GELU (tanh form, as in the
original release) and pre-LayerNorm blocks. The projection layers keep GPT-2's
``Conv1D`` name and ``[in, out]`` weight layout so the quantization code can find
and rewrite them the same way it would in the Hugging Face model.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class Conv1D(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_in, n_out))
        self.bias = nn.Parameter(torch.empty(n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self.weight).reshape(*x.shape[:-1], -1)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.c_attn = Conv1D(d, 3 * d)
        self.c_proj = Conv1D(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.c_attn(x).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).reshape(b, t, d))


class MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.c_fc = Conv1D(d, 4 * d)
        self.c_proj = Conv1D(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, d: int, heads: int, eps: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d, eps=eps)
        self.attn = Attention(d, heads)
        self.ln_2 = nn.LayerNorm(d, eps=eps)
        self.mlp = MLP(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT2(nn.Module):
    def __init__(self, vocab: int, ctx: int, d: int, layers: int, heads: int, eps: float):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(ctx, d)
        self.h = nn.ModuleList(Block(d, heads, eps) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d, eps=eps)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.wte(ids) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        return self.ln_f(x) @ self.wte.weight.T  # tied LM head


def load_gpt2(path: Path) -> GPT2:
    from safetensors.torch import load_file

    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model = GPT2(cfg["vocab_size"], cfg["n_positions"], cfg["n_embd"], cfg["n_layer"], cfg["n_head"],
                 cfg.get("layer_norm_epsilon", 1e-5))
    state = {k.removeprefix("transformer."): v for k, v in load_file(str(path / "model.safetensors")).items()}
    state = {k: v for k, v in state.items() if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")}
    state.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model.eval()


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path / "tokenizer.json"))


def perplexity_from_logits(logits: torch.Tensor, ids: torch.Tensor) -> tuple[float, int]:
    """Summed next-token NLL and the number of predicted tokens."""
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
    return float(nll), ids[:, 1:].numel()
