"""OPT plumbing: calibration and evaluation data, activation capture, applying scales, perplexity."""

from __future__ import annotations

import io
import json
from pathlib import Path

import torch
from torch import nn

from .quant import pseudo_quantize, quantize_keep_salient, search_clip, search_scale

SEQLEN = 2048
EVAL_SAMPLES = 40


def load_opt(path, device="cuda"):
    from transformers import AutoTokenizer, OPTForCausalLM
    tok = AutoTokenizer.from_pretrained(path)
    model = OPTForCausalLM.from_pretrained(path, dtype=torch.float16).to(device).eval()
    return model, tok


def pile_calib(path, tok, n_samples: int = 128, block: int = 512) -> torch.Tensor:
    """The AWQ recipe: first texts of pile-val of at most `block` tokens, concatenated, cut into blocks."""
    import zstandard
    ids = []
    taken = 0
    with open(path, "rb") as fh:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for line in reader:
            t = tok(json.loads(line)["text"].strip()).input_ids
            if not t or len(t) > block:
                continue
            ids.extend(t)
            taken += 1
            if taken >= n_samples:
                break
    n = len(ids) // block
    return torch.tensor(ids[: n * block]).view(n, block)


def wikitext_test_ids(parquet_path, tok) -> torch.Tensor:
    import pandas as pd
    text = "\n\n".join(pd.read_parquet(parquet_path)["text"].tolist())
    return tok(text, return_tensors="pt").input_ids


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, n_samples: int = EVAL_SAMPLES, seqlen: int = SEQLEN) -> float:
    dev = next(model.parameters()).device
    nlls = []
    for i in range(n_samples):
        batch = ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        logits = model(batch).logits.float()
        loss = nn.functional.cross_entropy(logits[0, :-1], batch[0, 1:])
        nlls.append(loss * seqlen)
    return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen)).item()


def decoder_layers(model):
    return model.model.decoder.layers


def linears(model) -> dict[str, nn.Linear]:
    return {n: m for n, m in model.model.decoder.layers.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def capture_inputs(model, calib: torch.Tensor, batch: int = 16) -> dict[str, torch.Tensor]:
    """Input activations (tokens, in_features) of every linear in the decoder, kept on CPU in fp16."""
    feats: dict[str, list[torch.Tensor]] = {}
    hooks = []
    for name, m in linears(model).items():
        def hook(_m, inp, _out, name=name):
            feats.setdefault(name, []).append(inp[0].reshape(-1, inp[0].shape[-1]).half().cpu())
        hooks.append(m.register_forward_hook(hook))
    dev = next(model.parameters()).device
    for i in range(0, len(calib), batch):
        model(calib[i:i + batch].to(dev))
    for h in hooks:
        h.remove()
    return {k: torch.cat(v) for k, v in feats.items()}


@torch.no_grad()
def apply_rtn(model, n_bit: int, group: int) -> None:
    for m in linears(model).values():
        m.weight.copy_(pseudo_quantize(m.weight, n_bit, group))


@torch.no_grad()
def apply_salient(model, feats, frac: float, n_bit: int, group: int, mode: str = "activation", seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for name, m in linears(model).items():
        if mode == "activation":
            imp = feats[name].to(m.weight.device).abs().float().mean(0)
        elif mode == "weight":
            imp = m.weight.abs().float().mean(0)
        else:
            imp = torch.rand(m.weight.shape[1], generator=g).to(m.weight.device)
        m.weight.copy_(quantize_keep_salient(m.weight, imp, frac, n_bit, group))


def _scale_prev(prev: nn.Module, s: torch.Tensor) -> None:
    """Divide the producer of the input by s: a LayerNorm (weight and bias) or a Linear (rows)."""
    prev.weight.div_(s.view(-1, *([1] * (prev.weight.dim() - 1))))
    if prev.bias is not None:
        prev.bias.div_(s)


@torch.no_grad()
def apply_awq(model, feats, n_bit: int, group: int, grid: int = 20, clip: bool = False,
              n_tokens: int = 16384) -> dict:
    """Scale search per input group, folded into the producer; then optional clip search; then RTN.

    Groups per layer: (q, k, v) after self_attn_layer_norm; out_proj after v_proj; fc1 after
    final_layer_norm; fc2 after fc1 (valid through ReLU because ReLU(x/s) = ReLU(x)/s for s > 0).
    The objective is the MSE of the group's own outputs on `n_tokens` calibration tokens.
    """
    log = {"alpha": {}, "loss": {}}
    dev = next(model.parameters()).device
    n_tok = len(next(iter(feats.values())))
    sel = torch.linspace(0, n_tok - 1, min(n_tokens, n_tok)).long()
    for li, layer in enumerate(decoder_layers(model)):
        a = layer.self_attn
        p = f"{li}.self_attn."
        groups = [
            ("qkv", layer.self_attn_layer_norm, [a.q_proj, a.k_proj, a.v_proj],
             [p + "q_proj", p + "k_proj", p + "v_proj"]),
            ("out", a.v_proj, [a.out_proj], [p + "out_proj"]),
            ("fc1", layer.final_layer_norm, [layer.fc1], [f"{li}.fc1"]),
            ("fc2", layer.fc1, [layer.fc2], [f"{li}.fc2"]),
        ]
        for tag, prev, mods, names in groups:
            xs = feats[names[0]][sel].to(dev)
            s, alpha, losses = search_scale([m.weight for m in mods], xs, n_bit, group, grid)
            _scale_prev(prev, s)
            for m in mods:
                m.weight.mul_(s.view(1, -1))
            s_cpu = s.cpu().half()
            for n in names:
                feats[n].div_(s_cpu)
            log["alpha"][f"{li}.{tag}"] = alpha
            log["loss"][f"{li}.{tag}"] = losses
    if clip:
        for name, m in linears(model).items():
            if name.endswith(("q_proj", "k_proj")):
                continue
            c = search_clip(m.weight, feats[name][sel].to(dev), n_bit, group)
            m.weight.copy_(pseudo_quantize(m.weight, n_bit, group, clip=c))
        for name, m in linears(model).items():
            if name.endswith(("q_proj", "k_proj")):
                m.weight.copy_(pseudo_quantize(m.weight, n_bit, group))
    else:
        apply_rtn(model, n_bit, group)
    return log


def model_size_bits(model, n_bit: int | None, group: int = 128, fp16_frac: float = 0.0) -> int:
    """Decoder linears at n_bit plus an fp16 scale and n_bit zero per group (or fp16 when n_bit is None)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * 16
    if n_bit is None:
        return total
    for m in linears(model).values():
        n = m.weight.numel()
        q = n * n_bit + (n // group) * (16 + n_bit)
        total += -n * 16 + int(q * (1 - fp16_frac) + n * 16 * fp16_frac)
    return total


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
