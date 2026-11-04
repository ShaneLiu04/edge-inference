"""Parse llama.cpp tool output: llama-bench JSONL, llama-perplexity logs (PPL and KL divergence), llama-quantize."""

from __future__ import annotations

import json
import re

_STAT = re.compile(r"^\s*([^:=]+?)\s*:\s*(-?[\d.]+)\s*(?:%\s*)?(?:±\s*(-?[\d.]+))?\s*%?\s*$")
_FINAL = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")
_QSIZE = re.compile(r"quant size\s*=\s*([\d.]+) MiB \(([\d.]+) BPW\)")
_MSIZE = re.compile(r"model size\s*=\s*([\d.]+) MiB \(([\d.]+) BPW\)")


def bench_rows(text: str) -> list[dict]:
    """llama-bench -o jsonl: one JSON object per line. Adds `test` = pp<N> / tg<N> / pp<N>+tg<M>."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        npp, ntg = int(r.get("n_prompt", 0)), int(r.get("n_gen", 0))
        r["test"] = f"pp{npp}" if ntg == 0 else (f"tg{ntg}" if npp == 0 else f"pp{npp}+tg{ntg}")
        rows.append(r)
    return rows


def final_ppl(text: str) -> tuple[float, float] | None:
    m = _FINAL.search(text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def kld_stats(text: str) -> dict[str, float]:
    """`name : value ± err` lines of the KL-divergence summary, keyed by a normalised name.

    llama-perplexity prints sections such as
        Mean    KLD:   0.012345 ±   0.000123
        Same top p: 95.123 ± 0.123 %
    Keys are lower-cased with runs of whitespace collapsed: "mean kld", "same top p".
    Errors, when printed, go under "<key> err".
    """
    out: dict[str, float] = {}
    started = False
    for line in text.splitlines():
        if "KL divergence statistics" in line or "Perplexity statistics" in line:
            started = True
            continue
        if not started:
            continue
        m = _STAT.match(line)
        if not m:
            continue
        key = re.sub(r"\s+", " ", m.group(1)).strip().lower()
        out[key] = float(m.group(2))
        if m.group(3) is not None:
            out[f"{key} err"] = float(m.group(3))
    return out


def quantize_sizes(text: str) -> dict[str, float]:
    """Sizes llama-quantize prints, MiB and bits per weight over every tensor (embeddings included)."""
    out = {}
    if m := _MSIZE.search(text):
        out["src_MiB"], out["src_bpw"] = float(m.group(1)), float(m.group(2))
    if m := _QSIZE.search(text):
        out["quant_MiB"], out["quant_bpw"] = float(m.group(1)), float(m.group(2))
    return out


def wikitext_raw(rows: list[str]) -> str:
    """The HF parquet keeps one line of wiki.test.raw per row, newline included, except that the file's
    blank lines (a single space then newline) are stored as empty strings."""
    return "".join(r if r else " \n" for r in rows)
