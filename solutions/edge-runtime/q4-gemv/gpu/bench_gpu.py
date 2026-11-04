"""GPU side of q4-gemv: does a 4.5-bit weight make decode 3.5x faster on a GDDR7 card, and where does it stop?

  python gpu/bench_gpu.py check        # kernels vs dequantised reference, seconds
  python gpu/bench_gpu.py bandwidth    # read-only and copy GB/s on 4 GB, the ceiling for every row below
  python gpu/bench_gpu.py gemv         # M = 1, every method, Llama-7B and Qwen2.5-7B shapes, L2 cold and hot
  python gpu/bench_gpu.py sweep        # M = 1 .. 4096 on one MLP shape: decode -> prefill crossover
  python gpu/bench_gpu.py decode       # per-token time of a Qwen2.5-7B layer stack from the gemv rows

"Cold" rotates through enough copies of the weight that the working set is >= 4 GB, so every call reads
DRAM. A single 4096 x 4096 fp16 matrix (33.5 MB) fits in the L2 of current cards; without rotation the
benchmark measures L2 bandwidth and every format looks the same. Rows go to artifacts/raw/gpu/.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]
sys.path[:0] = [str(LAB / "gpu"), str(LAB / "code"), str(ROOT)]

import triton_gemv as tg  # noqa: E402
from edge_harness.writer import JsonlWriter, write_csv, write_json  # noqa: E402

RAW = LAB / "artifacts/raw/gpu"
FIG = LAB / "artifacts/figures"
COLD_BYTES = 4 << 30

# Qwen2.5-7B: hidden 3584, MLP 18944, 28 layers, 28 query heads / 4 KV heads of 128, untied lm_head.
QWEN7B = {"attn_q_o": (3584, 3584), "attn_kv": (512, 3584), "mlp_gate_up": (18944, 3584),
          "mlp_down": (3584, 18944), "lm_head": (152064, 3584)}
QWEN7B_PER_LAYER = {"attn_q_o": 2, "attn_kv": 2, "mlp_gate_up": 2, "mlp_down": 1}
QWEN7B_LAYERS = 28
SHAPES = {"llama7b_4096x4096": (4096, 4096), "llama7b_11008x4096": (11008, 4096),
          **{f"qwen7b_{k}": v for k, v in QWEN7B.items()}}


def device_info() -> dict:
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "sms": p.multi_processor_count, "cc": f"{p.major}.{p.minor}",
            "l2_MB": getattr(p, "L2_cache_size", 0) / 2**20, "mem_GB": p.total_memory / 1e9,
            "torch": torch.__version__, "triton": __import__("triton").__version__}


def _time(fn, n_calls: int, trials: int = 7, launch: str = "eager") -> float:
    """Median microseconds per call over `trials` CUDA-event windows of `n_calls` calls each.

    eager: Python launches every kernel, so a kernel shorter than the launch path (~20 us for a Triton
    call) is timed as the launch path. graph: the n_calls are captured once into a CUDA graph and
    replayed, which is what llama.cpp and vLLM do for decode; this is the kernel time.
    """
    for i in range(3):
        fn(i)
    torch.cuda.synchronize()
    run = lambda: [fn(i) for i in range(n_calls)]  # noqa: E731
    if launch == "graph":
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            run()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        run = g.replay
        run()
    torch.cuda.synchronize()
    out = []
    for _ in range(trials):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / n_calls)
    return statistics.median(out)


def _copies(nbytes: int, cold: bool) -> int:
    return max(2, min(512, math.ceil(COLD_BYTES / nbytes))) if cold else 1


def _rand_weight(n: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, k), dtype=np.float32) * 0.02).astype(np.float32)


def _replicate(p: dict, c: int) -> list[dict]:
    return [p] + [{**p, "qs": p["qs"].clone(), "d": p["d"].clone()} for _ in range(c - 1)]


def check() -> None:
    torch.manual_seed(0)
    for n, k in ((256, 4096), (1000, 11008), (512, 3584)):
        w = _rand_weight(n, k)
        x = torch.randn(k, device="cuda", dtype=torch.float16)
        a = torch.randn(37, k, device="cuda", dtype=torch.float16)
        ref16 = torch.from_numpy(w).cuda().half()
        err = (tg.gemv_fp16(ref16, x).float() - (ref16.float() @ x.float())).abs().max().item()
        print(f"{n}x{k} fp16 gemv max|err| {err:.2e}")
        for fmt in ("q8_0", "q4_0"):
            p = tg.pack_for_gpu(w, fmt)
            wd = tg.dequant_torch(p).float()
            y_ref = wd @ x.float()
            err = (tg.gemv(p, x).float() - y_ref).abs().max().item() / y_ref.abs().max().item()
            print(f"{n}x{k} {fmt} gemv rel err {err:.2e}")
            assert err < 2e-3, (fmt, err)
            if fmt == "q4_0":
                c_ref = a.float() @ wd.t()
                err = (tg.gemm_q4_0(p, a).float() - c_ref).abs().max().item() / c_ref.abs().max().item()
                print(f"{n}x{k} q4_0 tensor-core gemm rel err {err:.2e}")
                assert err < 5e-3, err
    print("check ok")


def bandwidth() -> dict:
    n = COLD_BYTES // 2
    t = torch.empty(n, device="cuda", dtype=torch.float16).uniform_()
    dst = torch.empty_like(t)
    rd = _time(lambda i: t.sum(dtype=torch.float32), 10)
    cp = _time(lambda i: dst.copy_(t), 10)
    out = {**device_info(), "bytes": COLD_BYTES, "read_GBps": COLD_BYTES / rd / 1e3,
           "copy_GBps": 2 * COLD_BYTES / cp / 1e3}
    print(out)
    write_json(RAW / "bandwidth.json", out)
    return out


def _methods(n: int, k: int, cold: bool):
    """Yield (method, weight_bytes, copies, call(i)) with the weight replicated for L2-cold timing."""
    w = _rand_weight(n, k)
    x = torch.randn(k, device="cuda", dtype=torch.float16)
    w16 = torch.from_numpy(w).cuda().half()
    c = _copies(w16.numel() * 2, cold)
    ws = [w16] + [w16.clone() for _ in range(c - 1)]
    yield "cublas_fp16", w16.numel() * 2, c, lambda i: torch.mv(ws[i % c], x)
    yield "triton_fp16", w16.numel() * 2, c, lambda i: tg.gemv_fp16(ws[i % c], x)
    del ws
    for fmt in ("q8_0", "q4_0"):
        p = tg.pack_for_gpu(w, fmt)
        nb = tg.weight_bytes(p)
        c = _copies(nb, cold)
        ps = _replicate(p, c)
        yield f"{fmt}_triton_gemv", nb, c, lambda i, ps=ps, c=c: tg.gemv(ps[i % c], x)
        if fmt == "q4_0":
            x2 = x.view(1, k)
            yield "q4_0_tensorcore_gemm_m1", nb, c, lambda i, ps=ps, c=c: tg.gemm_q4_0(ps[i % c], x2)
            yield "q4_0_dequant_then_cublas", nb, c, lambda i, ps=ps, c=c: torch.mv(tg.dequant_torch(ps[i % c]), x)
        del ps
    torch.cuda.empty_cache()


def gemv(shapes: list[str]) -> None:
    info = device_info()
    rows = []
    with JsonlWriter(RAW / "gemv_gpu.jsonl", cwd=ROOT) as wr:
        wr.write({"kind": "device", **info})
        for name in shapes:
            n, k = SHAPES[name]
            for cold in (True, False):
                for method, nb, c, fn in _methods(n, k, cold):
                    for launch in ("eager", "graph"):
                        us = _time(fn, max(c, 50), launch=launch)
                        row = {"shape": name, "n": n, "k": k, "method": method, "l2": "cold" if cold else "hot",
                               "launch": launch, "copies": c, "weight_bytes": nb, "us": us,
                               "weight_GBps": nb / us / 1e3}
                        print(row, flush=True)
                        wr.write(row)
                        rows.append(row)
    write_csv(RAW / "gemv_gpu.csv", rows)


def sweep(ms: list[int]) -> None:
    n, k = SHAPES["llama7b_11008x4096"]
    w = _rand_weight(n, k)
    w16 = torch.from_numpy(w).cuda().half()
    p = tg.pack_for_gpu(w, "q4_0")
    c16, c4 = _copies(w16.numel() * 2, True), _copies(tg.weight_bytes(p), True)
    ws = [w16] + [w16.clone() for _ in range(c16 - 1)]
    ps = _replicate(p, c4)
    rows = []
    for m in ms:
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        flops = 2 * m * n * k
        for method, c, fn in (
            ("cublas_fp16", c16, lambda i: a @ ws[i % c16].t()),
            ("q4_0_tensorcore_gemm", c4, lambda i: tg.gemm_q4_0(ps[i % c4], a)),
            ("q4_0_dequant_then_cublas", c4, lambda i: a @ tg.dequant_torch(ps[i % c4]).t()),
        ):
            for launch in ("eager", "graph"):
                us = _time(fn, max(c, 20), trials=5, launch=launch)
                row = {"m": m, "n": n, "k": k, "method": method, "launch": launch, "us": us,
                       "TFLOPs": flops / us / 1e6}
                print(row, flush=True)
                rows.append(row)
    write_csv(RAW / "batch_sweep.csv", rows)


def decode() -> None:
    import csv

    rows = list(csv.DictReader(open(RAW / "gemv_gpu.csv", encoding="utf-8")))
    bw = __import__("json").loads((RAW / "bandwidth.json").read_text(encoding="utf-8"))
    out = []
    methods = sorted({(r["method"], r.get("launch", "eager")) for r in rows})
    for method, launch in methods:
        sel = [r for r in rows if r["method"] == method and r.get("launch", "eager") == launch and r["l2"] == "cold"]
        t = {r["shape"]: float(r["us"]) for r in sel}
        b = {r["shape"]: int(r["weight_bytes"]) for r in sel}
        keys = [f"qwen7b_{s}" for s in QWEN7B]
        if not all(s in t for s in keys):
            continue
        layer_us = sum(QWEN7B_PER_LAYER[s] * t[f"qwen7b_{s}"] for s in QWEN7B_PER_LAYER)
        total_us = QWEN7B_LAYERS * layer_us + t["qwen7b_lm_head"]
        wbytes = QWEN7B_LAYERS * sum(QWEN7B_PER_LAYER[s] * b[f"qwen7b_{s}"] for s in QWEN7B_PER_LAYER) + b["qwen7b_lm_head"]
        out.append({"method": method, "launch": launch, "linear_weight_GB": wbytes / 1e9, "us_per_token": total_us,
                    "tok_s_linear_only": 1e6 / total_us, "tok_s_bound_at_read_bw": bw["read_GBps"] * 1e9 / wbytes,
                    "fraction_of_read_bw": wbytes / total_us / 1e3 / bw["read_GBps"]})
        print(out[-1])
    write_csv(RAW / "decode_pred.csv", out)


def plot() -> None:
    import csv

    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(RAW / "gemv_gpu.csv", encoding="utf-8")))
    bw = __import__("json").loads((RAW / "bandwidth.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    shape = "llama7b_4096x4096"
    methods = list(dict.fromkeys(r["method"] for r in rows))
    combos = [(l2, la) for l2 in ("cold", "hot") for la in ("eager", "graph")
              if any(r["l2"] == l2 and r.get("launch", "eager") == la for r in rows)]
    width = 0.8 / len(combos)
    for i, (l2, la) in enumerate(combos):
        vals = [next(float(r["weight_GBps"]) for r in rows if r["shape"] == shape and r["method"] == m
                     and r["l2"] == l2 and r.get("launch", "eager") == la) for m in methods]
        axes[0].bar(np.arange(len(methods)) + i * width, vals, width, label=f"L2 {l2}, {la}")
    axes[0].axhline(bw["read_GBps"], color="k", ls="--", lw=1, label=f"read-only {bw['read_GBps']:.0f} GB/s")
    axes[0].set_xticks(np.arange(len(methods)) + 0.4 - width / 2, methods, rotation=25, fontsize=7)
    axes[0].set_ylabel("weight bytes / time (GB/s)")
    axes[0].set_title(f"{shape}, M = 1")
    axes[0].legend(fontsize=7)
    if (RAW / "batch_sweep.csv").exists():
        sw = list(csv.DictReader(open(RAW / "batch_sweep.csv", encoding="utf-8")))
        for m in dict.fromkeys(r["method"] for r in sw):
            sel = [r for r in sw if r["method"] == m and r.get("launch", "graph") == "graph"]
            axes[1].plot([int(r["m"]) for r in sel], [float(r["us"]) for r in sel], marker="o", label=m)
        axes[1].set_xscale("log", base=2)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("M (tokens in the batch)")
        axes[1].set_ylabel("µs per 11008×4096 matmul")
        axes[1].set_title("W4A16 wins while memory-bound, loses once compute-bound (CUDA graph)", fontsize=9)
        axes[1].legend(fontsize=7)
        axes[1].grid(True, which="both", alpha=0.3)
    fig.suptitle(bw.get("gpu", ""))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "gemv_gpu.png", dpi=150)
    print("wrote gemv_gpu.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["check", "bandwidth", "gemv", "sweep", "decode", "plot", "all"])
    ap.add_argument("--shapes", nargs="*", default=list(SHAPES))
    ap.add_argument("--ms", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096")
    args = ap.parse_args()
    steps = ["check", "bandwidth", "gemv", "sweep", "decode", "plot"] if "all" in args.steps else args.steps
    RAW.mkdir(parents=True, exist_ok=True)
    for s in steps:
        if s == "gemv":
            gemv(args.shapes)
        elif s == "sweep":
            sweep([int(m) for m in args.ms.split(",")])
        else:
            globals()[s]()


if __name__ == "__main__":
    main()
