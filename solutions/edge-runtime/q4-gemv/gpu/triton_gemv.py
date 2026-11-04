"""GGUF block formats on the GPU: Triton GEMV for decode (M = 1) and a W4A16 tensor-core GEMM for M > 1.

Weights are packed by ``blockq.pack`` (byte-identical to ggml) and then split into two device arrays,
structure of arrays instead of ggml's array of structs, so every load is a contiguous vector:

  Q8_0  qs int8  [N, K]       d fp16 [N, K/32]
  Q4_0  qs uint8 [N, K/2]     d fp16 [N, K/32]   byte j of a block = element j (low) | element j+16 (high)

W is [N, K] (out, in) as in GGUF; y = W x. The GEMV kernels use CUDA cores only: at M = 1 every
weight byte is used once, so tensor cores have nothing to amortise. ``gemm_q4_0`` dequantises a tile
in registers and feeds ``tl.dot``, the Marlin / AWQ-kernel idea, for the prefill side of the sweep.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

QK = 32


def pack_for_gpu(w: np.ndarray, fmt: str, device: str = "cuda") -> dict:
    """Pack an [N, K] fp32 matrix with the ggml-compatible packer and move it to SoA device arrays."""
    from blockq.pack import BLOCK_BYTES, QUANTIZERS

    n, k = w.shape
    raw = QUANTIZERS[fmt](w).reshape(n, k // QK, BLOCK_BYTES[fmt])
    d = np.ascontiguousarray(raw[:, :, :2]).view(np.float16).reshape(n, k // QK)
    qs = np.ascontiguousarray(raw[:, :, 2:]).reshape(n, -1)
    if fmt == "q8_0":
        qs = qs.view(np.int8)
    elif fmt != "q4_0":
        raise ValueError(fmt)
    return {"fmt": fmt, "n": n, "k": k,
            "qs": torch.from_numpy(qs).to(device), "d": torch.from_numpy(d).to(device)}


def dequant_torch(p: dict) -> torch.Tensor:
    """Reference dequantisation on the device, [N, K] fp16."""
    n, k = p["n"], p["k"]
    d = p["d"].float().view(n, k // QK, 1)
    if p["fmt"] == "q8_0":
        q = p["qs"].float().view(n, k // QK, QK)
    else:
        b = p["qs"].view(n, k // QK, 16)
        q = torch.cat([(b & 15), (b >> 4)], dim=2).float() - 8
    return (q * d).view(n, k).half()


def weight_bytes(p: dict) -> int:
    return p["qs"].numel() * p["qs"].element_size() + p["d"].numel() * 2


_GEMV_CONFIGS = [triton.Config({"BN": bn, "BK": bk}, num_warps=w)
                 for bn in (1, 2, 4, 8) for bk in (256, 512, 1024) for w in (2, 4, 8)]


@triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K"])
@triton.jit
def _gemv_fp16(w_ptr, x_ptr, y_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        cols = k0 + tl.arange(0, BK)
        kmask = cols < K
        x = tl.load(x_ptr + cols, mask=kmask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + rows[:, None] * K + cols[None, :], mask=(rows[:, None] < N) & kmask[None, :], other=0.0)
        acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    tl.store(y_ptr + rows, acc.to(tl.float16), mask=rows < N)


@triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K"])
@triton.jit
def _gemv_q8_0(qs_ptr, d_ptr, x_ptr, y_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr):
    NB: tl.constexpr = BK // 32
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    rmask = rows < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    blk = tl.arange(0, NB)
    j = tl.arange(0, 32)
    for k0 in range(0, K, BK):
        cols = k0 + blk[:, None] * 32 + j[None, :]                                    # [NB, 32]
        bmask = (k0 + blk * 32) < K
        x = tl.load(x_ptr + cols, mask=bmask[:, None], other=0.0).to(tl.float32)
        q = tl.load(qs_ptr + rows[:, None, None] * K + cols[None, :, :],
                    mask=rmask[:, None, None] & bmask[None, :, None], other=0)
        d = tl.load(d_ptr + rows[:, None] * (K // 32) + (k0 // 32 + blk)[None, :],
                    mask=rmask[:, None] & bmask[None, :], other=0.0)
        part = tl.sum(q.to(tl.float32) * x[None, :, :], axis=2)                       # [BN, NB]
        acc += tl.sum(part * d.to(tl.float32), axis=1)
    tl.store(y_ptr + rows, acc.to(tl.float16), mask=rmask)


@triton.autotune(configs=_GEMV_CONFIGS, key=["N", "K"])
@triton.jit
def _gemv_q4_0(qs_ptr, d_ptr, x_ptr, y_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr):
    NB: tl.constexpr = BK // 32
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    rmask = rows < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    blk = tl.arange(0, NB)
    j = tl.arange(0, 16)
    for k0 in range(0, K, BK):
        base = k0 + blk[:, None] * 32 + j[None, :]                                    # [NB, 16]
        bmask = (k0 + blk * 32) < K
        x_lo = tl.load(x_ptr + base, mask=bmask[:, None], other=0.0).to(tl.float32)
        x_hi = tl.load(x_ptr + base + 16, mask=bmask[:, None], other=0.0).to(tl.float32)
        byte_off = (k0 // 2) + blk[:, None] * 16 + j[None, :]
        b = tl.load(qs_ptr + rows[:, None, None] * (K // 2) + byte_off[None, :, :],
                    mask=rmask[:, None, None] & bmask[None, :, None], other=0x88)
        lo = (b & 15).to(tl.float32) - 8.0
        hi = (b >> 4).to(tl.float32) - 8.0
        part = tl.sum(lo * x_lo[None, :, :] + hi * x_hi[None, :, :], axis=2)
        d = tl.load(d_ptr + rows[:, None] * (K // 32) + (k0 // 32 + blk)[None, :],
                    mask=rmask[:, None] & bmask[None, :], other=0.0)
        acc += tl.sum(part * d.to(tl.float32), axis=1)
    tl.store(y_ptr + rows, acc.to(tl.float16), mask=rmask)


_GEMM_CONFIGS = [triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=s)
                 for bm in (16, 64, 128) for bn in (64, 128) for bk in (64, 128) for w in (4, 8) for s in (3, 4)]


@triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _gemm_q4_0(a_ptr, qs_ptr, d_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """C[M, N] = A[M, K] @ W^T with W in Q4_0. A's columns are loaded in the order the nibbles unpack."""
    NB: tl.constexpr = BK // 32
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    t = tl.arange(0, BK)
    # Flattened (block, j, lo/hi) -> element block*32 + hi*16 + j, matching tl.join(lo, hi) below.
    perm = (t // 32) * 32 + (t % 2) * 16 + (t % 32) // 2
    blk = tl.arange(0, NB)
    j = tl.arange(0, 16)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k0 + perm)[None, :], mask=(rm[:, None] < M) & ((k0 + t) < K)[None, :], other=0.0)
        bmask = (k0 + blk * 32) < K
        byte_off = (k0 // 2) + blk[:, None] * 16 + j[None, :]
        b = tl.load(qs_ptr + rn[:, None, None] * (K // 2) + byte_off[None, :, :],
                    mask=(rn < N)[:, None, None] & bmask[None, :, None], other=0x88)
        d = tl.load(d_ptr + rn[:, None] * (K // 32) + (k0 // 32 + blk)[None, :],
                    mask=(rn < N)[:, None] & bmask[None, :], other=0.0)
        lo = ((b & 15).to(tl.float16) - 8.0) * d[:, :, None]
        hi = ((b >> 4).to(tl.float16) - 8.0) * d[:, :, None]
        w = tl.reshape(tl.join(lo, hi), (BN, BK))
        acc = tl.dot(a, tl.trans(w), acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def gemv_fp16(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    n, k = w.shape
    y = torch.empty(n, device=x.device, dtype=torch.float16)
    _gemv_fp16[lambda m: (triton.cdiv(n, m["BN"]),)](w, x, y, n, k)
    return y


def gemv(p: dict, x: torch.Tensor) -> torch.Tensor:
    n, k = p["n"], p["k"]
    y = torch.empty(n, device=x.device, dtype=torch.float16)
    fn = _gemv_q8_0 if p["fmt"] == "q8_0" else _gemv_q4_0
    fn[lambda m: (triton.cdiv(n, m["BN"]),)](p["qs"], p["d"], x, y, n, k)
    return y


def gemm_q4_0(p: dict, a: torch.Tensor) -> torch.Tensor:
    assert p["fmt"] == "q4_0"
    m, k = a.shape
    n = p["n"]
    c = torch.empty(m, n, device=a.device, dtype=torch.float16)
    _gemm_q4_0[lambda g: (triton.cdiv(m, g["BM"]), triton.cdiv(n, g["BN"]))](a, p["qs"], p["d"], c, m, n, k)
    return c
