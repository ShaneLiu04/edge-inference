"""Whiteboard formulas for on-device deployment interviews.

Every function is closed-form and should be derivable without a board full of
framework API. Derivations live in interview/FORMULAS.md.

Examples
--------
    python tools/edge_calc.py gemv --n 4096 --k 4096 --dtype int8
    python tools/edge_calc.py qblock --params 5e8 --dtype q4_0 --bw 3.4e10
    python tools/edge_calc.py pipeline --tiles 8 --dma 2 --compute 3
    python tools/edge_calc.py bus --mt-s 4266 --bits 64
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GB = 1e9
MB = 1e6


@dataclass(frozen=True)
class DType:
    name: str
    weight_bytes: float
    act_bytes: float


DTYPES = {
    "fp32": DType("fp32", 4.0, 4.0),
    "fp16": DType("fp16", 2.0, 2.0),
    "int8": DType("int8", 1.0, 1.0),
}


def bus_bandwidth(mt_per_s: float, bits: int) -> float:
    """Peak DRAM bandwidth in bytes/s. mt_per_s is the data rate in MT/s."""
    if mt_per_s <= 0 or bits <= 0:
        raise ValueError("data rate and bus width must be positive")
    return mt_per_s * 1e6 * (bits / 8)


def sram_bandwidth_upper(clock_hz: float, bus_bytes: int) -> float:
    """Upper bound if the core issues one load per cycle and SRAM has no wait states."""
    if clock_hz <= 0 or bus_bytes <= 0:
        raise ValueError("clock and bus width must be positive")
    return clock_hz * bus_bytes


def ridge(peak_ops: float, bandwidth: float) -> float:
    """FLOP/byte where compute time equals memory time."""
    if peak_ops <= 0 or bandwidth <= 0:
        raise ValueError("peak and bandwidth must be positive")
    return peak_ops / bandwidth


def gemv_flops(n: int, k: int) -> float:
    """y = W x, W is n by k. One multiply and one add per product."""
    return 2.0 * n * k


def gemv_bytes(n: int, k: int, dtype: str) -> float:
    """Weights, the vector, and the output, each touched once. No cache credit."""
    spec = DTYPES[dtype]
    return spec.weight_bytes * n * k + spec.act_bytes * k + spec.act_bytes * n


def gemv_intensity(n: int, k: int, dtype: str) -> float:
    return gemv_flops(n, k) / gemv_bytes(n, k, dtype)


def weight_only_gemv_bytes(n: int, k: int, weight_bytes: float, act_bytes: float) -> float:
    """Int8 or Q4 weights with a wider activation/accumulator store."""
    return weight_bytes * n * k + act_bytes * (k + n)


def time_lower_bound(nbytes: float, bandwidth: float) -> float:
    """Seconds if the kernel is memory-bound and hits this bandwidth."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    return nbytes / bandwidth


def affine(q: float, scale: float, zero_point: float) -> float:
    """real = scale * (q - zero_point). Symmetric quantization uses zero_point = 0."""
    return scale * (q - zero_point)


def requant_mul(weight_scale: float, input_scale: float, output_scale: float) -> float:
    """Integer accumulator is in units of weight_scale * input_scale."""
    if output_scale == 0:
        raise ValueError("output scale must be non-zero")
    return (weight_scale * input_scale) / output_scale


def bias_int(bias_real: float, weight_scale: float, input_scale: float) -> float:
    """Real bias divided into the int32 accumulator, before rounding."""
    denom = weight_scale * input_scale
    if denom == 0:
        raise ValueError("scales must be non-zero")
    return bias_real / denom


def bn_fold_scale(gamma: float, var: float, eps: float = 1e-5) -> float:
    if var + eps <= 0:
        raise ValueError("variance + eps must be positive")
    return gamma / (var + eps) ** 0.5


def depthwise_intensity(kh: int, kw: int, elem_bytes: float) -> float:
    """FLOP/byte if each input is loaded once and weights are negligible.

    A 3x3 depthwise layer has almost no weight reuse across channels. The
    intensity is 2*KH*KW/elem_bytes only when the sliding window hits in SRAM.
    A cache miss per tap divides this by KH*KW.
    """
    if kh <= 0 or kw <= 0 or elem_bytes <= 0:
        raise ValueError("kernel and element size must be positive")
    return (2.0 * kh * kw) / elem_bytes


def im2col_workspace(oh: int, ow: int, kh: int, kw: int, ic: int, elem_bytes: float) -> int:
    """Extra activation bytes for an explicit im2col buffer. Direct conv does not allocate this."""
    return int(oh * ow * kh * kw * ic * elem_bytes)


def qblock_bytes_per_param(kind: str) -> float:
    """llama.cpp-style block quant, block length 32.

    Q8_0: 32 int8 values + fp16 scale = 34 bytes.
    Q4_0: 32 4-bit values + fp16 scale = 18 bytes.
    """
    if kind == "q8_0":
        return 34 / 32
    if kind == "q4_0":
        return 18 / 32
    raise ValueError(f"unknown block quant {kind}")


def pipeline_time(n_tiles: int, dma: float, compute: float) -> float:
    """Double-buffered tiling. Single buffer is n_tiles * (dma + compute).

    T = n * max(dma, compute) + min(dma, compute). Overlap does not create
    bandwidth: if dma is already the longer side, the bound stays dma.
    """
    if n_tiles < 1:
        raise ValueError("need at least one tile")
    if dma < 0 or compute < 0:
        raise ValueError("times must be non-negative")
    return n_tiles * max(dma, compute) + min(dma, compute)


def single_buffer_time(n_tiles: int, dma: float, compute: float) -> float:
    if n_tiles < 1:
        raise ValueError("need at least one tile")
    return n_tiles * (dma + compute)


@dataclass(frozen=True)
class Buffer:
    start: int
    end: int
    size: int

    def live_at(self, t: int) -> bool:
        return self.start <= t <= self.end


def peak_live(buffers: list[Buffer]) -> int:
    """Fragmentation-free lower bound: maximum sum of simultaneously live sizes."""
    if not buffers:
        return 0
    times = {b.start for b in buffers} | {b.end for b in buffers}
    return max(sum(b.size for b in buffers if b.live_at(t)) for t in times)


def first_fit_arena(buffers: list[Buffer]) -> int:
    """Address-space high-water mark of first-fit. Can exceed peak_live."""
    if any(b.size < 0 or b.end < b.start for b in buffers):
        raise ValueError("sizes must be non-negative and intervals ordered")
    order = sorted(range(len(buffers)), key=lambda i: (buffers[i].start, i))
    offsets = [-1] * len(buffers)
    for i in order:
        buf = buffers[i]
        occupied: list[tuple[int, int]] = []
        for j, off in enumerate(offsets):
            if off < 0:
                continue
            other = buffers[j]
            if other.end < buf.start or other.start > buf.end:
                continue
            occupied.append((off, off + other.size))
        occupied.sort()
        cursor = 0
        placed = None
        for lo, hi in occupied:
            if cursor + buf.size <= lo:
                placed = cursor
                break
            cursor = max(cursor, hi)
        offsets[i] = cursor if placed is None else placed
    if not buffers:
        return 0
    return max(offsets[i] + buffers[i].size for i in range(len(buffers)))


def _cmd_gemv(args: argparse.Namespace) -> None:
    intensity = gemv_intensity(args.n, args.k, args.dtype)
    flops = gemv_flops(args.n, args.k)
    nbytes = gemv_bytes(args.n, args.k, args.dtype)
    print(f"dtype {args.dtype}  N={args.n} K={args.k}")
    print(f"FLOPs {flops:.6g}  bytes {nbytes:.6g}  intensity {intensity:.4f} FLOP/byte")
    if args.bw:
        print(f"memory lower bound {time_lower_bound(nbytes, args.bw) * 1e3:.4f} ms at {args.bw / GB:.3g} GB/s")


def _cmd_qblock(args: argparse.Namespace) -> None:
    bpp = qblock_bytes_per_param(args.dtype)
    nbytes = args.params * bpp
    print(f"{args.dtype}  {bpp:.4f} bytes/param  weights {nbytes / MB:.3f} MB")
    if args.bw:
        print(f"one full weight read {time_lower_bound(nbytes, args.bw) * 1e3:.3f} ms")


def _cmd_pipeline(args: argparse.Namespace) -> None:
    single = single_buffer_time(args.tiles, args.dma, args.compute)
    double = pipeline_time(args.tiles, args.dma, args.compute)
    print(f"tiles {args.tiles}  dma {args.dma}  compute {args.compute}")
    print(f"single {single:.6g}  double {double:.6g}  speedup {single / double:.3f}")


def _cmd_bus(args: argparse.Namespace) -> None:
    bw = bus_bandwidth(args.mt_s, args.bits)
    print(f"LPDDR {args.mt_s:g} MT/s x {args.bits} bit = {bw / GB:.3f} GB/s peak")


def _cmd_sram(args: argparse.Namespace) -> None:
    bw = sram_bandwidth_upper(args.clock_mhz * 1e6, args.bus_bytes)
    print(f"{args.clock_mhz:g} MHz x {args.bus_bytes} B = {bw / MB:.3f} MB/s upper bound")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="On-device deployment whiteboard calculator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gemv")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    p.add_argument("--bw", type=float, default=0.0, help="bytes/s, optional")
    p.set_defaults(func=_cmd_gemv)

    p = sub.add_parser("qblock")
    p.add_argument("--params", type=float, required=True)
    p.add_argument("--dtype", choices=("q4_0", "q8_0"), required=True)
    p.add_argument("--bw", type=float, default=0.0)
    p.set_defaults(func=_cmd_qblock)

    p = sub.add_parser("pipeline")
    p.add_argument("--tiles", type=int, required=True)
    p.add_argument("--dma", type=float, required=True)
    p.add_argument("--compute", type=float, required=True)
    p.set_defaults(func=_cmd_pipeline)

    p = sub.add_parser("bus")
    p.add_argument("--mt-s", type=float, required=True, dest="mt_s")
    p.add_argument("--bits", type=int, required=True)
    p.set_defaults(func=_cmd_bus)

    p = sub.add_parser("sram")
    p.add_argument("--clock-mhz", type=float, required=True)
    p.add_argument("--bus-bytes", type=int, default=4)
    p.set_defaults(func=_cmd_sram)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
