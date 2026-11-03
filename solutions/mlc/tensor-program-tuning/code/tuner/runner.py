"""Compile a batch of schedules into one executable, run it, return GFLOP/s and correctness per schedule."""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .schedule import Problem, Schedule, render

HARNESS = r"""
#include "edgebench.h"
#include <cmath>
using namespace eb;
constexpr int M = {M}, N = {N}, K = {K};
{kernels}
typedef void (*Kern)(const float*, const float*, const float*, float*);
struct Entry {{ const char* key; Kern fn; int tj; }};
static const Entry entries[] = {{ {table} }};
int main(int argc, char** argv) {{
    int repeats = static_cast<int>(arg_ll(argc, argv, "repeats", 5));
    Buffer<float> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(K) * N), C(static_cast<size_t>(M) * N),
                  R(static_cast<size_t>(M) * N), Bp(static_cast<size_t>(K) * N);
    for (size_t i = 0; i < A.n; ++i) A[i] = static_cast<float>((i * 7) % 13) / 13.0f - 0.5f;
    for (size_t i = 0; i < B.n; ++i) B[i] = static_cast<float>((i * 5) % 11) / 11.0f - 0.5f;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {{
            double s = 0;
            for (int k = 0; k < K; ++k) s += static_cast<double>(A[static_cast<size_t>(i) * K + k]) * B[static_cast<size_t>(k) * N + j];
            R[static_cast<size_t>(i) * N + j] = static_cast<float>(s);
        }}
    for (const Entry& e : entries) {{
        for (int jb = 0; jb < N / e.tj; ++jb)
            for (int k = 0; k < K; ++k)
                for (int jj = 0; jj < e.tj; ++jj)
                    Bp[(static_cast<size_t>(jb) * K + k) * e.tj + jj] = B[static_cast<size_t>(k) * N + jb * e.tj + jj];
        auto run = [&] {{ std::memset(C.data(), 0, C.n * sizeof(float)); e.fn(A.data(), B.data(), Bp.data(), C.data()); clobber(); }};
        run();
        double err = 0, ref = 0;
        for (size_t i = 0; i < C.n; ++i) {{ err = std::max(err, static_cast<double>(std::fabs(C[i] - R[i]))); ref = std::max(ref, static_cast<double>(std::fabs(R[i]))); }}
        Stats st = time_fn(run, 1, repeats, 2e-2);
        Json().s("key", e.key).f("gflops", 2.0 * M * N * K / st.p50 / 1e9).f("max_rel_err", err / ref).stats(st).emit();
    }}
    return 0;
}}
"""


def source(p: Problem, batch: list[Schedule]) -> str:
    kernels = "\n\n".join(render(p, s, f"k{n}") for n, s in enumerate(batch))
    table = ", ".join(f'{{"{s.key()}", k{n}, {s.tj}}}' for n, s in enumerate(batch))
    return HARNESS.format(M=p.M, N=p.N, K=p.K, kernels=kernels, table=table)


def compile_batch(p: Problem, batch: list[Schedule], work: Path, include: Path, cxx: str = "g++",
                  flags: tuple[str, ...] = ()) -> Path:
    from edge_harness.cpp import DEFAULT_FLAGS, EXE, PLATFORM_FLAGS

    src = source(p, batch)
    key = hashlib.sha1((src + " ".join(flags)).encode()).hexdigest()[:12]
    work.mkdir(parents=True, exist_ok=True)
    cpp = work / f"tune_{p.name}_{key}.cpp"
    exe = work / f"tune_{p.name}_{key}{EXE}"
    if not exe.exists():
        cpp.write_text(src, encoding="utf-8")
        cmd = [cxx, *DEFAULT_FLAGS, *PLATFORM_FLAGS, *flags, f"-I{include}", str(cpp), "-o", str(exe)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:])
    return exe


def measure(p: Problem, schedules: list[Schedule], work: Path, include: Path, batch: int = 16,
            jobs: int = 4, repeats: int = 5, flags: tuple[str, ...] = ()) -> dict[str, dict]:
    """Compile in parallel (one g++ per core), then run the binaries one at a time so timings do not overlap.

    ``flags`` are appended to the compile line, e.g. ``("-mprefer-vector-width=512",)``: GCC's default
    on AVX-512 parts is 256-bit vectors for auto-vectorized loops.
    """
    batches = [schedules[i:i + batch] for i in range(0, len(schedules), batch)]
    with ThreadPoolExecutor(jobs) as pool:
        exes = list(pool.map(lambda b: compile_batch(p, b, work, include, flags=flags), batches))
    out: dict[str, dict] = {}
    for exe in exes:
        proc = subprocess.run([str(exe), f"--repeats={repeats}"], capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(f"{exe.name} exited {proc.returncode}: {proc.stderr[-500:]}")
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                row = json.loads(line)
                out[row["key"]] = row
    return out
