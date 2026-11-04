"""Build and run the small C++ benchmarks with g++.

Binaries are cached under ``<lab>/cpp/bin`` keyed by a hash of the source and
flags, so a test run does not recompile on every call. Benchmarks print one
JSON object per line on stdout; ``run_json`` parses them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

# EDGE_MARCH=haswell builds everything as an AVX2-only machine would, to exercise the fallbacks.
DEFAULT_FLAGS = ("-O3", "-std=c++17", f"-march={os.environ.get('EDGE_MARCH', 'native')}", "-fno-math-errno")
EXE = ".exe" if os.name == "nt" else ""
# MinGW keeps the Windows 16-byte stack alignment, so a spilled zmm register written with
# vmovdqa64 faults. The assembler can rewrite aligned vector moves as unaligned ones.
PLATFORM_FLAGS = ("-Wa,-muse-unaligned-vector-move",) if os.name == "nt" else ()


def bench_threads(default: tuple[int, ...] = (1, 4)) -> tuple[int, ...]:
    """Thread counts for a sweep. EDGE_THREADS=1,2,4,8,16 widens it on a bigger host."""
    env = os.environ.get("EDGE_THREADS", "").strip()
    return tuple(int(t) for t in env.split(",")) if env else default


class BuildError(RuntimeError):
    pass


def compiler() -> str | None:
    for name in ("g++", "clang++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def build(src: Path, out_dir: Path, flags: tuple[str, ...] = DEFAULT_FLAGS,
          extra: tuple[str, ...] = (), includes: tuple[Path, ...] = ()) -> Path:
    cxx = compiler()
    if cxx is None:
        raise BuildError("no C++ compiler on PATH")
    src = Path(src)
    sources = [src.read_bytes()]
    for inc in includes:
        for header in sorted(Path(inc).glob("*.h")):
            sources.append(header.read_bytes())
    extra = (*PLATFORM_FLAGS, *extra)
    key = hashlib.sha1(b"\0".join(sources) + " ".join((*flags, *extra)).encode()).hexdigest()[:12]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / f"{src.stem}-{key}{EXE}"
    if exe.exists():
        return exe
    cmd = [cxx, *flags, *extra, *(f"-I{p}" for p in includes), str(src), "-o", str(exe)]
    if os.name != "nt":
        cmd.append("-lpthread")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise BuildError(f"{' '.join(cmd)}\n{proc.stderr}")
    return exe


def run_json(exe: Path, args: list[str] | tuple[str, ...] = (), timeout: float = 600) -> list[dict]:
    proc = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise BuildError(f"{exe.name} exited {proc.returncode}\n{proc.stderr}")
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows
