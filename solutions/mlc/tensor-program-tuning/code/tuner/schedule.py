"""A tiny tensor-program search space for C = A @ B (row-major, fp32), rendered to C++.

A schedule fixes, in the TensorIR / Halide sense:

* ``outer``  order of the three tile loops (i0, j0, k0), e.g. "ijk";
* ``inner``  order of the three point loops inside a tile, e.g. "ikj" (j innermost vectorizes);
* ``ti, tj, tk``  tile sizes, each dividing the problem size;
* ``pack_b`` whether B is pre-transformed to tile-major [N/tj][K][tj] (a layout rewrite of a constant
  weight, done once at compile time, like TVM's AlterOpLayout);
* ``unroll`` the unroll factor requested on the innermost loop.

Sizes are compile-time constants in the generated code, as they are after shape specialization.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import asdict, dataclass

ORDERS = ["".join(p) for p in itertools.permutations("ijk")]


@dataclass(frozen=True)
class Problem:
    name: str
    M: int
    N: int
    K: int

    @property
    def flops(self) -> int:
        return 2 * self.M * self.N * self.K


@dataclass(frozen=True)
class Schedule:
    outer: str = "ijk"
    inner: str = "ikj"
    ti: int = 32
    tj: int = 64
    tk: int = 64
    pack_b: bool = False
    unroll: int = 1

    def key(self) -> str:
        return f"o{self.outer}_i{self.inner}_t{self.ti}x{self.tj}x{self.tk}_p{int(self.pack_b)}_u{self.unroll}"

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_key(cls, key: str) -> Schedule:
        o, i, t, p, u = key.split("_")
        ti, tj, tk = map(int, t[1:].split("x"))
        return cls(o[1:], i[1:], ti, tj, tk, p == "p1", int(u[1:]))


def divisors(n: int, options: tuple[int, ...]) -> list[int]:
    return [d for d in options if d <= n and n % d == 0]


class Space:
    TI = (1, 2, 4, 8, 16, 32, 64)
    TJ = (16, 32, 64, 128, 256)
    TK = (8, 16, 32, 64, 128, 256, 512)

    def __init__(self, p: Problem):
        self.p = p
        self.ti = divisors(p.M, self.TI)
        self.tj = divisors(p.N, self.TJ)
        self.tk = divisors(p.K, self.TK)

    def all(self) -> list[Schedule]:
        return [Schedule(o, i, a, b, c, pk, u)
                for o in ORDERS for i in ORDERS for a in self.ti for b in self.tj for c in self.tk
                for pk in (False, True) for u in (1, 4)]

    def size(self) -> int:
        return len(ORDERS) ** 2 * len(self.ti) * len(self.tj) * len(self.tk) * 4

    def sample(self, rng: random.Random) -> Schedule:
        return Schedule(rng.choice(ORDERS), rng.choice(ORDERS), rng.choice(self.ti), rng.choice(self.tj),
                        rng.choice(self.tk), rng.random() < 0.5, rng.choice((1, 4)))

    def neighbors(self, s: Schedule, rng: random.Random, n: int) -> list[Schedule]:
        """Mutate one knob at a time (the moves an evolutionary search makes)."""
        out = []
        for _ in range(n):
            knob = rng.choice(("outer", "inner", "ti", "tj", "tk", "pack_b", "unroll"))
            d = s.as_dict()
            d[knob] = {"outer": lambda: rng.choice(ORDERS), "inner": lambda: rng.choice(ORDERS),
                       "ti": lambda: rng.choice(self.ti), "tj": lambda: rng.choice(self.tj),
                       "tk": lambda: rng.choice(self.tk), "pack_b": lambda: not s.pack_b,
                       "unroll": lambda: 5 - s.unroll}[knob]()
            out.append(Schedule(**d))
        return out


def render(p: Problem, s: Schedule, fn: str) -> str:
    """One C++ function ``void fn(const float* A, const float* B, const float* Bp, float* C)``."""
    for dim, tile in ((p.M, s.ti), (p.N, s.tj), (p.K, s.tk)):
        if dim % tile:
            raise ValueError(f"{s.key()}: tile {tile} does not divide {dim}; the loops have no remainder handling")
    var = {"i": ("i0", "ti", "M"), "j": ("j0", "tj", "N"), "k": ("k0", "tk", "K")}
    size = {"M": p.M, "N": p.N, "K": p.K, "ti": s.ti, "tj": s.tj, "tk": s.tk}
    lines = [f"static void {fn}(const float* __restrict A, const float* __restrict B, "
             f"const float* __restrict Bp, float* __restrict C) {{"]
    ind = "    "
    for d in s.outer:
        v, t, n = var[d]
        lines.append(f"{ind}for (int {v} = 0; {v} < {size[n]}; {v} += {size[t]})")
        ind += "    "
    lines.append(f"{ind}{{")
    ind += "    "
    if s.pack_b:
        # Bp holds B as [N/tj][K][tj]: the tile for (k0, j0) is contiguous rows of tj floats.
        lines.append(f"{ind}const float* __restrict bt = Bp + (size_t)(j0 / {s.tj}) * {p.K * s.tj} + (size_t)k0 * {s.tj};")
        b_expr = f"bt[(size_t)kk * {s.tj} + jj]"
    else:
        b_expr = f"B[(size_t)(k0 + kk) * {p.N} + j0 + jj]"
    loop = {"i": f"for (int ii = 0; ii < {s.ti}; ++ii)", "j": f"for (int jj = 0; jj < {s.tj}; ++jj)",
            "k": f"for (int kk = 0; kk < {s.tk}; ++kk)"}
    for n, d in enumerate(s.inner):
        if n == 2 and s.unroll > 1:
            lines.append(f"{ind}#pragma GCC unroll {s.unroll}")
        lines.append(f"{ind}{loop[d]}")
        ind += "    "
    lines.append(f"{ind}C[(size_t)(i0 + ii) * {p.N} + j0 + jj] += A[(size_t)(i0 + ii) * {p.K} + k0 + kk] * {b_expr};")
    lines.append("    " * (len(s.outer) + 1) + "}")
    lines.append("}")
    return "\n".join(lines)


def footprint_bytes(s: Schedule) -> dict[str, int]:
    """Bytes of each operand tile touched by one innermost tile computation."""
    return {"A": 4 * s.ti * s.tk, "B": 4 * s.tk * s.tj, "C": 4 * s.ti * s.tj}
