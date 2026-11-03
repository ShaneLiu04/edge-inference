"""Winograd F(m, r) transforms from Toom-Cook interpolation, plus the numerics that limit them.

Construction (transposition of Toom-Cook polynomial multiplication, as in Lavin & Gray 2016
and the wincnn generator): pick n - 1 = m + r - 2 finite points a_j and the point at infinity.

* G (n x r):  G[j, k] = a_j^k / f_j with f_j = prod_{l != j} (a_j - a_l);  infinity row = e_{r-1}.
* A (n x m):  A[j, i] = a_j^i;                                             infinity row = e_{m-1}.
* B (n x n):  column j holds the coefficients of M_j(x) = prod_{l != j} (x - a_l);
              the infinity column holds those of M(x) = prod_l (x - a_l).

1-D correlation y_i = sum_k g_k d_{i+k} is then y = A^T [(G g) * (B^T d)], and the 2-D tile is
Y = A^T [(G g G^T) * (B^T d B)] A. Matrices are built in exact rationals and converted at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

INF = None

# Point sets used in practice: cuDNN offers F(2,3) and F(4,3); NNPACK's wt8x8 is F(6,3).
POINTS = {
    2: [0, 1, -1, INF],
    4: [0, 1, -1, 2, -2, INF],
    6: [0, 1, -1, 2, -2, Fraction(1, 2), Fraction(-1, 2), INF],
}


def _poly_mul(p: list[Fraction], q: list[Fraction]) -> list[Fraction]:
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out


def _poly_from_roots(roots: list[Fraction]) -> list[Fraction]:
    """Coefficients, lowest degree first."""
    p = [Fraction(1)]
    for r in roots:
        p = _poly_mul(p, [-r, Fraction(1)])
    return p


@dataclass(frozen=True)
class Transform:
    m: int
    r: int
    points: tuple
    AT: np.ndarray  # (m, n)
    G: np.ndarray   # (n, r)
    BT: np.ndarray  # (n, n)

    @property
    def n(self) -> int:
        return self.m + self.r - 1

    def mults_per_output_2d(self) -> float:
        return self.n ** 2 / self.m ** 2

    def reduction_2d(self) -> float:
        """Multiplications of direct conv over Winograd, element-wise stage only."""
        return self.r ** 2 / self.mults_per_output_2d()

    def input_growth_2d(self) -> float:
        """max |B^T d B| / max |d|: the L1 row norm of B^T, squared. Sets the bits int8 inputs need."""
        return float(np.abs(self.BT).sum(axis=1).max() ** 2)


def transform(m: int, r: int = 3, points: list | None = None) -> Transform:
    pts = list(POINTS[m] if points is None else points)
    n = m + r - 1
    if len(pts) != n:
        raise ValueError(f"F({m},{r}) needs {n} points, got {len(pts)}")
    if pts[-1] is not INF:
        raise ValueError("last point must be infinity")
    finite = [Fraction(p) for p in pts[:-1]]
    if len(set(finite)) != len(finite):
        raise ValueError("points must be distinct")

    G = [[Fraction(0)] * r for _ in range(n)]
    A = [[Fraction(0)] * m for _ in range(n)]
    B = [[Fraction(0)] * n for _ in range(n)]
    for j, a in enumerate(finite):
        others = finite[:j] + finite[j + 1:]
        f = Fraction(1)
        for o in others:
            f *= a - o
        for k in range(r):
            G[j][k] = a ** k / f
        for i in range(m):
            A[j][i] = a ** i
        for d, c in enumerate(_poly_from_roots(others)):
            B[d][j] = c
    G[n - 1][r - 1] = Fraction(1)
    A[n - 1][m - 1] = Fraction(1)
    for d, c in enumerate(_poly_from_roots(finite)):
        B[d][n - 1] = c

    def arr(rows):
        return np.array([[float(v) for v in row] for row in rows], dtype=np.float64)

    return Transform(m, r, tuple(pts), arr(A).T, arr(G), arr(B).T)


def correlate1d(t: Transform, d: np.ndarray, g: np.ndarray) -> np.ndarray:
    return t.AT @ ((t.G @ g) * (t.BT @ d))


def tile2d(t: Transform, d: np.ndarray, g: np.ndarray, dtype=np.float64) -> np.ndarray:
    """One output tile; every product and sum is rounded to ``dtype``."""
    AT, G, BT = (x.astype(dtype) for x in (t.AT, t.G, t.BT))
    d = d.astype(dtype)
    g = g.astype(dtype)
    U = G @ g @ G.T
    V = BT @ d @ BT.T
    return AT @ (U * V) @ AT.T


def direct_tile(d: np.ndarray, g: np.ndarray, m: int) -> np.ndarray:
    r = g.shape[0]
    out = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            out[i, j] = (d[i:i + r, j:j + r] * g).sum()
    return out


def conv2d_winograd(x: np.ndarray, w: np.ndarray, t: Transform, dtype=np.float32) -> np.ndarray:
    """Multi-channel 3x3 'valid' correlation. x: (C, H, W), w: (K, C, r, r). Sum over C in the transform domain."""
    C, H, W = x.shape
    K = w.shape[0]
    m, r, n = t.m, t.r, t.n
    oh, ow = H - r + 1, W - r + 1
    th, tw = -(-oh // m), -(-ow // m)
    xp = np.zeros((C, th * m + r - 1, tw * m + r - 1), dtype=dtype)
    xp[:, :H, :W] = x
    AT, G, BT = (a.astype(dtype) for a in (t.AT, t.G, t.BT))
    U = np.einsum("ij,kcjl,ml->kcim", G, w.astype(dtype), G)                      # (K, C, n, n)
    tiles = np.lib.stride_tricks.sliding_window_view(xp, (n, n), axis=(1, 2))[:, ::m, ::m]  # (C, th, tw, n, n)
    V = np.einsum("ij,cpqjl,ml->cpqim", BT, tiles, BT)                             # (C, th, tw, n, n)
    M = np.einsum("kcim,cpqim->kpqim", U, V)                                        # (K, th, tw, n, n)
    Y = np.einsum("ij,kpqjl,ml->kpqim", AT, M, AT)                                  # (K, th, tw, m, m)
    y = Y.transpose(0, 1, 3, 2, 4).reshape(K, th * m, tw * m)
    return y[:, :oh, :ow]


def conv2d_direct(x: np.ndarray, w: np.ndarray, dtype=np.float64) -> np.ndarray:
    C, H, W = x.shape
    K, _, r, _ = w.shape
    oh, ow = H - r + 1, W - r + 1
    win = np.lib.stride_tricks.sliding_window_view(x.astype(dtype), (r, r), axis=(1, 2))  # (C, oh, ow, r, r)
    return np.einsum("cpqij,kcij->kpq", win, w.astype(dtype))[:, :oh, :ow]
