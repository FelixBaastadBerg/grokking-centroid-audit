"""
DIR A -- topology of the LOSS LANDSCAPE, not of a representation.

This dodges the |G| ceiling that killed P1-P3 completely: the objects here are points in
weight space and levels of a scalar function, neither of which is capped by the group order.

Two quantities, both with exact known-answer tests in tools/dirA_validate.py:

  minimax_barrier(F, a, b)
      the lowest ceiling c such that a and b lie in the same connected component of
      {F <= c}. On a double well this is EXACTLY the saddle height, which is how we test it.
      Computed by union-find over pixels in increasing value order -- the standard
      bottleneck-shortest-path construction, no approximation.

  betti_sublevel(F, levels)
      b0 and b1 of the sublevel set {F <= t} on a 2D grid. b0 by labelling; b1 by planar
      Alexander duality -- a hole is a component of the complement that does not touch the
      grid boundary. Exact for cubical grids.

The "is the generalizing basin reachable from the memorizing one" question is then a barrier
height, and "did a barrier collapse" is that height as a function of training time.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------- connectivity
class _UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb
        return ra != rb


def minimax_barrier(F: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float:
    """
    Lowest c with a, b in one component of {F <= c}. On a landscape with a single saddle
    between two wells this is the saddle height exactly.
    """
    H, W = F.shape
    order = np.argsort(F, axis=None, kind="stable")
    uf = _UF(H * W)
    active = np.zeros(H * W, bool)
    ia, ib = a[0] * W + a[1], b[0] * W + b[1]
    for flat in order:
        r, c = divmod(int(flat), W)
        active[flat] = True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and active[rr * W + cc]:
                uf.union(flat, rr * W + cc)
        if active[ia] and active[ib] and uf.find(ia) == uf.find(ib):
            return float(F.flat[flat])
    return float("inf")


def betti_sublevel(F: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """b0, b1 of {F <= t} for each t. b1 by planar duality: bounded complement components."""
    b0 = np.zeros(len(levels), int)
    b1 = np.zeros(len(levels), int)
    for i, t in enumerate(levels):
        S = F <= t
        if not S.any():
            continue
        lab, n = ndimage.label(S)
        b0[i] = n
        comp = ~S
        if comp.any():
            lc, nc = ndimage.label(comp)
            touching = set(lc[0, :]) | set(lc[-1, :]) | set(lc[:, 0]) | set(lc[:, -1])
            touching.discard(0)
            b1[i] = nc - len(touching)
    return b0, b1


def barrier_1d(vals: np.ndarray) -> float:
    """Linear-interpolation barrier: peak of the path relative to its higher endpoint."""
    return float(vals.max() - max(vals[0], vals[-1]))


# ---------------------------------------------------------------- weight-space planes
def plane_basis(wa: np.ndarray, wb: np.ndarray, wc: np.ndarray | None = None, seed: int = 0):
    """
    Orthonormal 2D basis for a plane containing wa and wb (and wc if given, else a random
    third direction). Returns (origin, u, v) with wa, wb expressed in that frame.
    """
    d1 = wb - wa
    n1 = np.linalg.norm(d1)
    u = d1 / (n1 + 1e-12)
    if wc is not None:
        d2 = wc - wa
    else:
        d2 = np.random.default_rng(seed).normal(size=wa.shape).astype(wa.dtype) * (n1 / np.sqrt(wa.size))
    d2 = d2 - (d2 @ u) * u
    v = d2 / (np.linalg.norm(d2) + 1e-12)
    return u, v


def grid_coords(wa, wb, u, v, n=41, pad=0.6):
    """Grid over the plane spanned by u,v covering wa and wb with `pad` margin."""
    ca = np.array([0.0, 0.0])
    cb = np.array([(wb - wa) @ u, (wb - wa) @ v])
    span = max(np.abs(cb).max(), 1e-9)
    lo, hi = -pad * span, (1 + pad) * span
    ax = np.linspace(lo, hi, n)
    ay = np.linspace(-(0.5 + pad) * span, (0.5 + pad) * span, n)
    return ax, ay, ca, cb


def nearest_idx(ax, ay, c):
    return int(np.argmin(np.abs(ay - c[1]))), int(np.argmin(np.abs(ax - c[0])))
