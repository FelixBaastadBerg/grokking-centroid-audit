"""
A GROUP-AGNOSTIC structural order parameter.

Motivation: the per-frequency circularity statistic only exists for cyclic groups,
because it uses the DFT basis of Z_p. That left the S_4 (non-abelian) arm with no
comparable measurement -- so the cyclic-vs-non-abelian contrast could not actually
be made. This module fixes that.

The idea: a representation that respects the group structure has a Gram matrix that
is a *function of the group difference*,

        K[g, h] = <emb(g), emb(h)> = f(g^-1 h)

For Z_p this says exactly "K is circulant", which is equivalent to the embedding
being diagonalised by the Fourier basis -- so this is a strict generalisation of the
Fourier test, not a different one. But it is computable for ANY finite group with no
irrep machinery, which makes Z_23 and S_4 directly comparable.

Order parameter: the fraction of Gram-matrix variance explained by g^-1 h,

        equivariance_R2 = 1 - E_d[ Var(K[g,h] | g^-1 h = d) ] / Var(K)

1.0 = perfectly group-structured; 0 = no group structure. Reported against a
row-permutation null, exactly as the per-frequency statistic is.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# group tables
# --------------------------------------------------------------------------
def cyclic_diff_table(p: int) -> np.ndarray:
    """d[g,h] = (h - g) mod p, the 'difference' in Z_p."""
    idx = np.arange(p)
    return (idx[None, :] - idx[:, None]) % p


def symmetric_group_elements(n: int) -> list[tuple]:
    from itertools import permutations
    return sorted(permutations(range(n)))


def symmetric_diff_table(n: int) -> np.ndarray:
    """d[g,h] = index of g^-1 h in S_n."""
    P = symmetric_group_elements(n)
    idx = {q: i for i, q in enumerate(P)}
    m = len(P)
    inv = []
    for g in P:
        gi = [0] * n
        for i, gi_ in enumerate(g):
            gi[gi_] = i
        inv.append(tuple(gi))
    D = np.zeros((m, m), dtype=np.int64)
    for a, g in enumerate(P):
        ginv = inv[a]
        for b, h in enumerate(P):
            D[a, b] = idx[tuple(ginv[h[k]] for k in range(n))]
    return D


def diff_table_for(task: str, p: int) -> np.ndarray | None:
    if task in ("add", "sub", "mul", "div"):
        return cyclic_diff_table(p)
    if task == "perm":
        n = {6: 3, 24: 4, 120: 5}.get(p)
        return symmetric_diff_table(n) if n else None
    return None


# --------------------------------------------------------------------------
def equivariance_r2(E: np.ndarray, D: np.ndarray, normalise: bool = True) -> float:
    """
    Fraction of Gram-matrix variance explained by the group difference.
    E: (|G|, d) embeddings indexed by group element. D: (|G|,|G|) difference table.
    """
    Ec = E - E.mean(0, keepdims=True)
    if normalise:
        nrm = np.linalg.norm(Ec, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Ec = Ec / nrm                      # correlation-like: removes norm variation
    K = Ec @ Ec.T
    iu = ~np.eye(K.shape[0], dtype=bool)   # ignore the diagonal (trivially constant)
    k = K[iu].ravel()
    d = D[iu].ravel()
    tot = k.var()
    if tot <= 1e-15:
        return np.nan
    within = 0.0
    for val in np.unique(d):
        m = d == val
        if m.sum() > 1:
            within += m.sum() * k[m].var()
    within /= len(k)
    return float(1.0 - within / tot)


def equivariance_with_null(E: np.ndarray, D: np.ndarray, n_null: int = 16,
                           seed: int = 0) -> dict:
    """Real value plus a row-permutation null (destroys g -> emb(g), keeps the cloud)."""
    real = equivariance_r2(E, D)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_null):
        nulls.append(equivariance_r2(E[rng.permutation(E.shape[0])], D))
    nulls = np.array([x for x in nulls if np.isfinite(x)], dtype=float)
    if nulls.size == 0:
        return {"equiv_r2": real, "equiv_r2_null_mean": np.nan,
                "equiv_r2_null_sd": np.nan, "equiv_r2_z": np.nan}
    return {
        "equiv_r2": real,
        "equiv_r2_null_mean": float(nulls.mean()),
        "equiv_r2_null_sd": float(nulls.std()),
        "equiv_r2_z": float((real - nulls.mean()) / nulls.std()) if nulls.std() > 0 else np.nan,
    }


# --------------------------------------------------------------------------
# LEFT vs RIGHT vs CONJUGACY -- the non-abelian blind spot
# --------------------------------------------------------------------------
"""
equivariance_r2 above tests K[g,h] = f(g^-1 h), i.e. invariance under the LEFT
action. For an ABELIAN group g^-1 h == h g^-1, so left and right coincide and the
single test is complete. For a NON-ABELIAN group they are different structures, and
a representation that is right-equivariant would be invisible to the left test.

That means the S_4 'low equiv_R2' result could be an artifact of testing only one
side. These functions close that gap:

  left   K[g,h] = f(g^-1 h)
  right  K[g,h] = f(h g^-1)
  conj   K[g,h] = f(conjugacy class of g^-1 h)   -- weaker, class-function structure
"""


def right_diff_table(task: str, p: int) -> np.ndarray | None:
    """d[g,h] = index of h g^-1."""
    if task in ("add", "sub", "mul", "div"):
        idx = np.arange(p)
        return (idx[None, :] - idx[:, None]) % p          # abelian: same as left
    if task == "perm":
        n = {6: 3, 24: 4, 120: 5}.get(p)
        if n is None:
            return None
        P = symmetric_group_elements(n)
        index = {q: i for i, q in enumerate(P)}
        inv = []
        for g in P:
            gi = [0] * n
            for i, gi_ in enumerate(g):
                gi[gi_] = i
            inv.append(tuple(gi))
        D = np.zeros((len(P), len(P)), dtype=np.int64)
        for a, g in enumerate(P):
            ginv = inv[a]
            for b, h in enumerate(P):
                D[a, b] = index[tuple(h[ginv[k]] for k in range(n))]
        return D
    return None


def conjugacy_table(task: str, p: int) -> np.ndarray | None:
    """d[g,h] = conjugacy-class id of g^-1 h (cycle type for S_n)."""
    if task in ("add", "sub", "mul", "div"):
        return cyclic_diff_table(p)                        # abelian: classes are singletons
    if task == "perm":
        n = {6: 3, 24: 4, 120: 5}.get(p)
        if n is None:
            return None
        P = symmetric_group_elements(n)
        L = cyclic_type_ids(P, n)
        D = symmetric_diff_table(n)
        return L[D]
    return None


def cyclic_type_ids(P, n: int) -> np.ndarray:
    """Map each permutation to an integer id for its cycle type."""
    types = {}
    out = np.zeros(len(P), dtype=np.int64)
    for i, g in enumerate(P):
        seen = [False] * n
        cyc = []
        for s in range(n):
            if seen[s]:
                continue
            ln = 0; j = s
            while not seen[j]:
                seen[j] = True; j = g[j]; ln += 1
            cyc.append(ln)
        key = tuple(sorted(cyc))
        if key not in types:
            types[key] = len(types)
        out[i] = types[key]
    return out


# --------------------------------------------------------------------------
# THE LOCKED ORDER PARAMETER
# --------------------------------------------------------------------------
def equivariance_order_parameter(E: np.ndarray, task: str, p: int,
                                 n_null: int = 16, seed: int = 0) -> dict:
    """
    THE order parameter for this project. Frozen -- no further statistic search.

    equiv_R2 := max( R2_left, R2_right )

    The max is not a fudge: for an ABELIAN group the two tests are identical by
    construction, so the max changes nothing there. For a NON-ABELIAN group they are
    genuinely different structures, and which side a network uses is determined by
    the task's action, not by whether the network learned anything. Testing only the
    left side reported S_4 as structureless (0.115 at test acc 0.912) when it is in
    fact strongly right-equivariant (0.877). A one-sided statistic silently returns a
    confident zero for the other side.

    Both sides are always reported alongside the max so the choice is auditable.
    Null: row permutation of the embedding, per run, on the same side.
    """
    Dl = diff_table_for(task, p)
    Dr = right_diff_table(task, p)
    if Dl is None:
        return {}
    left = equivariance_r2(E, Dl)
    right = equivariance_r2(E, Dr) if Dr is not None else np.nan
    side = "left" if (not np.isfinite(right) or left >= right) else "right"
    D = Dl if side == "left" else Dr

    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_null):
        Ep = E[rng.permutation(E.shape[0])]
        nl = equivariance_r2(Ep, Dl)
        nr = equivariance_r2(Ep, Dr) if Dr is not None else np.nan
        nulls.append(np.nanmax([nl, nr]))          # null takes the max too, or it is biased
    nulls = np.array([x for x in nulls if np.isfinite(x)], float)
    val = float(np.nanmax([left, right]))
    out = {"equiv_R2": val, "equiv_R2_left": float(left),
           "equiv_R2_right": float(right) if np.isfinite(right) else np.nan,
           "equiv_side": side}
    if nulls.size:
        out.update({"equiv_R2_null_mean": float(nulls.mean()),
                    "equiv_R2_null_sd": float(nulls.std()),
                    "equiv_R2_z": float((val - nulls.mean()) / nulls.std()) if nulls.std() > 0 else np.nan})
    return out
