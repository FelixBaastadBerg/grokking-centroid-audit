"""
DIR C -- Representation Topology Divergence (Barannikov et al., ICML 2022), done properly.

Two earlier from-scratch attempts failed RTD's own defining test and were discarded: a
mapping-cylinder matrix gave RTD(A,A) = 1.235 (it computes H1 of the union, not of the pair),
and a coned distance matrix gave 0.000 for every input including circle-vs-disk (collapsing
a block puts every cross-edge at t=0, so the complex is contractible from the start).

The construction below is the textbook one and is correct by an argument, not by tuning.
R-Cross-Barcode_k(A, B) is the barcode of the PAIR (VR_w, VR_A) with w = min(A, B); since
w <= A we have VR_A subset of VR_w at every threshold, and

        H_*(X, A)  ~=  reduced H_*(X u cone(A))

so the pair is computed by building VR_w explicitly and coning off VR_A: simplex sigma
enters at diam_w(sigma), and sigma u {c} enters at diam_A(sigma). A 1-cycle that exists in
VR_w at scale w is then killed exactly when A fills it in.

Sanity by construction: if A == B then w == A, every simplex and its cone appear at the same
value, nothing persists, and RTD(A, A) = 0 identically. That is checked numerically in
tools/dirC_validate.py along with circle-vs-disk, before any of this touches real data.
"""
from __future__ import annotations
import numpy as np
import gudhi


def _diam(idx, D):
    if len(idx) < 2:
        return 0.0
    m = 0.0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            v = D[idx[i], idx[j]]
            if v > m:
                m = v
    return float(m)


def cross_barcode(A: np.ndarray, B: np.ndarray, maxdim: int = 1):
    """R-Cross-Barcode_k(A,B) for k = 0..maxdim, as lists of (birth, death)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    assert A.shape == B.shape and A.ndim == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    w = np.minimum(A, B)

    st = gudhi.RipsComplex(distance_matrix=w, max_edge_length=float(A.max()) * 1.001
                           ).create_simplex_tree(max_dimension=maxdim + 1)
    cone = n
    add = []
    for sigma, _ in st.get_simplices():
        if len(sigma) <= maxdim + 1:
            add.append((list(sigma) + [cone], _diam(sigma, A)))
    st.insert([cone], 0.0)
    for s, f in add:
        st.insert(s, f)
    st.compute_persistence(homology_coeff_field=2, persistence_dim_max=False)
    out = {}
    for k in range(maxdim + 1):
        iv = st.persistence_intervals_in_dimension(k)
        out[k] = np.asarray(iv, float).reshape(-1, 2)
    return out


def rtd(A: np.ndarray, B: np.ndarray, dim: int = 1, symmetrise: bool = True) -> float:
    """Total finite lifetime of R-Cross-Barcode_dim. Zero iff the pair is trivial."""
    tot, runs = 0.0, ((A, B), (B, A)) if symmetrise else ((A, B),)
    for X, Y in runs:
        d = cross_barcode(X, Y, maxdim=dim)[dim]
        if len(d):
            f = d[np.isfinite(d[:, 1])]
            tot += float((f[:, 1] - f[:, 0]).sum()) if len(f) else 0.0
    return tot / len(runs)


def dmat(X: np.ndarray, normalise: bool = True) -> np.ndarray:
    """Euclidean distances, scaled to unit mean so RTD is invariant to global rescaling."""
    X = np.asarray(X, float)
    d = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0.0))
    if normalise:
        m = d[np.triu_indices_from(d, 1)].mean()
        if m > 0:
            d = d / m
    return d


def subsample(n: int, k: int, seed: int = 0) -> np.ndarray:
    """RTD is O(n^3) in simplices; matched subsampling keeps the index pairing intact."""
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))
