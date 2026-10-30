"""
Cohomological circular coordinates (de Silva, Morozov, Vejdemo-Johansson 2011).

Why this, for this project: our ambient persistent homology reported b_1 = 0 on every
trained embedding, and the reason is that a modular-arithmetic representation is a
SUPERPOSITION of Fourier modes -- a curve winding a high-dimensional torus. There is
no single short-scale 1-cycle for Rips to find, so vanilla PH returns a confident
zero. Circular coordinates attack the same structure from cohomology instead: they
construct an explicit map

        theta : X -> R/Z

from a persistent H^1 cocycle. If the representation really does wind, theta recovers
the winding directly -- turning our own false negative into a positive, and giving a
topological quantity that is comparable with `equiv_R2` without assuming the group.

Algorithm:
  1. persistent cohomology over Z/q (q prime), pick the longest-lived H^1 class
  2. choose a scale eps inside that class's (birth, death) interval
  3. build the Rips 1-skeleton at eps; lift the Z/q cocycle to an integer cocycle
     (representatives in (-q/2, q/2])
  4. least-squares  min_f || d0 f - alpha ||_2   over vertex functions f
  5. theta = f mod 1
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial.distance import pdist, squareform

try:
    from ripser import ripser
    HAVE_RIPSER = True
except Exception:                                          # pragma: no cover
    HAVE_RIPSER = False


def circular_coordinate(X: np.ndarray, coeff: int = 17, rank: int = 0,
                        eps_frac: float = 0.5, thresh: float | None = None):
    """
    Returns (theta, info). theta is in [0,1) per point; info carries the chosen bar,
    the scale used, and the residual of the least-squares lift.

    rank: 0 = most persistent H^1 class, 1 = second, ... (a torus has >= 2).
    eps_frac: where in (birth, death) to build the complex, as a fraction.
    """
    if not HAVE_RIPSER:
        raise RuntimeError("ripser not available")
    n = X.shape[0]
    D = squareform(pdist(X))
    kw = dict(maxdim=1, coeff=coeff, do_cocycles=True, distance_matrix=True)
    if thresh is not None:
        kw["thresh"] = thresh
    res = ripser(D, **kw)
    dgm1 = res["dgms"][1]
    if dgm1.shape[0] == 0:
        return None, {"reason": "no H1 classes"}

    life = dgm1[:, 1] - dgm1[:, 0]
    order = np.argsort(-life)
    if rank >= len(order):
        return None, {"reason": f"only {len(order)} H1 classes"}
    k = int(order[rank])
    birth, death = float(dgm1[k, 0]), float(dgm1[k, 1])
    if not np.isfinite(death):
        death = float(D.max())
    eps = birth + eps_frac * (death - birth)

    cocycle = np.asarray(res["cocycles"][1][k], dtype=np.int64)
    if cocycle.size == 0:
        return None, {"reason": "empty cocycle"}

    # lift Z/q -> Z with representatives in (-q/2, q/2]
    val = cocycle[:, 2] % coeff
    val = np.where(val > coeff // 2, val - coeff, val)

    # Rips 1-skeleton at eps
    iu = np.triu_indices(n, 1)
    mask = D[iu] <= eps
    ei, ej = iu[0][mask], iu[1][mask]
    if len(ei) < n:
        return None, {"reason": "graph too sparse at eps", "eps": eps}
    edge_id = {}
    for e, (a, b) in enumerate(zip(ei, ej)):
        edge_id[(int(a), int(b))] = e
    m = len(ei)

    # target cocycle vector on those edges
    alpha = np.zeros(m)
    for a, b, v in zip(cocycle[:, 0], cocycle[:, 1], val):
        a, b = int(a), int(b)
        s = 1.0
        if a > b:
            a, b = b, a; s = -1.0
        e = edge_id.get((a, b))
        if e is not None:
            alpha[e] = s * v

    # coboundary d0: (d0 f)(a,b) = f(b) - f(a)
    rows = np.repeat(np.arange(m), 2)
    cols = np.empty(2 * m, dtype=np.int64)
    cols[0::2] = ei; cols[1::2] = ej
    data = np.empty(2 * m); data[0::2] = -1.0; data[1::2] = 1.0
    d0 = coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()

    sol = lsqr(d0, alpha, atol=1e-10, btol=1e-10, iter_lim=5000)
    f = sol[0]
    resid = float(np.linalg.norm(d0 @ f - alpha) / (np.linalg.norm(alpha) + 1e-12))
    theta = np.mod(f, 1.0)
    return theta, {"birth": birth, "death": death, "lifetime": death - birth,
                   "eps": eps, "n_edges": m, "rel_residual": resid, "rank": rank}


# --------------------------------------------------------------------------
def winding_fit(theta: np.ndarray, u: np.ndarray, max_w: int | None = None):
    """
    How many times does theta wind as the underlying parameter u goes once round?

    Fits theta ~ w*u + phi (mod 1) over integer w, returning the best w and the
    circular residual. u should be in [0,1). A clean winding gives resid near 0;
    incoherent coordinates give resid near the 0.289 of a uniform random phase.
    """
    # NYQUIST: for u = n/p with integer n, windings w and w+p are indistinguishable.
    # Without this cap the fit aliases onto |w| >= p and can "explain" pure noise.
    n_pts = len(u)
    cap = (n_pts - 1) // 2 if max_w is None else min(max_w, (n_pts - 1) // 2)
    best = (None, np.inf, None)
    for w in range(-cap, cap + 1):
        if w == 0:
            continue
        d = np.mod(theta - w * u, 1.0)
        # circular mean/spread of the residual phase
        ang = 2 * np.pi * d
        R = np.abs(np.mean(np.exp(1j * ang)))
        spread = np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12)))) / (2 * np.pi)
        if spread < best[1]:
            best = (w, float(spread), float(np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi)))
    return {"winding": best[0], "circ_resid": best[1], "phase": best[2]}
