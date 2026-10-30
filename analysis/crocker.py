"""
CROCKER plots (Topaz, Ziegelmeier, Halverson) of the TRAINING TRAJECTORY.

The wedge: Birdal/Andreeva measure the trajectory but collapse it to one scalar per
run; Monod resolves time but measures representations. Nobody has produced a
time-resolved topological summary of the weight-space path.

We established (WEDGE.md) that our full-batch trajectory is a *smooth curve*, so its
PH-dimension is pinned at ~1 and carries no event. But a smooth curve still has
SHAPE, and the shape question is the interesting one:

    does the optimizer's path close a loop, or fold back on itself, at grokking?

A sliding window of a smooth arc is contractible: beta_0 = 1, beta_1 = 0. beta_1
becomes nonzero exactly when the path *returns near a region it already visited* --
a loop closure. That is a genuine, localizable topological event, and it is precisely
the kind of thing CROCKER was built to display.

Scale handling: distances are normalised ONCE, globally, by the median consecutive
step length of the whole trajectory. This makes beta(t, eps) comparable across t --
per-window normalisation would destroy exactly the comparison we want.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

try:
    from ripser import ripser
    HAVE_RIPSER = True
except Exception:                                        # pragma: no cover
    HAVE_RIPSER = False


def betti_curve_from_dgm(dgm: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if dgm is None or len(dgm) == 0:
        return np.zeros_like(grid)
    d = dgm[np.isfinite(dgm).all(axis=1)] if dgm.size else dgm
    if len(d) == 0:
        return np.zeros_like(grid)
    b, dd = d[:, 0][None, :], d[:, 1][None, :]
    g = grid[:, None]
    return ((b <= g) & (g < dd)).sum(axis=1).astype(float)


def normalise_trajectory(M: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale so that the median consecutive step length is 1."""
    step = np.linalg.norm(np.diff(M, axis=0), axis=1)
    s = float(np.median(step[step > 0])) if np.any(step > 0) else 1.0
    return M / s, s


def crocker_trajectory(M: np.ndarray, epochs: np.ndarray, window: int = 40,
                       stride: int = 4, n_scales: int = 48,
                       max_scale: float | None = None, maxdim: int = 1):
    """
    Sliding-window CROCKER of a trajectory.

    Returns (mid_epochs, scale_grid, B0, B1) where B_k[t, s] = beta_k of the window
    centred at time t, at filtration scale s. Scales are in units of the median
    consecutive step, so they are interpretable: eps = 5 means "five typical steps".
    """
    if not HAVE_RIPSER:
        raise RuntimeError("ripser not available")
    Mn, _ = normalise_trajectory(M)
    # TWO PREREQUISITES, both learned the hard way:
    #  * the window must exceed the feature's TEMPORAL extent, or a loop looks like
    #    an arc and beta_1 is a confident zero;
    #  * max_scale must exceed its SPATIAL diameter, or the filtration is truncated
    #    before the cycle can close.
    # So max_scale defaults to the largest within-window diameter actually present.
    if max_scale is None:
        diam = 0.0
        for st in range(0, len(Mn) - window + 1, max(stride, 1)):
            W = Mn[st:st + window]
            diam = max(diam, float(pdist(W).max()) if len(W) > 1 else 0.0)
        max_scale = 1.05 * diam if diam > 0 else 1.0
    grid = np.linspace(0.0, max_scale, n_scales)
    mids, B0, B1 = [], [], []
    for start in range(0, len(Mn) - window + 1, stride):
        W = Mn[start:start + window]
        D = squareform(pdist(W))
        res = ripser(D, maxdim=maxdim, distance_matrix=True, thresh=max_scale)
        B0.append(betti_curve_from_dgm(res["dgms"][0], grid))
        B1.append(betti_curve_from_dgm(res["dgms"][1], grid) if maxdim >= 1 else np.zeros_like(grid))
        mids.append(float(epochs[start + window // 2]))
    return np.array(mids), grid, np.array(B0), np.array(B1)


def loop_summary(mids, grid, B1):
    """
    Per-window scalar summaries of loop content:
      loop_area   integral of beta_1 over scale (total loop content)
      loop_max    max beta_1 at any scale
      loop_scale  scale at which beta_1 peaks (how far the path reaches back)
    """
    dg = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0
    area = B1.sum(axis=1) * dg
    mx = B1.max(axis=1)
    arg = np.array([grid[int(np.argmax(row))] if row.max() > 0 else np.nan for row in B1])
    return {"mid_epoch": mids, "loop_area": area, "loop_max": mx, "loop_scale": arg}
