"""
H-F: trajectory persistence -- Birdal et al. (2021) applied to the training path.

The optimization trajectory is a sequence of weight states; treated as a point
cloud in parameter space it has a persistent-homology dimension. We compute:

  * PH-dim via the alpha-weighted MST/E_alpha scaling law
        E^alpha_0(X_n) ~ n^((d-alpha)/d)   =>   d = alpha / (1 - slope)
    which at alpha=1 is exactly the MST length scaling. This is the quantity in
    Birdal et al. 2021 that bounds generalization.
  * the same quantity on SLIDING WINDOWS of the trajectory, giving PH-dim as a
    function of training time -- the object the charter asks for ("why/when/how").
"""
from __future__ import annotations
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


def _mst_edges_from_points(X: np.ndarray) -> np.ndarray:
    D = squareform(pdist(X))
    T = minimum_spanning_tree(D)
    return np.asarray(T.data, dtype=np.float64)


def ph_dim(X: np.ndarray, alpha: float = 1.0, n_list=None, n_repeats: int = 5,
           seed: int = 0) -> tuple[float, float]:
    """
    PH-dimension of a point cloud via E_alpha scaling. Returns (dim, r2_of_fit).
    Unlike the repo's original mst_dimension_with_fit this reports the fit quality
    so a meaningless extrapolation can be discarded rather than silently used.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if n_list is None:
        hi = N
        n_list = [n for n in (16, 24, 32, 48, 64, 96, 128, 192, 256) if n <= hi]
    if len(n_list) < 3:
        return np.nan, np.nan
    xs, ys = [], []
    for n in n_list:
        for _ in range(n_repeats):
            Y = X[rng.choice(N, size=n, replace=False)]
            e = _mst_edges_from_points(Y)
            e = e[np.isfinite(e) & (e > 0)]
            if e.size == 0:
                continue
            val = float((e ** alpha).sum())
            if val > 0:
                xs.append(np.log(n)); ys.append(np.log(val))
    if len(xs) < 6:
        return np.nan, np.nan
    xs, ys = np.array(xs), np.array(ys)
    slope, icept = np.polyfit(xs, ys, 1)
    yhat = slope * xs + icept
    ss_res = float(((ys - yhat) ** 2).sum()); ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = np.nan if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    if slope >= 1.0:
        return np.inf, r2
    return float(alpha / (1.0 - slope)), float(r2)


def trajectory_matrix(run, layers=("W0", "W1", "W2"), project_dim: int | None = None,
                      seed: int = 0) -> np.ndarray:
    """Flatten each checkpoint's weights into one vector -> (n_ckpt, n_params)."""
    parts = [run.traj[k].reshape(run.traj[k].shape[0], -1) for k in layers]
    M = np.concatenate(parts, axis=1).astype(np.float64)
    if project_dim is not None and project_dim < M.shape[1]:
        rng = np.random.default_rng(seed)
        R = rng.normal(size=(M.shape[1], project_dim)) / np.sqrt(project_dim)
        M = M @ R                      # Johnson-Lindenstrauss: distances preserved
    return M


def sliding_ph_dim(M: np.ndarray, epochs: np.ndarray, window: int = 40,
                   step: int = 5, alpha: float = 1.0, seed: int = 0) -> "np.ndarray":
    """PH-dim of the trajectory within a sliding window of checkpoints."""
    import pandas as pd
    rows = []
    for start in range(0, len(M) - window + 1, step):
        W = M[start:start + window]
        d, r2 = ph_dim(W, alpha=alpha, n_list=[n for n in (12, 16, 24, 32, 40) if n <= window],
                       n_repeats=6, seed=seed)
        # normalised speed and curvature of the path in the same window
        dif = np.diff(W, axis=0); speed = float(np.linalg.norm(dif, axis=1).mean())
        rows.append({"start_epoch": float(epochs[start]),
                     "mid_epoch": float(epochs[start + window // 2]),
                     "end_epoch": float(epochs[start + window - 1]),
                     "ph_dim": d, "ph_dim_r2": r2, "step_norm": speed})
    return pd.DataFrame(rows)
