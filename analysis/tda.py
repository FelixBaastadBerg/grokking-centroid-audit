"""
Persistent-homology toolkit for the topology-of-grokking program.

Design rules (learned the hard way in Phase 0):
  * NEVER row-normalise before computing PH. `normalize_point_cloud(mode="row_l2")`
    radially projects the cloud onto the unit sphere and removes one dimension --
    that was the bug behind the divergent `mst_dim`. Everything here works on RAW
    clouds, with an explicit, documented scaling choice instead.
  * Every summary produced here must pass the synthetic ground-truth check in
    `tools/he_validate_ph.py` before it is used for any claim.

Scale handling: PH lifetimes are scale-dependent, so comparing raw total
persistence across checkpoints partly measures how big the activations got. We
therefore report both raw summaries and summaries on a diameter-normalised cloud
(divide all distances by the max pairwise distance -- a global scalar, which
preserves shape and hence topology, unlike row-wise normalisation).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

try:
    from ripser import ripser
    HAVE_RIPSER = True
except Exception:  # pragma: no cover
    HAVE_RIPSER = False


# --------------------------------------------------------------------------
# cloud preparation
# --------------------------------------------------------------------------
def prepare_cloud(X: np.ndarray, max_points: int = 400, seed: int = 0,
                  dedup: bool = True, scale: str = "none") -> np.ndarray:
    """
    Subsample / clean a point cloud for PH.

    scale: "none"      -> raw coordinates
           "diameter"  -> divide by max pairwise distance (global scalar; shape and
                          therefore topology preserved, lifetimes made comparable)
           "std"       -> divide by mean coordinate std (global scalar)
    NOTE: deliberately NO row-wise normalisation option.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    X = X[np.isfinite(X).all(axis=1)]
    if X.shape[0] == 0:
        return X

    if dedup:
        # exact duplicates create zero distances -> log(0) blowups downstream
        X = np.unique(X, axis=0)

    if X.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(X.shape[0], size=max_points, replace=False)]

    if scale == "diameter" and X.shape[0] > 1:
        d = pdist(X)
        m = float(d.max()) if d.size else 0.0
        if m > 0:
            X = X / m
    elif scale == "std":
        s = float(X.std())
        if s > 0:
            X = X / s
    return X


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
def persistence_diagrams(X: np.ndarray, maxdim: int = 2, thresh: float | None = None,
                         coeff: int = 2) -> list[np.ndarray]:
    """Vietoris-Rips diagrams H0..H_maxdim. Returns list of (n_features, 2) arrays."""
    if not HAVE_RIPSER:
        raise RuntimeError("ripser not available")
    if X.shape[0] < 4:
        return [np.empty((0, 2)) for _ in range(maxdim + 1)]
    kw = dict(maxdim=maxdim, coeff=coeff)
    if thresh is not None:
        kw["thresh"] = thresh
    dgms = ripser(X, **kw)["dgms"]
    while len(dgms) < maxdim + 1:
        dgms.append(np.empty((0, 2)))
    return dgms


def _finite(dgm: np.ndarray) -> np.ndarray:
    if dgm.size == 0:
        return np.empty((0, 2))
    return dgm[np.isfinite(dgm).all(axis=1)]


def diagram_summaries(dgm: np.ndarray, prefix: str) -> dict:
    """
    Scalar summaries of one persistence diagram.
    All are standard and all are validated against synthetic shapes.
    """
    d = _finite(dgm)
    if d.shape[0] == 0:
        z = {f"{prefix}_n_above_{t:.2f}": 0.0 for t in (0.10, 0.15, 0.20, 0.25)}
        return {**z, f"{prefix}_n": 0.0, f"{prefix}_total_pers": 0.0, f"{prefix}_max_pers": 0.0,
                f"{prefix}_pers_entropy": np.nan, f"{prefix}_mean_pers": np.nan,
                f"{prefix}_n_significant": 0.0, f"{prefix}_gap_ratio": np.nan}

    life = d[:, 1] - d[:, 0]
    life = life[np.isfinite(life) & (life > 0)]
    if life.size == 0:
        z = {f"{prefix}_n_above_{t:.2f}": 0.0 for t in (0.10, 0.15, 0.20, 0.25)}
        return {**z, f"{prefix}_n": 0.0, f"{prefix}_total_pers": 0.0, f"{prefix}_max_pers": 0.0,
                f"{prefix}_pers_entropy": np.nan, f"{prefix}_mean_pers": np.nan,
                f"{prefix}_n_significant": 0.0, f"{prefix}_gap_ratio": np.nan}

    total = float(life.sum())
    p = life / total
    entropy = float(-(p * np.log(p + 1e-300)).sum())
    srt = np.sort(life)[::-1]
    # "significant" = lifetime above half the longest; a crude but stable Betti proxy
    n_sig = float((life >= 0.5 * srt[0]).sum())
    # ratio of longest to second-longest lifetime: large => one dominant feature
    gap = float(srt[0] / srt[1]) if srt.size > 1 and srt[1] > 0 else np.inf

    # Absolute-threshold Betti counts. The relative "half the longest" heuristic
    # (kept below as _n_significant) FAILS validation -- on a plane it reports ~8
    # spurious H1 features -- so these calibrated counts are what we actually use.
    counts = {f"{prefix}_n_above_{t:.2f}": float((life >= t).sum())
              for t in (0.10, 0.15, 0.20, 0.25)}

    return {
        **counts,
        f"{prefix}_n": float(life.size),
        f"{prefix}_total_pers": total,
        f"{prefix}_max_pers": float(srt[0]),
        f"{prefix}_mean_pers": float(life.mean()),
        f"{prefix}_pers_entropy": entropy,
        f"{prefix}_n_significant": n_sig,
        f"{prefix}_gap_ratio": gap,
    }


def betti_curve(dgm: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Betti number as a function of filtration value (for CROCKER plots)."""
    d = _finite(dgm)
    if d.shape[0] == 0:
        return np.zeros_like(grid)
    births, deaths = d[:, 0][None, :], d[:, 1][None, :]
    g = grid[:, None]
    return ((births <= g) & (g < deaths)).sum(axis=1).astype(float)


def persistence_landscape(dgm: np.ndarray, grid: np.ndarray, n_layers: int = 3) -> np.ndarray:
    """Persistence landscape, first `n_layers` layers, sampled on `grid`."""
    d = _finite(dgm)
    out = np.zeros((n_layers, grid.size))
    if d.shape[0] == 0:
        return out
    b, dth = d[:, 0][None, :], d[:, 1][None, :]
    g = grid[:, None]
    tent = np.maximum(np.minimum(g - b, dth - g), 0.0)   # (n_grid, n_features)
    tent = -np.sort(-tent, axis=1)
    for k in range(min(n_layers, tent.shape[1])):
        out[k] = tent[:, k]
    return out


def cloud_persistence_features(X: np.ndarray, maxdim: int = 2, max_points: int = 400,
                               seed: int = 0, scale: str = "diameter") -> dict:
    """One-stop: prepare cloud -> diagrams -> summaries for H0/H1/H2."""
    Xp = prepare_cloud(X, max_points=max_points, seed=seed, scale=scale)
    if Xp.shape[0] < 10:
        return {"n_points": float(Xp.shape[0])}
    dgms = persistence_diagrams(Xp, maxdim=maxdim)
    out = {"n_points": float(Xp.shape[0])}
    for k, dgm in enumerate(dgms):
        out.update(diagram_summaries(dgm, f"H{k}"))
    return out
