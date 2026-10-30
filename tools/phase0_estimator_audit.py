"""
PHASE 0 -- Verify the instruments.

Audits every intrinsic-dimension estimator in logger.py against synthetic point
clouds of KNOWN intrinsic dimension, embedded in ambient dim 200 (matching the
real MLP hidden width).

Critically, this audits the estimators AS THE LOGGER ACTUALLY CALLS THEM, i.e.
with normalize_point_cloud(mode="row_l2") applied first -- and also without it,
because if row_l2 normalisation is what destroys dimension recovery, that is
itself the finding.

Outputs:
    results/phase0_estimator_audit.csv     raw per-(shape,estimator,M,seed) values
    results/phase0_verdicts.csv            aggregated KEEP/FIX/DROP table
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

warnings.filterwarnings("ignore")

from logger import (  # noqa: E402
    correlation_dimension,
    mst_dimension_with_fit,
    twonn_intrinsic_dimension,
    participation_ratio,
    mst_edges,
    ph0_summaries_from_mst_edges,
    e_alpha_from_mst_edges,
    magnitude_from_point_cloud,
    normalize_point_cloud,
    weight_effective_rank,
    weight_spectral_entropy,
)

AMBIENT = 200
N_POINTS = 2048
SEEDS = [0, 1, 2]
M_LIST = [500, 1000, 2000]


# --------------------------------------------------------------------------
# synthetic manifolds of known intrinsic dimension
# --------------------------------------------------------------------------
def _embed(X: np.ndarray, ambient: int, rng: np.random.Generator) -> np.ndarray:
    """Pad to `ambient` dims and apply a random rotation (isometry: preserves ID)."""
    n, d = X.shape
    if d > ambient:
        raise ValueError("intrinsic dim exceeds ambient")
    Z = np.zeros((n, ambient), dtype=np.float64)
    Z[:, :d] = X
    Q, _ = np.linalg.qr(rng.normal(size=(ambient, ambient)))
    return Z @ Q


def make_cloud(shape: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Returns (points, true_intrinsic_dimension)."""
    if shape.startswith("cube"):
        d = int(shape.split("_")[1])
        X = rng.uniform(0.0, 1.0, size=(n, d))
        return _embed(X, AMBIENT, rng), float(d)

    if shape.startswith("sphere"):
        # surface of S^d lives in R^(d+1) and has intrinsic dimension d
        d = int(shape.split("_")[1])
        V = rng.normal(size=(n, d + 1))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        return _embed(V, AMBIENT, rng), float(d)

    if shape == "helix":
        t = rng.uniform(0, 8 * np.pi, size=n)
        X = np.column_stack([np.cos(t), np.sin(t), 0.15 * t])
        return _embed(X, AMBIENT, rng), 1.0

    if shape == "swissroll":
        t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=n))
        h = 21.0 * rng.uniform(size=n)
        X = np.column_stack([t * np.cos(t), h, t * np.sin(t)]) / 10.0
        return _embed(X, AMBIENT, rng), 2.0

    if shape == "plane":
        X = rng.uniform(0.0, 1.0, size=(n, 2))
        return _embed(X, AMBIENT, rng), 2.0

    raise ValueError(shape)


SHAPES = [
    ("cube_1", 1.0), ("cube_2", 2.0), ("cube_3", 3.0),
    ("cube_5", 5.0), ("cube_8", 8.0), ("cube_10", 10.0),
    ("helix", 1.0), ("swissroll", 2.0), ("plane", 2.0),
    ("sphere_2", 2.0), ("sphere_5", 5.0),
]


# --------------------------------------------------------------------------
# estimator adapters -- each returns a dict of named scalar outputs
# --------------------------------------------------------------------------
def run_estimators(X: np.ndarray) -> dict:
    out: dict[str, float] = {}

    mst_dim, mst_alpha, mst_r2 = mst_dimension_with_fit(
        X, n_list=tuple(n for n in (64, 128, 256, 512) if n <= X.shape[0]), n_repeats=3
    )
    out["mst_dim"] = mst_dim
    out["mst_alpha"] = mst_alpha
    out["mst_r2"] = mst_r2
    # the dimension implied by alpha, computed defensively (the claimed quantity)
    out["mst_dim_from_alpha"] = (
        1.0 / (1.0 - mst_alpha) if np.isfinite(mst_alpha) and mst_alpha < 1.0 else np.inf
    )

    cd, cr2 = correlation_dimension(X, max_points=min(512, X.shape[0]))
    out["corr_dim"] = cd
    out["corr_r2"] = cr2

    out["twonn_id"] = twonn_intrinsic_dimension(X[: min(1024, X.shape[0])])
    out["pca_participation_ratio"] = participation_ratio(X)

    sub = X[: min(512, X.shape[0])]
    edges = mst_edges(sub)
    ph0 = ph0_summaries_from_mst_edges(edges)
    out["ph0_total_persistence"] = ph0["ph0_total_persistence"]
    out["ph0_persistence_entropy"] = ph0["ph0_persistence_entropy"]
    for a in (0.5, 1.0, 2.0):
        out[f"E_alpha_{a:g}"] = e_alpha_from_mst_edges(edges, a)

    mag, pmag = magnitude_from_point_cloud(X, max_points=min(256, X.shape[0]))
    out["mag_euclidean_s1"] = mag
    out["pmag_euclidean_s1"] = pmag

    return out


# --------------------------------------------------------------------------
# PH0 dimension via the E_alpha scaling law, and MST-vs-ripser consistency
# --------------------------------------------------------------------------
def ph0_dimension_via_scaling(X: np.ndarray, alpha: float = 1.0,
                              n_list=(64, 128, 256, 512), n_repeats: int = 3,
                              seed: int = 0) -> float:
    """
    Birdal et al.: E^alpha_0(X_n) ~ n^((d-alpha)/d)  =>  d = alpha / (1 - slope).
    At alpha=1 this is algebraically identical to the MST length scaling, so it is
    the natural cross-check on the MST implementation.
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    N = X.shape[0]
    for n in n_list:
        if n > N:
            continue
        for _ in range(n_repeats):
            Y = X[rng.choice(N, size=n, replace=False)]
            e = mst_edges(Y)
            val = e_alpha_from_mst_edges(e, alpha)
            if np.isfinite(val) and val > 0:
                xs.append(np.log(n))
                ys.append(np.log(val))
    if len(xs) < 6:
        return float("nan")
    slope, _ = np.polyfit(np.array(xs), np.array(ys), 1)
    if slope >= 1.0:
        return float("inf")
    return float(alpha / (1.0 - slope))


def mst_vs_ripser_consistency(X: np.ndarray) -> dict:
    """PH0 finite death times must equal MST edge lengths exactly."""
    try:
        from ripser import ripser
    except Exception as exc:  # pragma: no cover
        return {"ph0_available": False, "note": str(exc)}

    sub = X[: min(300, X.shape[0])]
    dgm = ripser(sub, maxdim=0)["dgms"][0]
    deaths = np.sort(dgm[np.isfinite(dgm[:, 1]), 1])
    edges = np.sort(mst_edges(sub))
    if len(deaths) != len(edges):
        return {"ph0_available": True, "n_match": False,
                "n_deaths": int(len(deaths)), "n_edges": int(len(edges))}
    return {
        "ph0_available": True,
        "n_match": True,
        "max_abs_diff": float(np.max(np.abs(deaths - edges))),
        "n_edges": int(len(edges)),
    }


# --------------------------------------------------------------------------
def main() -> int:
    rows = []
    for shape, true_d in SHAPES:
        for seed in SEEDS:
            rng = np.random.default_rng(1000 * seed + hash(shape) % 997)
            X_full, td = make_cloud(shape, N_POINTS, rng)
            assert td == true_d
            for M in M_LIST:
                if M > X_full.shape[0]:
                    continue
                X = X_full[:M]
                for norm_mode in ("none", "row_l2"):
                    Xn = normalize_point_cloud(X, mode=norm_mode)
                    vals = run_estimators(Xn)
                    vals["ph0_dim_scaling"] = ph0_dimension_via_scaling(Xn, seed=seed)
                    rows.append({
                        "shape": shape, "true_d": true_d, "M": M,
                        "seed": seed, "norm_mode": norm_mode, **vals,
                    })
                    print(f"  {shape:<10} M={M:<5} seed={seed} norm={norm_mode:<7} "
                          f"mst_dim={vals['mst_dim']:>10.3f} mst_alpha={vals['mst_alpha']:>7.4f} "
                          f"corr={vals['corr_dim']:>7.3f} twonn={vals['twonn_id']:>7.3f}",
                          flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase0_estimator_audit.csv", index=False)
    print(f"\nwrote {RESULTS / 'phase0_estimator_audit.csv'}  ({len(df)} rows)")

    # MST vs ripser consistency, on a couple of shapes
    cons = {}
    for shape in ("cube_3", "cube_8", "swissroll"):
        rng = np.random.default_rng(7)
        X, _ = make_cloud(shape, 600, rng)
        cons[shape] = mst_vs_ripser_consistency(X)
    (RESULTS / "phase0_mst_vs_ph0.json").write_text(json.dumps(cons, indent=2), encoding="utf-8")
    print("MST vs ripser PH0:", json.dumps(cons, indent=2))

    # weight-matrix metrics: known-rank sanity check
    wrows = []
    rng = np.random.default_rng(0)
    for r in (1, 2, 5, 10, 50, 200):
        W = rng.normal(size=(200, r)) @ rng.normal(size=(r, 200))
        wrows.append({"true_rank": r,
                      "weight_effective_rank": weight_effective_rank(W),
                      "weight_spectral_entropy": weight_spectral_entropy(W)})
    wdf = pd.DataFrame(wrows)
    wdf.to_csv(RESULTS / "phase0_weight_metrics.csv", index=False)
    print("\nweight metrics vs known rank:\n", wdf.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
