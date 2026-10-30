"""
THE INTERVENTION. Non-circular demonstration, on clouds where the truth is constructed.

Observational partial correlation cannot separate "dimension" from "clustering" on training
trajectories, because every quantity is monotone in training time (Analysis 1). So instead of
controlling statistically, we CONSTRUCT the two factors independently and vary one at a time.

Construction: K clusters, each an isotropic d-dimensional Gaussian patch of scale sigma, centres
spread at scale R, all embedded in a 60-dimensional ambient space by a random linear map.
The TRUE intrinsic dimension of this set is d for every K and every sigma -- each point lies on a
d-dimensional patch, and a finite union of d-manifolds is still d-dimensional. Clusteredness is
set by sigma/R and measured independently by silhouette.

  ARM A   hold TRUE d fixed, sweep clusteredness.
          If an estimator is measuring dimension, it should not move. Every one of them does.

  ARM B   sweep TRUE d, holding clusteredness FIXED (sigma tuned per d by bisection on
          silhouette). If an estimator is measuring dimension, it should track d.
          At low clusteredness they do -- the positive control. At high clusteredness they stop.

Estimators cover exactly the published family the paper engages: ph_dim (Birdal), correlation
dimension, TwoNN (Facco), MLE-ID (Levina-Bickel), Monod's H1 max/total persistence, and
spectral entropy / effective rank (the 2604.13123 family).
"""
from __future__ import annotations
import os, sys, warnings, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from analysis import geom
from tools.linchpin import h1_stats

OUT = REPO / "RESULTS"
N = 400
AMB = 60


def spectral_entropy(X):
    """Shannon entropy of the normalised covariance eigenspectrum (the 2604.13123 quantity)."""
    Xc = np.asarray(X, float) - np.mean(X, 0)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    lam = lam[lam > 0]
    if not len(lam):
        return np.nan
    p = lam / lam.sum()
    return float(-np.sum(p * np.log(p)))


def make_cloud(d, K, sigma, n=N, amb=AMB, seed=0, R=1.0):
    """K patches of TRUE intrinsic dimension d. True ID = d for every K and sigma."""
    g = np.random.default_rng(seed)
    cen = g.normal(size=(K, amb)) * R if K > 1 else np.zeros((1, amb))
    per = int(np.ceil(n / K))
    P = g.normal(size=(d, amb))                       # shared embedding: same manifold everywhere
    pts = []
    for c in cen:
        z = g.normal(size=(per, d))
        pts.append(c + (z @ P) * sigma)
    return np.vstack(pts)[:n]


def clusteredness(X, kmax=10, seed=0):
    best = -1.0
    for kk in range(2, kmax + 1):
        try:
            lab = KMeans(n_clusters=kk, n_init=3, max_iter=50, random_state=seed).fit_predict(X)
            if len(np.unique(lab)) > 1:
                best = max(best, silhouette_score(X, lab))
        except Exception:
            pass
    return float(best)


def estimators(X):
    pd_, _ = geom.ph_dim(X)
    cd, _ = geom.correlation_dim(X)
    h = h1_stats(X)
    return {"ph_dim": pd_, "corr_dim": cd, "twonn": geom.twonn(X),
            "mle_k10": geom.mle_id(X, 10), "mle_k20": geom.mle_id(X, 20),
            "pca_pr": geom.pca_participation_ratio(X), "eff_rank": geom.effective_rank(X),
            "spec_entropy": spectral_entropy(X),
            "monod_H1_max": h["H1_max"], "monod_H1_total": h["H1_total"]}


def tune_sigma(d, K, target_sil, seed=0, lo=1e-3, hi=3.0, iters=16):
    """Bisect sigma so the cloud hits a target silhouette -- how clusteredness is held fixed."""
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        s = clusteredness(make_cloud(d, K, mid, seed=seed))
        if s > target_sil:
            lo = mid                 # tighter clusters than wanted -> raise sigma
        else:
            hi = mid
    return np.sqrt(lo * hi)


EST = ["ph_dim", "corr_dim", "twonn", "mle_k10", "mle_k20", "pca_pr", "eff_rank",
       "spec_entropy", "monod_H1_max", "monod_H1_total"]

if __name__ == "__main__":
    t0 = time.time()
    pd.set_option("display.width", 220)

    # ---------------------------------------------------------------- GATE
    print("=== GATE: every estimator on a SINGLE blob of known dimension (no clustering) ===\n")
    rows = []
    for d in (2, 4, 6, 8, 10):
        for seed in (0, 1, 2):
            X = make_cloud(d, 1, 1.0, seed=seed)
            rows.append({"true_d": d, "seed": seed, **estimators(X)})
    Gt = pd.DataFrame(rows)
    print(Gt.groupby("true_d")[EST].mean().round(2).to_string())
    from scipy.stats import spearmanr
    print("\n  spearman(estimator, true d) on unclustered clouds:")
    for e in EST:
        print(f"    {e:16s} {spearmanr(Gt[e], Gt.true_d).statistic:+.3f}")
    Gt.to_csv(OUT / "interv_gate.csv", index=False)

    # ---------------------------------------------------------------- ARM A
    print(f"\n\n=== ARM A: TRUE dimension FIXED at d=5, clusteredness swept ===")
    print("    (any movement here is movement that is NOT dimension)\n")
    rows = []
    for sigma in (1.5, 0.8, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01):
        for seed in (0, 1, 2):
            X = make_cloud(5, 8, sigma, seed=seed)
            rows.append({"sigma": sigma, "seed": seed, "true_d": 5,
                         "silhouette": clusteredness(X), **estimators(X)})
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "interv_armA.csv", index=False)
    print(A.groupby("sigma")[["silhouette"] + EST].mean().round(2).to_string())
    print("\n  |spearman(estimator, clusteredness)| with TRUE DIMENSION HELD FIXED:")
    for e in EST:
        r = spearmanr(A[e], A.silhouette).statistic
        rng = A.groupby("sigma")[e].mean()
        print(f"    {e:16s} rho={r:+.3f}   range {rng.min():.2f} -> {rng.max():.2f} "
              f"({rng.max()/max(abs(rng.min()),1e-9):.1f}x)")

    # ---------------------------------------------------------------- ARM B
    print(f"\n\n=== ARM B: TRUE dimension SWEPT, clusteredness HELD FIXED ===")
    print("    (sigma bisected per d to hit each target silhouette)\n")
    rows = []
    for target in (0.0, 0.35, 0.60, 0.85):
        for d in (2, 4, 6, 8, 10):
            for seed in (0, 1, 2):
                K = 1 if target <= 0.0 else 8
                sig = 1.0 if K == 1 else tune_sigma(d, K, target, seed=seed)
                X = make_cloud(d, K, sig, seed=seed)
                rows.append({"target_sil": target, "true_d": d, "seed": seed, "sigma": sig,
                             "silhouette": clusteredness(X), **estimators(X)})
        print(f"  target silhouette {target} done ({time.time()-t0:.0f}s)", flush=True)
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "interv_armB.csv", index=False)
    print()
    print("  achieved silhouette per level:")
    print(B.groupby("target_sil").silhouette.mean().round(3).to_string())
    print("\n  spearman(estimator, TRUE d) at each fixed clusteredness level:")
    tab = {}
    for target, g in B.groupby("target_sil"):
        tab[target] = {e: spearmanr(g[e], g.true_d).statistic for e in EST}
    T = pd.DataFrame(tab).round(3)
    T.columns = [f"sil={c}" for c in T.columns]
    print(T.to_string())
    T.to_csv(OUT / "interv_armB_summary.csv")
    print(f"\ndone in {time.time()-t0:.0f}s")
