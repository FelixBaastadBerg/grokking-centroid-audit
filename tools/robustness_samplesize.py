"""
Sample-size / |G|-ceiling sensitivity for the surviving claims.

This matters more than a generic robustness check, because the two objects in play have very
different point counts:

  activations       p^2 * train_fraction   -- 423 at p=23, up to ~3500 at p=59
  token embeddings  exactly |G|            -- 23, 24, 59, 120.  Monod's object lives HERE.

So a triage conclusion established at n=400 may or may not transfer to the n=23 cloud that the
topological claims are actually computed on. Both intervention arms are therefore re-run across
n = 24 … 1600, and the estimator triage is reported per point count.
"""
from __future__ import annotations
import os, sys, warnings, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import spearmanr
from tools.intervention import make_cloud, clusteredness, estimators, EST, tune_sigma

OUT = REPO / "RESULTS"
SIZES = [24, 48, 100, 200, 400, 800, 1600]

if __name__ == "__main__":
    t0 = time.time()
    pd.set_option("display.width", 230)

    print("=== S1: ARM A fold change vs point count (true d fixed at 5) ===")
    print("    does 'estimators respond to clustering' survive at |G|-scale clouds?\n")
    rows = []
    for n in SIZES:
        recs = []
        K = 8 if n >= 48 else 4
        for sigma in (1.5, 0.6, 0.25, 0.1, 0.03, 0.01):
            for seed in (0, 1, 2):
                X = make_cloud(5, K, sigma, n=n, seed=seed)
                recs.append({"sigma": sigma, **estimators(X)})
        A = pd.DataFrame(recs)
        r = {"n": n}
        for e in EST:
            m = A.groupby("sigma")[e].mean()
            r[e] = m.max() / max(abs(m.min()), 1e-9)
        rows.append(r)
        print(f"  n={n} done ({time.time()-t0:.0f}s)", flush=True)
    RA = pd.DataFrame(rows).set_index("n")
    RA.to_csv(OUT / "robust_size_armA.csv")
    print("\n  FOLD CHANGE by point count:")
    print(RA.round(1).to_string())

    print("\n\n=== S2: ARM B (true d swept, clusteredness 0.70) vs point count ===")
    print("    rho(estimator, true d); does the triage survive at small n?\n")
    rows = []
    for n in SIZES:
        K = 8 if n >= 48 else 4
        recs = []
        for d in (2, 4, 6, 8, 10):
            for seed in (0, 1, 2):
                sig = tune_sigma(d, K, 0.70, seed=seed)
                X = make_cloud(d, K, sig, n=n, seed=seed)
                recs.append({"true_d": d, "sil": clusteredness(X), **estimators(X)})
        B = pd.DataFrame(recs)
        r = {"n": n, "sil": B.sil.mean()}
        for e in EST:
            r[e] = spearmanr(B[e], B.true_d).statistic
        rows.append(r)
        print(f"  n={n} done ({time.time()-t0:.0f}s)", flush=True)
    RB = pd.DataFrame(rows).set_index("n")
    RB.to_csv(OUT / "robust_size_armB.csv")
    print("\n  rho(estimator, TRUE d) by point count:")
    print(RB.round(2).to_string())

    print("\n\n=== S3: the |G| ceiling directly -- n = 23, 24, 59, 120 (real embedding sizes) ===\n")
    rows = []
    for n in (23, 24, 59, 120):
        recs = []
        for d in (2, 4, 6, 8):
            for seed in range(5):
                sig = tune_sigma(d, 4, 0.70, seed=seed)
                X = make_cloud(d, 4, sig, n=n, seed=seed)
                recs.append({"true_d": d, **estimators(X)})
        B = pd.DataFrame(recs)
        r = {"n": n}
        for e in EST:
            r[e] = spearmanr(B[e], B.true_d).statistic
        rows.append(r)
        print(f"  n={n} done ({time.time()-t0:.0f}s)", flush=True)
    RC = pd.DataFrame(rows).set_index("n")
    RC.to_csv(OUT / "robust_size_Gceiling.csv")
    print("\n  rho(estimator, TRUE d) at |G|-scale point counts, clusteredness 0.70:")
    print(RC.round(2).to_string())
    print(f"\ndone in {time.time()-t0:.0f}s")
