"""
E3 + E4 -- disentangle total point count from points-per-cluster, and replicate at the exact
moduli the examined topological work uses.

E3: vary K in {1,2,4,8} at fixed n, across n in {23,24,59,113,149,197,400}. The earlier "|G|
ceiling" wording conflated two things: how many points there are in total, and how many points
there are per cluster. This separates them so the claim can be worded to match the data.

E4: n = 113, 149, 197 are the moduli used by the examined topological work; the confound
magnitude at those scales is what actually bears on their setting, and is reported prominently
rather than only the n = 24 extreme.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import spearmanr
from tools.intervention import make_cloud, clusteredness, estimators, EST, tune_sigma

OUT = REPO / "RESULTS"
NS = [23, 24, 59, 113, 149, 197, 400]
KS = [1, 2, 4, 8]

if __name__ == "__main__":
    t0 = time.time()
    pd.set_option("display.width", 240)

    print("=== E3/E4: Arm A fold change by (n, K), true dimension fixed at 5 ===")
    print("    points-per-cluster = n/K is shown so the two factors can be read apart\n")
    rows = []
    for n in NS:
        for K in KS:
            if n // K < 6:
                continue
            recs = []
            for sigma in (1.5, 0.6, 0.25, 0.1, 0.03, 0.01):
                for seed in (0, 1, 2):
                    X = make_cloud(5, K, sigma, n=n, seed=seed)
                    recs.append({"sigma": sigma, **estimators(X)})
            A = pd.DataFrame(recs)
            r = {"n": n, "K": K, "per_cluster": n // K}
            for e in EST:
                m = A.groupby("sigma")[e].mean()
                r[e] = m.max() / max(abs(m.min()), 1e-9) if m.notna().sum() > 2 else np.nan
            rows.append(r)
        print(f"  n={n} done ({time.time()-t0:.0f}s)", flush=True)
    RA = pd.DataFrame(rows)
    RA.to_csv(OUT / "e34_armA.csv", index=False)

    cols = ["ph_dim", "corr_dim", "monod_H1_max", "monod_H1_total", "mle_k10", "twonn"]
    print("\n  fold change (true d fixed at 5):")
    print(RA[["n", "K", "per_cluster"] + cols].round(1).to_string(index=False))

    print("\n  --- E4: at the moduli the examined work uses ---")
    tgt = RA[RA.n.isin([113, 149, 197])].groupby("n")[cols].median().round(1)
    print(tgt.to_string())
    print("\n  vs the extreme and the mid range:")
    print(RA[RA.n.isin([23, 24, 400])].groupby("n")[cols].median().round(1).to_string())

    print("\n  --- E3: is it total n, or points per cluster? ---")
    sub = RA.dropna(subset=["ph_dim"])
    for c in ["ph_dim", "monod_H1_total", "monod_H1_max"]:
        s = sub.dropna(subset=[c])
        if len(s) > 6:
            rn = spearmanr(s.n, s[c]).statistic
            rp = spearmanr(s.per_cluster, s[c]).statistic
            rk = spearmanr(s.K, s[c]).statistic
            print(f"    {c:16s} spearman with n {rn:+.3f}   with points-per-cluster {rp:+.3f}"
                  f"   with K {rk:+.3f}")

    print("\n\n=== Arm B: rho(estimator, true d) at clusteredness 0.70, by n ===\n")
    rows = []
    for n in NS:
        K = 4 if n < 100 else 8
        recs = []
        for d in (2, 4, 6, 8):
            for seed in (0, 1, 2):
                sig = tune_sigma(d, K, 0.70, seed=seed)
                X = make_cloud(d, K, sig, n=n, seed=seed)
                recs.append({"true_d": d, **estimators(X)})
        B = pd.DataFrame(recs)
        r = {"n": n, "K": K}
        for e in EST:
            r[e] = spearmanr(B[e], B.true_d).statistic if B[e].notna().sum() > 6 else np.nan
        rows.append(r)
        print(f"  n={n} done ({time.time()-t0:.0f}s)", flush=True)
    RB = pd.DataFrame(rows)
    RB.to_csv(OUT / "e34_armB.csv", index=False)
    print()
    print(RB[["n", "K"] + cols + ["spec_entropy", "eff_rank"]].round(2).to_string(index=False))
    print(f"\ndone in {time.time()-t0:.0f}s")
