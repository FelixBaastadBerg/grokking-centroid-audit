"""
A3 -- does the number of classes K matter on its own, or only through points-per-class m?

The existing E3/E4 sweep cannot answer this. In that sweep the total point count n is held fixed
while K is varied, so m = n/K falls exactly as K rises: the reported rho(fold, K) = +0.954 and
rho(fold, m) = -0.898 are one fact seen twice, not two pieces of evidence. Any claim that "K matters
separately" needs a design in which K and m move independently.

The three quantities satisfy n = K * m identically, so no design can fix all three. What can be done
is to sweep K and m on a grid and let n follow, then ask whether the response is described by m
alone or needs K as well. The paper already establishes that n is the weak factor (rho = -0.245
against -0.898 for m), so a K x m grid confounds K with the weak variable rather than the strong one
-- much better controlled than the original sweep.

Two estimators, chosen to contrast the two classes of the framework:
    ph_dim  spanning support   -- expected to feel K, because each extra component contributes its
                                  own between-component filtration events
    twonn   fixed-order support -- expected not to, once each class supplies two neighbours

Fold is measured against a matched UNCLUSTERED cloud at the same (K, m, seed), the same pairing used
everywhere else in the paper, so the m-dependent finite-sample bias divides out.
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
from analysis import geom
from tools.b1_gride import cloud

OUT = REPO / "RESULTS"
KS, MS, SEEDS = [2, 4, 8, 16, 32], [8, 16, 32, 64], (0, 1, 2)
N_MAX = 1024
D_TRUE = 5


def one(args):
    K, m, seed = args
    out = {"K": K, "m": m, "seed": seed, "n": K * m}
    for sigma, tag in ((0.05, "clustered"), (1.5, "unclustered")):
        X, _ = cloud(D_TRUE, K, m, sigma, seed=seed)
        out[f"ph_dim_{tag}"] = geom.ph_dim(X)[0]
        out[f"twonn_{tag}"] = geom.twonn(X)
    return out


def main():
    from multiprocessing import Pool
    t0 = time.time()
    jobs = [(K, m, s) for K in KS for m in MS for s in SEEDS if K * m <= N_MAX]
    print("=== A3: is K a separate factor, or only m? ===")
    print(f"    {len(jobs)} clouds over K x m, n = K*m follows; true class-conditional d = "
          f"{D_TRUE}\n", flush=True)
    rows = []
    with Pool(processes=10) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs), 1):
            rows.append(r)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    for e in ("ph_dim", "twonn"):
        D[f"{e}_fold"] = D[f"{e}_unclustered"] / D[f"{e}_clustered"]
    D.to_csv(OUT / "a3_cluster_count.csv", index=False)

    for e in ("ph_dim", "twonn"):
        print(f"\n--- {e}: fold change (clustered vs matched unclustered), mean over seeds ---")
        P = D.pivot_table(index="m", columns="K", values=f"{e}_fold")
        print("   m \\ K " + "".join(f"{k:>8}" for k in P.columns))
        for m_, r in P.iterrows():
            print(f"   {m_:>5} " + "".join(f"{v:>8.2f}" if np.isfinite(v) else f"{'--':>8}"
                                           for v in r.values))
        # association with K at FIXED m, which is the question
        per_m = []
        for m_, g in D.groupby("m"):
            gg = g.dropna(subset=[f"{e}_fold"])
            if gg.K.nunique() > 2:
                per_m.append((m_, spearmanr(gg.K, gg[f"{e}_fold"]).statistic))
        print(f"   rho(fold, K) at fixed m: "
              + ", ".join(f"m={m_}: {r:+.2f}" for m_, r in per_m))
        if per_m:
            print(f"   median over m: {np.median([r for _, r in per_m]):+.3f}")
        per_K = []
        for K_, g in D.groupby("K"):
            gg = g.dropna(subset=[f"{e}_fold"])
            if gg.m.nunique() > 2:
                per_K.append((K_, spearmanr(gg.m, gg[f"{e}_fold"]).statistic))
        print(f"   rho(fold, m) at fixed K: "
              + ", ".join(f"K={K_}: {r:+.2f}" for K_, r in per_K))
        if per_K:
            print(f"   median over K: {np.median([r for _, r in per_K]):+.3f}")

    print(f"\nwrote a3_cluster_count.csv   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
