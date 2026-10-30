"""
T0.4 -- local sensitivity ratio at the real operating regime.

The v3 framing is non-identifiability: the observed signal is not diagnostic of
class-conditional dimension because separation alone can reproduce it -- NOT because the
estimators have lost all dimension sensitivity (Arm B shows PH-dim still tracks true d).
The honest quantity is the local ratio

    R  =  |ds / dlog gamma|  /  |ds / dd|

evaluated where real networks sit (m ~ 17 points per class, K = 23, silhouette ~ 0.65):
how many "dimensions" of apparent change does one e-fold of separation produce? R >> 1 means
concurrent changes of separation swamp plausible changes of dimension; R ~ 0 means the
statistic reads dimension cleanly.

Finite differences on the standard synthetic construction: d in {4,5,6} at the matched gamma,
gamma in {1.5, 3, 6} at d = 5; 8 seeds. Output: RESULTS/v3_sensitivity.csv
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from tools.e1_centroid import measure

OUT = REPO / "RESULTS"
K, M, AMB = 23, 17, 60
GAMMAS = (1.5, 3.0, 6.0)      # log-spaced around the converged operating point
DS = (4, 5, 6)
SEEDS = range(8)
ESTS = ["ph_dim", "twonn", "eff_rank", "pca_pr", "H1_total", "mle_k10", "corr_dim"]


def cloud(d, gamma, seed):
    sigma = 1.0 / (gamma * np.sqrt(d))
    g = np.random.default_rng(seed)
    P = g.normal(size=(d, AMB))
    cen = g.normal(size=(K, AMB)) * 3
    # centres are scale-3: separation-to-scale gamma is set against unit centre scale, so
    # rescale sigma accordingly (same convention as the Arm A sweep via GAMMA_OF)
    X = np.vstack([cen[k] + (g.normal(size=(M, d)) @ P) * sigma * 3 for k in range(K)])
    Xc = X - X.mean(0)
    return Xc / (np.sqrt((Xc ** 2).sum(1).mean()) + 1e-12), np.repeat(np.arange(K), M)


def main():
    rows, t0 = [], time.time()
    for d in DS:
        for gamma in GAMMAS:
            if d != 5 and gamma != 3.0:
                continue                     # the two finite-difference axes only
            for sd in SEEDS:
                X, y = cloud(d, gamma, sd)
                rows.append({"d": d, "gamma": gamma, "seed": sd, **measure(X, y)})
        print(f"  d={d} done ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "v3_sensitivity_raw.csv", index=False)

    mid = D[(D.d == 5)].groupby("gamma")[ESTS].median()
    dd = D[(D.gamma == 3.0)].groupby("d")[ESTS].median()
    out = []
    print(f"\n  {'estimator':<10}{'ds/dlog gamma':>15}{'ds/dd':>10}{'R':>8}   sil at operating "
          f"point: {D[(D.d==5)&(D.gamma==3.0)].sil_true.median():.3f}")
    for e in ESTS:
        dgd = (mid.loc[6.0, e] - mid.loc[1.5, e]) / np.log(6.0 / 1.5)
        ddd = (dd.loc[6, e] - dd.loc[4, e]) / 2.0
        R = abs(dgd) / max(abs(ddd), 1e-9)
        out.append({"estimator": e, "ds_dloggamma": float(dgd), "ds_dd": float(ddd),
                    "R": float(R)})
        print(f"  {e:<10}{dgd:>15.3f}{ddd:>10.3f}{R:>8.1f}")
    pd.DataFrame(out).to_csv(OUT / "v3_sensitivity.csv", index=False)
    print(f"\nwrote v3_sensitivity.csv ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
