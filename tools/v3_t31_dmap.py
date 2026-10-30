"""
T3.1 -- the reliability map across true dimension d in {2, 5, 10, 20}, and whether the mixture
variable collapses the surfaces.

The published map (Fig. 1a-b) is computed at d = 5 only, so the m ~ 20 / silhouette ~ 0.5
boundary could in principle be a d = 5 accident. This repeats the Arm A sweep (same m x sigma
grid, same construction as twonn_floor.csv, K = 8) at four true dimensions and asks:

  1. does the qualitative boundary (PH-dim fold >= 1.25 at m <~ 20 or sil >~ 0.5; TwoNN flat in
     the sub-gap regime) hold at every d?
  2. does the T1.3 mixture variable u = gamma^alpha m^-(1-alpha/d) collapse the fold surfaces
     across d, where the naive u = gamma m^-kappa cannot?

Output: RESULTS/v3_dmap.csv, printed boundary summary.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from analysis import geom
from tools.intervention import make_cloud, clusteredness

OUT = REPO / "RESULTS"
DS = [2, 5, 10, 20]
MS = [4, 8, 16, 32, 64, 128]
SIGMAS = [0.02, 0.05, 0.1, 0.25, 0.6, 1.5]
SEEDS = range(3)
K = 8


def main():
    t0 = time.time()
    rows = []
    for d in DS:
        for m in MS:
            for sig in SIGMAS:
                for seed in SEEDS:
                    X = make_cloud(d, K, sig, n=m * K, seed=seed)
                    rows.append({"d": d, "m": m, "sigma": sig, "seed": seed,
                                 "silhouette": clusteredness(X),
                                 "twonn": geom.twonn(X), "ph_dim": geom.ph_dim(X)[0]})
            print(f"  d={d} m={m} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "v3_dmap.csv", index=False)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "v3_dmap.csv", index=False)

    print("\n=== boundary summary: fold against the least-clustered cell of the same (d, m) ===")
    D["gamma"] = 1.0 / (D.sigma * np.sqrt(D.d))
    res = []
    for (d, m), g in D.groupby(["d", "m"]):
        gg = g.groupby("sigma")[["ph_dim", "twonn", "silhouette"]].mean().sort_index()
        base = gg.iloc[-1]                      # largest sigma = least clustered
        for sig, r in gg.iterrows():
            res.append({"d": d, "m": m, "sigma": sig,
                        "gamma": 1.0 / (sig * np.sqrt(d)),
                        "sil": r.silhouette,
                        "ph_fold": base.ph_dim / r.ph_dim if r.ph_dim > 0 else np.nan,
                        "tw_fold": base.twonn / r.twonn if r.twonn > 0 else np.nan})
    R = pd.DataFrame(res)
    R.to_csv(OUT / "v3_dmap_folds.csv", index=False)
    for d, g in R.groupby("d"):
        # smallest m at which the row stays below 1.25x fold everywhere with sil < 0.5
        safe = g[(g.ph_fold >= 1.25) & np.isfinite(g.ph_fold)]
        bd_m = safe.m.max() if len(safe) else np.nan
        s05 = g[(g.sil >= 0.5) & np.isfinite(g.ph_fold)]
        say = (f"  d={d:>2}: PH fold>=1.25 seen up to m={bd_m}; at sil>=0.5 median PH fold "
               f"{s05.ph_fold.median():5.2f}, TwoNN fold {s05.tw_fold.median():5.2f}")
        print(say)

    print("\n=== collapse test: R^2 of fold ~ sigmoid(u), naive vs mixture u ===")
    from scipy.optimize import curve_fit
    def sigm(u, A, u0, b):
        return A + (1 - A) / (1 + (u / u0) ** b)
    ok = R.dropna(subset=["ph_fold"])
    ok = ok[ok.ph_fold > 0]
    y = 1.0 / ok.ph_fold.values                  # response relative to unclustered
    for name, u in [("naive  u = gamma * m^-0.3 (no d)", ok.gamma * ok.m ** -0.3),
                    ("mixture u = gamma * m^-(1-1/d)", ok.gamma * ok.m ** -(1 - 1 / ok.d))]:
        try:
            popt, _ = curve_fit(sigm, u, y, p0=[0.2, 1.0, 1.0], maxfev=20000)
            r2 = 1 - np.sum((y - sigm(u, *popt)) ** 2) / np.sum((y - y.mean()) ** 2)
        except Exception:
            r2 = np.nan
        print(f"  {name:<38} R^2 = {r2:.3f}")


if __name__ == "__main__":
    main()
