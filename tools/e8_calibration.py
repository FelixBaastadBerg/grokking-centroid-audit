"""
E8 (scoped) -- calibration study behind the "monotonic vs approximately calibrated" wording and
the TwoNN recommendation.

On UNCLUSTERED clouds of known dimension: is the statistic merely monotone in d, or is it
numerically close to d? Reports slope, bias, MAE and rank correlation. Then the same estimators'
clustering-sensitivity from Arm A, so the recommendation rests on both properties at once.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import spearmanr
from tools.intervention import make_cloud, estimators, EST

OUT = REPO / "RESULTS"
DIMS = [2, 3, 4, 5, 6, 8, 10]
CAL = ["ph_dim", "corr_dim", "twonn", "mle_k10", "mle_k20", "pca_pr", "eff_rank"]

if __name__ == "__main__":
    rows = []
    for d in DIMS:
        for seed in range(5):
            X = make_cloud(d, 1, 1.0, n=400, seed=seed)
            rows.append({"true_d": d, "seed": seed, **estimators(X)})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "e8_calibration.csv", index=False)

    print("=== E8: calibration on UNCLUSTERED clouds (n=400, ambient 60, 5 seeds) ===")
    print("    monotone = tracks d in rank; calibrated = numerically close to d\n")
    A = pd.read_csv(OUT / "robust_armA_boot.csv")
    out = []
    for e in EST:
        s = D[["true_d", e]].dropna()
        if len(s) < 10:
            continue
        rho = spearmanr(s.true_d, s[e]).statistic
        sl, ic = np.polyfit(s.true_d, s[e], 1)
        r = {"statistic": e, "rank_rho": rho, "slope": sl}
        if e in CAL:
            r["bias"] = float((s[e] - s.true_d).mean())
            r["MAE"] = float((s[e] - s.true_d).abs().mean())
        m = A.groupby("sigma")[e].mean()
        r["clustering_fold"] = m.max() / max(abs(m.min()), 1e-9)
        out.append(r)
    R = pd.DataFrame(out)
    R["calibrated"] = np.where(R.MAE.notna() & (R.MAE < 1.2), "yes",
                               np.where(R.MAE.notna(), "no", "n/a (different units)"))
    print(R.round(3).to_string(index=False))
    R.to_csv(OUT / "e8_summary.csv", index=False)
    print("\n  'n/a (different units)' marks statistics that are not on a dimension scale at all")
    print("  (spectral entropy, H1 persistence), for which calibration is undefined and only")
    print("  monotonicity can be assessed.")
