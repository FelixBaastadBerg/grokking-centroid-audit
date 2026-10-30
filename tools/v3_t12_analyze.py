"""
T1.2 (analysis pass) -- summarise v3_dissociate.csv at matched realised silhouette.

Per family and estimator: interpolate each seed's estimator-vs-silhouette curve at sil = 0.5
(inside every family's swept range except `circular`, which cannot reach it -- itself reported),
then median + percentile bootstrap over seeds. Also the two exact checks:
  * fourier3_perm vs fourier3: same union cloud geometry -- label-free statistics must agree
    within seed noise while the equivariance order parameter collapses;
  * fourier3_phase vs fourier3: phase randomisation is an isometry of the centroid set.
Output: RESULTS/v3_dissociate_summary.csv
"""
from __future__ import annotations
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd

OUT = REPO / "RESULTS"
ESTS = ["ph_dim", "twonn", "eff_rank", "H1_total", "H1_max"]
TARGET = 0.5


def main():
    D = pd.read_csv(OUT / "v3_dissociate.csv")
    rng = np.random.default_rng(0)
    rows = []
    for fam, g in D.groupby("family"):
        rec = {"family": fam,
               "sil_max_reached": float(g.groupby("seed").sil_true.max().median()),
               "equiv_R2": float(g.equiv_R2_centroids.median()),
               "cen_eff_rank": float(g.cen_eff_rank.median())}
        for e in ESTS:
            vals = []
            for sd, gg in g.groupby("seed"):
                gg = gg.sort_values("sil_true")
                if gg.sil_true.min() <= TARGET <= gg.sil_true.max():
                    vals.append(float(np.interp(TARGET, gg.sil_true, gg[e])))
            if len(vals) >= 5:
                v = np.array(vals)
                bs = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(4000)]
                rec[e] = float(np.median(v))
                rec[f"{e}_lo"], rec[f"{e}_hi"] = (float(np.percentile(bs, 2.5)),
                                                  float(np.percentile(bs, 97.5)))
            else:
                rec[e] = np.nan
        rows.append(rec)
    S = pd.DataFrame(rows).set_index("family")
    S.to_csv(OUT / "v3_dissociate_summary.csv")

    print("=== T1.2 at matched realised silhouette 0.5 ===")
    print(S[["ph_dim", "twonn", "eff_rank", "H1_max", "H1_total", "equiv_R2",
             "cen_eff_rank", "sil_max_reached"]].round(3).to_string())
    fams = [f for f in S.index if np.isfinite(S.loc[f, "ph_dim"])]
    print(f"\nPH-dim across families: {S.loc[fams,'ph_dim'].min():.2f}"
          f"-{S.loc[fams,'ph_dim'].max():.2f}  (spread "
          f"{S.loc[fams,'ph_dim'].max()/S.loc[fams,'ph_dim'].min():.2f}x)")
    print(f"H1_max fourier vs (random_full+simplex)/2: "
          f"{S.loc['fourier3','H1_max']/np.mean([S.loc['random_full','H1_max'], S.loc['simplex','H1_max']]):.2f}x")
    print(f"equiv_R2: fourier {S.loc['fourier3','equiv_R2']:.3f} -> permuted "
          f"{S.loc['fourier3_perm','equiv_R2']:.3f}; random {S.loc['random_full','equiv_R2']:.3f}")
    print(f"circular max silhouette reached: {S.loc['circular','sil_max_reached']:.3f}")


if __name__ == "__main__":
    main()
