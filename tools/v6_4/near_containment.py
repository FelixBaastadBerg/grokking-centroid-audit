"""
v6.4 Phase 2.3 -- what happens between exact invariance and no guarantee at all.

The proposition gives exact invariance when L_q = 0 both before and after the transformation.
Real representations can have small but non-zero leakage, and the paper should say what is
actually known about that regime without inventing a tolerance it has not proved.

Reference, fixed before inspecting the results: a seed's CLASS-CONTAINED PLATEAU value is its
estimate at the largest separation on the sweep (alpha = 8), where L_q = 0 for every seed and
every estimator considered. Deviation is |estimate(alpha) - plateau|, reported per seed rather
than as an aggregate curve, against the leakage L_q measured at the same alpha.

Three regimes are separated in the output, and the paper must not blur them:
  exact licence          L_q = 0 at both endpoints -> the estimate cannot move at all
  empirical reassurance  small non-zero L_q in THESE sweeps -> small observed movement
  no guarantee           nothing here proves a bound for arbitrary non-zero leakage

Reads RESULTS_v6_4/fullgrid_sweep.csv. Writes RESULTS_v6_4/near_containment*.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

OUT = REPO / "RESULTS_v6_4"
PAIRS = [("twonn", "L2"), ("mle_k10", "L10")]


def main():
    S = pd.read_csv(OUT / "fullgrid_sweep.csv")
    S = S[S.pipeline == "raw"].copy()

    rows = []
    for est, lk in PAIRS:
        for seed, g in S.groupby("seed"):
            g = g.sort_values("alpha")
            plateau = float(g[g.alpha == g.alpha.max()][est].iloc[0])
            for _, r in g.iterrows():
                rows.append({"seed": int(seed), "estimator": est, "leak_col": lk,
                             "alpha": r.alpha, "leakage": r[lk], "value": r[est],
                             "plateau": plateau,
                             "abs_dev": abs(r[est] - plateau),
                             "rel_dev": abs(r[est] - plateau) / abs(plateau)})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "near_containment.csv", index=False)

    # first alpha at which leakage is exactly zero, per seed
    first = []
    for (est, seed), g in D.groupby(["estimator", "seed"]):
        z = g[g.leakage == 0].sort_values("alpha")
        first.append({"estimator": est, "seed": seed,
                      "alpha_Lq_zero": float(z.alpha.min()) if len(z) else np.nan,
                      "max_abs_dev_after_zero": float(z.abs_dev.max()) if len(z) else np.nan,
                      "max_rel_dev_after_zero": float(z.rel_dev.max()) if len(z) else np.nan})
    F = pd.DataFrame(first)
    F.to_csv(OUT / "near_containment_exact.csv", index=False)

    # banded description of the leaky regime -- empirical only, no fitted tolerance
    D["band"] = pd.cut(D.leakage, [-1e-12, 0, 0.01, 0.05, 0.25, 1.0],
                       labels=["Lq = 0", "0 < Lq <= 0.01", "0.01 < Lq <= 0.05",
                               "0.05 < Lq <= 0.25", "Lq > 0.25"])
    B = (D.groupby(["estimator", "band"], observed=True)
           .agg(n=("rel_dev", "size"), n_seeds=("seed", "nunique"),
                median_rel_dev=("rel_dev", "median"),
                max_rel_dev=("rel_dev", "max"),
                max_abs_dev=("abs_dev", "max")).reset_index())
    B.to_csv(OUT / "near_containment_bands.csv", index=False)

    print("=== exact regime: first alpha with L_q = 0, and the movement after it ===")
    for est in D.estimator.unique():
        q = F[F.estimator == est]
        print(f"  {est:<9} L_q = 0 from alpha "
              f"{q.alpha_Lq_zero.min():.3g}-{q.alpha_Lq_zero.max():.3g} across seeds; "
              f"max |deviation| over the whole contained stretch "
              f"{q.max_abs_dev_after_zero.max():.2e} "
              f"({q.max_rel_dev_after_zero.max():.2e} relative)")

    print("\n=== leaky regime: observed movement by leakage band (empirical, not a bound) ===")
    print(f"  {'estimator':<9}{'band':<20}{'n':>4}{'seeds':>7}"
          f"{'median rel dev':>16}{'max rel dev':>14}{'max abs dev':>14}")
    for _, r in B.iterrows():
        print(f"  {r.estimator:<9}{str(r.band):<20}{r.n:>4}{r.n_seeds:>7}"
              f"{r.median_rel_dev:>15.2%}{r.max_rel_dev:>14.2%}{r.max_abs_dev:>14.3f}")

    print("\n=== per-seed detail at the smallest non-zero leakage on the sweep ===")
    nz = D[(D.leakage > 0) & (D.estimator == "twonn")]
    m = nz.loc[nz.groupby("seed").leakage.idxmin()]
    print(m[["seed", "alpha", "leakage", "value", "plateau", "abs_dev", "rel_dev"]]
          .round(5).to_string(index=False))

    # contrast: what PH-dimension does over the same exactly-contained stretch
    P = S[S.alpha >= 1.5].groupby("seed").ph_dim.agg(["max", "min"])
    P["ratio"] = P["max"] / P["min"]
    print(f"\n=== over the SAME exactly-contained stretch (alpha >= 1.5, L_2 = 0 in every seed):")
    print(f"  TwoNN moves by at most {F[F.estimator=='twonn'].max_abs_dev_after_zero.max():.1e} "
          f"units; PH-dimension falls by a factor of "
          f"{P.ratio.min():.2f}-{P.ratio.max():.2f} across seeds.")
    print("\nwrote near_containment.csv, near_containment_exact.csv, near_containment_bands.csv")


if __name__ == "__main__":
    main()
