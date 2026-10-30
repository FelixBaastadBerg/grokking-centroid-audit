"""
Item 6, second pass -- L_q against estimator ERROR, not against fold change.

The first pass plotted fold change against L_q and got a degenerate answer, for a reason worth
recording: the fold is normalised against each construction's least-clustered cell, and that cell is
exactly the one with the highest leakage. Fold and L_q are therefore confounded by the
normalisation, and binning one by the other says nothing.

The quantity the diagnostic is actually about is estimator ERROR. The true class-conditional
dimension is 5 by construction everywhere in these sweeps, so |estimate - 5| is available directly
and is not normalised against anything. That is what is plotted here, pooled across the centroid
arrangements of item 8 and the adversarial within-class geometries of item 9, and separately for
real activations along the alpha sweep.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import spearmanr

OUT = REPO / "RESULTS"
D_TRUE = 5
BINS = [-1e-9, 1e-9, 0.02, 0.05, 0.12, 0.30, 0.60, 1.01]


def main():
    S = pd.read_csv(OUT / "v2_geometry_sweeps.csv")
    for est in ("ph_dim", "twonn", "corr_dim", "cond_twonn"):
        S[f"err_{est}"] = (S[est] - D_TRUE).abs()
    S.to_csv(OUT / "v2_leakage_axis.csv", index=False)

    print("=== Item 6: estimator error against support leakage (true d = 5 throughout) ===")
    for qcol in ("L2", "L5", "L10"):
        print(f"\n  binned by {qcol}, pooled over all {len(S)} clouds from items 8 and 9")
        S["bin"] = pd.cut(S[qcol], BINS)
        print(f"  {'bin':<16}{'n':>6}" + "".join(f"{'|'+e+'-5|':>16}"
                                                 for e in ("ph_dim", "twonn", "cond_twonn")))
        for b, g in S.groupby("bin", observed=True):
            print(f"  {str(b):<16}{len(g):>6}"
                  + "".join(f"{g[f'err_{e}'].median():>16.2f}"
                            for e in ("ph_dim", "twonn", "cond_twonn")))
        for e in ("ph_dim", "twonn", "cond_twonn"):
            g = S.dropna(subset=[qcol, f"err_{e}"])
            r = spearmanr(g[qcol], g[f"err_{e}"])
            print(f"    rho(error, {qcol}) for {e:<11} {r.statistic:+.3f}  (p = {r.pvalue:.1e})")

    print("\n  same axis on the REAL activations, along the intervention")
    try:
        R = pd.read_csv(OUT / "v2_leakage_alpha.csv")
    except FileNotFoundError:
        print("    v2_leakage_alpha.csv not written yet"); return
    for src, g0 in R.groupby("source"):
        P = g0.groupby("alpha")[["L2", "twonn", "ph_dim"]].median()
        tw0 = P[P.L2 <= 1e-9].twonn
        print(f"    [{src}] TwoNN spread where L2 = 0: "
              f"{tw0.max()/max(abs(tw0.min()),1e-9):.4f}x over {len(tw0)} alpha values; "
              f"whole sweep {P.twonn.max()/max(abs(P.twonn.min()),1e-9):.3f}x")


if __name__ == "__main__":
    main()
