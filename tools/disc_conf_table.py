"""
Rewrite 9: an explicit discovery / confirmation table. Splits are by alternating seeds within
each (task, p, train-fraction) stratum -- the same rule used throughout -- and are applied here
to each headline result so the confirmation half can be read on its own.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from tools.corpus_io import read_corpus
from scipy.stats import spearmanr

OUT = REPO / "RESULTS"


def split_of(df, key="dir"):
    inv = pd.read_csv(OUT / "_inventory.csv").set_index("dir")
    lab = {}
    for _, g in inv.reset_index().groupby(["task", "p", "tf"]):
        for i, d in enumerate(sorted(g.dir)):
            lab[d] = "discovery" if i % 2 == 0 else "confirmation"
    return df[key].map(lab)


def boot(v, n=4000, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 3:
        return (np.nan, np.nan, np.nan)
    g = np.random.default_rng(seed)
    bs = [np.median(g.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if __name__ == "__main__":
    rows = []

    # --- operating point ---
    S = read_corpus("operating_point.csv")
    S["split"] = split_of(S)
    g = S[S.grokked]
    for sp in ("discovery", "confirmation"):
        v = g[g.split == sp].sil_final.values
        m, lo, hi = boot(v)
        rows.append({"hypothesis": "grokked nets reach the clustered regime",
                     "frozen decision": "label silhouette, true classes, final checkpoint",
                     "split": sp, "n": len(v), "effect": f"{m:+.3f}", "ci": f"[{lo:+.3f}, {hi:+.3f}]"})

    # --- redundancy ---
    D = pd.read_csv(OUT / "comeasure.csv")
    D["split"] = split_of(D)
    for sp in ("discovery", "confirmation"):
        per = [spearmanr(x.ph_dim, x.gap_ratio).statistic for _, x in
               D[D.split == sp].groupby("dir") if x.ph_dim.std() > 0 and x.gap_ratio.std() > 0]
        m, lo, hi = boot(per)
        rows.append({"hypothesis": "PH-dim co-moves with NN clusteredness",
                     "frozen decision": "within-run Spearman, ph_dim vs gap_ratio",
                     "split": sp, "n": len(per), "effect": f"{m:+.3f}", "ci": f"[{lo:+.3f}, {hi:+.3f}]"})

    # --- E1 real-activation intervention ---
    E = read_corpus("e1_centroid.csv")
    E["split"] = split_of(E)
    A = E[(E.sweep == "alpha") & (E.norm == "trace")]
    for sig, nm in (("ph_dim", "PH-dim"), ("twonn", "TwoNN")):
        for sp in ("discovery", "confirmation"):
            folds = []
            for d, x in A[A.split == sp].groupby("dir"):
                m0 = x.groupby("alpha")[sig].mean()
                if m0.notna().sum() > 3 and abs(m0.min()) > 1e-9:
                    folds.append(m0.max() / abs(m0.min()))
            m, lo, hi = boot(folds)
            rows.append({"hypothesis": f"E1: {nm} responds to class separation alone",
                         "frozen decision": "alpha sweep, trace-normalised, real activations",
                         "split": sp, "n": len(folds),
                         "effect": f"{m:.2f}x" if np.isfinite(m) else "n/a",
                         "ci": f"[{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else "n/a"})

    # --- timing ---
    T = pd.read_csv(OUT / "geom_timing.csv")
    for sp in ("discovery", "confirmation"):
        s = T[(T.split == sp) & (T.measure == "ph_dim")]
        m, lo, hi = boot(s.phase.values)
        rows.append({"hypothesis": "PH-dim phase precedes accuracy",
                     "frozen decision": "phase, movement-gated, uniform sampling",
                     "split": sp, "n": int(s.phase.notna().sum()),
                     "effect": f"{m:.3f}", "ci": f"[{lo:.3f}, {hi:.3f}]"})

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "discovery_confirmation.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.to_string(index=False))
