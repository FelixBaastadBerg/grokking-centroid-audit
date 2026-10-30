"""
E5 -- preprocessing robustness for every H1 result.
E6 -- detrended redundancy: does the ph_dim / clusteredness association survive removing the
      shared trend in training time?

E5 matters because the source describes preprocessing only as "centered and normalized", which is
ambiguous. If the topology conclusion flips under a different reading, that must be stated.

E6 matters because the raw within-run rho = 0.991 is trajectory CO-MOVEMENT: both series are
monotone in training time, so a high correlation is partly guaranteed. Four checks:
  first differences, residuals after regressing on log-epoch, cross-run association at matched
  checkpoints, and matched-test-accuracy comparison.
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
from ripser import ripser
from tools.intervention import make_cloud, clusteredness

OUT = REPO / "RESULTS"


# ---------------------------------------------------------------- E5
def prep(X, how):
    X = np.asarray(X, float)
    if how == "none":
        Y = X
    elif how == "center":
        Y = X - X.mean(0)
    elif how == "global":                       # centre + Frobenius / sqrt(n)
        Y = X - X.mean(0)
        Y = Y / (np.linalg.norm(Y) / np.sqrt(len(Y)) + 1e-12)
    elif how == "rownorm":
        Y = X - X.mean(0)
        Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    elif how == "whiten":
        Y = X - X.mean(0)
        C = np.cov(Y, rowvar=False) + 1e-8 * np.eye(Y.shape[1])
        w, V = np.linalg.eigh(C)
        Y = Y @ V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    else:
        raise ValueError(how)
    D = np.sqrt(np.maximum(((Y[:, None] - Y[None]) ** 2).sum(-1), 0))
    if how != "nodiam":
        m = D.max()
        D = D / m if m > 0 else D
    return D


def h1(D):
    dg = ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]
    if not len(dg):
        return 0.0, 0.0, 0
    dg = dg[np.isfinite(dg[:, 1])]
    life = dg[:, 1] - dg[:, 0]
    if not len(life):
        return 0.0, 0.0, 0
    return float(life.max()), float(life.sum()), int(len(life))


def e5():
    print("=== E5: does the H1 conclusion depend on preprocessing? ===")
    print("    Arm A fold change of H1 statistics under each pipeline, true d fixed at 5\n")
    rows = []
    for how in ("none", "center", "global", "rownorm", "whiten"):
        recs = []
        for sigma in (1.5, 0.6, 0.25, 0.1, 0.03, 0.01):
            for seed in (0, 1, 2):
                X = make_cloud(5, 8, sigma, n=300, seed=seed)
                mx, tot, nb = h1(prep(X, how))
                recs.append({"sigma": sigma, "H1_max": mx, "H1_total": tot, "n_bars": nb})
        A = pd.DataFrame(recs)
        r = {"preprocessing": how}
        for c in ("H1_max", "H1_total", "n_bars"):
            m = A.groupby("sigma")[c].mean()
            r[c + "_fold"] = m.max() / max(abs(m.min()), 1e-9)
            r[c + "_lo"] = m.min(); r[c + "_hi"] = m.max()
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "e5_preproc.csv", index=False)
    print(R.round(3).to_string(index=False))
    print("\n  'global' is our reading of \"centered and normalized\"; 'rownorm' is the other")
    print("  plausible reading. If the fold change is large under both, the conclusion is")
    print("  invariant to the ambiguity.")


# ---------------------------------------------------------------- E6
def e6():
    print("\n\n=== E6: does the ph_dim / clusteredness redundancy survive detrending? ===\n")
    D = pd.read_csv(OUT / "comeasure.csv").sort_values(["dir", "epoch"])
    res = []

    raw, fd, resid = [], [], []
    for d, g in D.groupby("dir"):
        g = g.dropna(subset=["ph_dim", "gap_ratio", "epoch"])
        if len(g) < 12:
            continue
        raw.append(spearmanr(g.ph_dim, g.gap_ratio).statistic)
        dx, dy = np.diff(g.ph_dim.values), np.diff(g.gap_ratio.values)
        if np.std(dx) > 1e-12 and np.std(dy) > 1e-12:
            fd.append(spearmanr(dx, dy).statistic)
        le = np.log(np.maximum(g.epoch.values, 1))
        ex = g.ph_dim.values - np.polyval(np.polyfit(le, g.ph_dim.values, 1), le)
        ey = g.gap_ratio.values - np.polyval(np.polyfit(le, g.gap_ratio.values, 1), le)
        if np.std(ex) > 1e-12 and np.std(ey) > 1e-12:
            resid.append(spearmanr(ex, ey).statistic)
    res.append(("raw within-run (trajectory co-movement)", np.median(raw), len(raw)))
    res.append(("first differences", np.median(fd), len(fd)))
    res.append(("residuals after regressing on log-epoch", np.median(resid), len(resid)))

    # cross-run at matched checkpoint index
    D["frac"] = D.groupby("dir").t.transform(lambda s: s / max(s.max(), 1))
    cross = []
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        w = D[(D.frac >= lo) & (D.frac < hi)].groupby("dir")[["ph_dim", "gap_ratio"]].median()
        if len(w) > 6:
            cross.append(spearmanr(w.ph_dim, w.gap_ratio).statistic)
    res.append(("cross-run at matched training fraction", np.median(cross), len(cross)))

    # matched test accuracy
    D["accbin"] = pd.cut(D.test_acc, [-0.01, 0.05, 0.3, 0.6, 0.9, 1.01])
    mt = []
    for b, g in D.groupby("accbin", observed=True):
        w = g.groupby("dir")[["ph_dim", "gap_ratio"]].median()
        if len(w) > 6:
            mt.append(spearmanr(w.ph_dim, w.gap_ratio).statistic)
    res.append(("cross-run within matched test-accuracy bins", np.median(mt), len(mt)))

    print(f"  {'check':46s} {'median rho':>11s}  {'n':>4s}")
    for nm, v, n in res:
        print(f"  {nm:46s} {v:>+11.3f}  {n:>4d}")
    pd.DataFrame(res, columns=["check", "rho", "n"]).to_csv(OUT / "e6_detrend.csv", index=False)

    print("\n  structural note: ph_dim and gap_ratio are computed from the SAME nearest-neighbour")
    print("  distances of the same cloud, so they are not two independent measurements; the")
    print("  detrended values say how much of the association is trend and how much is not.")


if __name__ == "__main__":
    t0 = time.time()
    e5(); e6()
    print(f"\ndone in {time.time()-t0:.0f}s")
