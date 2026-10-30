"""
LINCHPIN analysis. Headline claim, per the corrected positioning:

  The general ID/clustering confound is PRIOR ART (Campadelli 2015, Camastra 2016; and the
  neural-collapse <-> ID link outside grokking, 2510.01105). We do not claim it.

  What is NOT covered by that literature, and is claimed here:
    (1) the confound extends to TOPOLOGY -- Monod et al.'s H1 signal, not a dimension estimator
    (2) it defuses specific LEADING/CAUSAL claims, which a general correlation-confound cannot

So the plain ID columns are reported as setup, and H1 is reported first.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import spearmanr
from tools.linchpin import partial_spearman, ccf_lag

OUT = REPO / "RESULTS"
CLUST = "silhouette"


if __name__ == "__main__":
    D = pd.read_csv(OUT / "linchpin_series.csv").sort_values(["dir", "epoch"])
    pd.set_option("display.width", 210)
    G = D[D.groupby("dir").test_acc.transform("last") >= 0.9]
    print(f"loaded {D.shape}; {D.dir.nunique()} runs, {G.dir.nunique()} grokked\n")

    # ---------------------------------------------------------------- P1 reproduce
    print("=== P1: does Monod's H1 signal reproduce on our runs? ===")
    print("    (their object: token-embedding rows; their statistic: H1 max / total persistence)\n")
    rows = []
    for dirn, g in G.groupby("dir"):
        g = g.sort_values("epoch")
        fin = g.test_acc.values[-3:].mean()
        t50 = g[g.test_acc >= 0.5 * fin].epoch.min()
        pre = g[g.epoch < t50]; post = g[g.epoch >= t50]
        if len(pre) < 3 or len(post) < 3:
            continue
        lag_max, r_max = ccf_lag(g.H1_max.values, g.test_acc.values)
        step = int(np.median(np.diff(g.epoch.values))) if len(g) > 2 else np.nan
        rows.append({"dir": dirn, "arch": g.arch.iloc[0], "p": g.p.iloc[0],
                     "H1max_pre": pre.H1_max.max(), "H1max_post": post.H1_max.max(),
                     "H1max_peak_epoch": g.loc[g.H1_max.idxmax(), "epoch"], "t50": t50,
                     "peak_rel_t50": g.loc[g.H1_max.idxmax(), "epoch"] - t50,
                     "lag_ckpts": lag_max, "lag_r": r_max, "ckpt_step": step})
    P1 = pd.DataFrame(rows)
    P1["lag_steps"] = P1.lag_ckpts * P1.ckpt_step
    print(f"  n={len(P1)} grokked runs")
    print(f"  H1_max peak occurs at median {P1.peak_rel_t50.median():+.0f} epochs relative to t50")
    print(f"  CCF-of-first-differences lag (their statistic; >0 = topology moves AFTER accuracy):")
    print(f"    median {P1.lag_steps.median():+.0f} steps   "
          f"(IQR {P1.lag_steps.quantile(.25):+.0f} to {P1.lag_steps.quantile(.75):+.0f}),  "
          f"lags in {(P1.lag_steps>0).mean():.0%} of runs")
    print(f"    their reported figure for transformers is ~+1000 steps (topology lags)\n")
    P1.to_csv(OUT / "linchpin_p1.csv", index=False)

    # ---------------------------------------------------------------- P2 the control
    print("=== P2 (HEADLINE): does the H1 signal survive a clustering control? ===")
    print(f"    clustering metric: {CLUST} (label-free; gate-validated 0.07 blob -> 0.98 tight)\n")
    res = []
    for sig in ["H1_max", "H1_total", "H1_nbars"]:
        raw, par, rc = [], [], []
        for dirn, g in G.groupby("dir"):
            s = g[[sig, "test_acc", CLUST]].dropna()
            if len(s) < 8 or s[sig].std() == 0:
                continue
            raw.append(spearmanr(s[sig], s.test_acc).statistic)
            rc.append(spearmanr(s[sig], s[CLUST]).statistic)
            par.append(partial_spearman(s[sig].values, s.test_acc.values, s[CLUST].values))
        res.append({"signal": sig, "n": len(raw), "rho_vs_test_acc": np.median(raw),
                    "rho_vs_clustering": np.median(rc),
                    "partial_given_clustering": np.nanmedian(par),
                    "frac_partial_below_0.2": float(np.mean(np.abs(par) < 0.2))})
    R = pd.DataFrame(res).round(3)
    print(R.to_string(index=False))
    R.to_csv(OUT / "linchpin_p2.csv", index=False)

    print("\n  same control on the plain ID estimators (SETUP -- this part is prior art):")
    setup = []
    for sig in ["gap_ratio", "nn1_ratio"]:
        raw, par = [], []
        for dirn, g in G.groupby("dir"):
            s = g[[sig, "test_acc", CLUST]].dropna()
            if len(s) < 8 or s[sig].std() == 0:
                continue
            raw.append(spearmanr(s[sig], s.test_acc).statistic)
            par.append(partial_spearman(s[sig].values, s.test_acc.values, s[CLUST].values))
        setup.append({"signal": sig, "n": len(raw), "rho_vs_test_acc": np.median(raw),
                      "partial_given_clustering": np.nanmedian(par)})
    print(pd.DataFrame(setup).round(3).to_string(index=False))

    # does clustering itself survive a control by H1? (direction check)
    print("\n  REVERSE direction -- is clustering explained by H1? (if both collapse, neither")
    print("  dominates and the claim must be weakened to 'not separable')")
    rev = []
    for sig in ["H1_max", "H1_total"]:
        par = []
        for dirn, g in G.groupby("dir"):
            s = g[[sig, "test_acc", CLUST]].dropna()
            if len(s) < 8 or s[CLUST].std() == 0:
                continue
            par.append(partial_spearman(s[CLUST].values, s.test_acc.values, s[sig].values))
        rev.append({"controlled_for": sig, "n": len(par),
                    "partial_clustering_vs_testacc": np.nanmedian(par)})
    print(pd.DataFrame(rev).round(3).to_string(index=False))

    # ---------------------------------------------------------------- timing under control
    print("\n=== P2b: is the LEAD/LAG structure also a clustering effect? ===")
    print("    CCF lag of H1 vs accuracy, compared with CCF lag of CLUSTERING vs accuracy\n")
    lg = []
    for dirn, g in G.groupby("dir"):
        g = g.sort_values("epoch")
        step = int(np.median(np.diff(g.epoch.values)))
        lh, _ = ccf_lag(g.H1_max.values, g.test_acc.values)
        lc, _ = ccf_lag(g[CLUST].values, g.test_acc.values)
        lg.append({"dir": dirn, "lag_H1": lh * step if np.isfinite(lh) else np.nan,
                   "lag_clustering": lc * step if np.isfinite(lc) else np.nan})
    L = pd.DataFrame(lg)
    print(f"  median lag, H1 vs accuracy        : {L.lag_H1.median():+.0f} steps")
    print(f"  median lag, clustering vs accuracy: {L.lag_clustering.median():+.0f} steps")
    print(f"  correlation between the two lags across runs: "
          f"{spearmanr(L.lag_H1, L.lag_clustering, nan_policy='omit').statistic:+.3f}")
    L.to_csv(OUT / "linchpin_lags.csv", index=False)
