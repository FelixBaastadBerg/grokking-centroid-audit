"""
E9 analysis: acceleration with CIs, and whether the intervention is SELECTIVE.

Two readouts, because a fixed epoch confounds "what the regulariser changed" with "how far
training got":
    fixed    epoch 2000, before any run groks
    matched  the memorisation checkpoint (train accuracy first reaches 1.0)
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import spearmanr

OUT = REPO / "RESULTS"
MED = ["eff_rank", "spec_entropy", "pca_pr", "twonn", "mle_k10", "ph_dim", "sil_true", "nc1",
       "nc_wb", "cov_trace", "rep_norm", "weight_norm", "margin", "train_loss", "grad_norm",
       "gap_ratio"]


def boot(v, n=4000, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return (np.nan, np.nan, np.nan)
    g = np.random.default_rng(seed)
    bs = [np.median(g.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if __name__ == "__main__":
    pd.set_option("display.width", 240)
    D = pd.read_csv(OUT / "e9_geomdr.csv")
    runs = D.groupby(["lam", "seed"]).agg(t_grok=("t_grok", "first"),
                                          t_mem=("t_mem", "first"),
                                          final=("test_acc", lambda s: s.values[-3:].mean())
                                          ).reset_index()
    print(f"=== E9: {len(runs)} runs, {runs.lam.nunique()} lambdas x {runs.seed.nunique()} seeds ===\n")

    print("--- acceleration, with bootstrap CIs over seeds ---")
    base = runs[runs.lam == 0].t_grok.values
    b0 = np.median(base)
    print(f"  {'lambda':>8} {'n':>3} {'t_grok median':>14} {'95% CI':>18} {'speed-up':>10} "
          f"{'CI':>16} {'final test':>11}")
    rows = []
    for lam, g in runs.groupby("lam"):
        m, lo, hi = boot(g.t_grok.values)
        rng = np.random.default_rng(1)
        sp = []
        for _ in range(4000):
            a = rng.choice(base, len(base), replace=True)
            b = rng.choice(g.t_grok.values, len(g), replace=True)
            if np.median(b) > 0:
                sp.append(np.median(a) / np.median(b))
        fm, _, _ = boot(g.final.values)
        print(f"  {lam:>8} {len(g):>3} {m:>14.0f} {f'[{lo:.0f}, {hi:.0f}]':>18} "
              f"{b0/m:>10.2f}x {f'[{np.percentile(sp,2.5):.2f}, {np.percentile(sp,97.5):.2f}]':>16} "
              f"{fm:>11.3f}")
        rows.append({"lam": lam, "n": len(g), "t_grok": m, "lo": lo, "hi": hi,
                     "speedup": b0 / m, "sp_lo": np.percentile(sp, 2.5),
                     "sp_hi": np.percentile(sp, 97.5), "final_test": fm})
    pd.DataFrame(rows).to_csv(OUT / "e9_acceleration.csv", index=False)
    print(f"\n  memorisation time t_mem is {sorted(runs.t_mem.unique())} for every run --- the")
    print("  regulariser does not change WHEN the network memorises, so the matched-state")
    print("  readout below compares runs at the same training state.")

    print("\n\n--- is the intervention SELECTIVE? mediators at two readout points ---")
    print("    a selective intervention would move ONE mediator; values are medians over 5 seeds,")
    print("    shown as the ratio to the lambda=0 baseline\n")
    for label, sel in (("FIXED epoch 2000", D[D.epoch == 2000]),
                       ("MATCHED state (memorisation, train acc = 1.0)",
                        D[D.epoch == D.t_mem])):
        piv = sel.groupby("lam")[MED].median()
        base_row = piv.loc[0.0]
        rel = (piv / base_row.replace(0, np.nan)).round(2)
        print(f"  --- {label} ---")
        print(rel.to_string())
        n_moved = ((rel.loc[0.01] - 1).abs() > 0.25).sum()
        print(f"  mediators moved by >25% at lambda=0.01: {n_moved} of {len(MED)}\n")

    print("--- do the mediators separate? correlation with t_grok across all 20 runs ---")
    at2000 = D[D.epoch == 2000].set_index(["lam", "seed"])
    tg = runs.set_index(["lam", "seed"]).t_grok
    rows = []
    for m in MED:
        s = at2000[m].reindex(tg.index)
        ok = s.notna() & tg.notna()
        if ok.sum() > 6:
            rows.append({"mediator": m, "rho_with_t_grok": spearmanr(s[ok], tg[ok]).statistic,
                         "p": spearmanr(s[ok], tg[ok]).pvalue})
    R = pd.DataFrame(rows).sort_values("rho_with_t_grok", key=abs, ascending=False)
    print(R.round(3).to_string(index=False))
    R.to_csv(OUT / "e9_mediators.csv", index=False)
    strong = (R.rho_with_t_grok.abs() > 0.7).sum()
    print(f"\n  {strong} of {len(R)} mediators correlate with t_grok at |rho| > 0.7 --- if many do,")
    print("  no single one is identified as THE mediator.")
