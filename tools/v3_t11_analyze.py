"""
T1.1 (analysis pass) -- is within-class spectral compression the geometric quantity that tracks
grokking?

Inputs: v3_within_series.csv (per-checkpoint within/global/centroid spectra for every MLP run),
temporal_equiv_series.csv (per-checkpoint equivariance order parameter), v3_geomdr_within.csv
(regulariser runs, if finished).

Questions, in the work order's order:
  Q1  does terminal within-class compression distinguish grokked from non-grokked runs?
  Q2  does the onset of within-class compression predict t_grok, controlling for separation?
  Q3  does it co-emerge with the equivariance order parameter?
  Q4  does it generalise across tasks and layers?
  Q5  does the GeomDR regulariser move it?  (separate CSV; reported if present)
  Q6  does it beat GLOBAL effective rank at onset prediction?

Conventions: grokked = final test acc >= 0.9 with train acc ~ 1; t_grok = first epoch reaching
50% of final test accuracy (the locked rule); random-label runs are excluded from grokked/
non-grokked contrasts and reported separately. Onsets use the same half-rise rule as
temporal_equiv (rise_epoch), applied to the DROP of each statistic.

Output: RESULTS/v3_t11_summary.txt (+ v3_t11_run_table.csv)
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

OUT = REPO / "RESULTS"
LINES = []


def say(s=""):
    print(s); LINES.append(s)


def half_fall_epoch(ep, y, tail=5):
    """First epoch at which y has fallen half way from its early plateau to its late plateau."""
    y = np.asarray(y, float); ep = np.asarray(ep, float)
    ok = np.isfinite(y)
    if ok.sum() < 8:
        return np.nan
    e, v = ep[ok], y[ok]
    base, top = np.median(v[:tail]), np.median(v[-tail:])
    if base - top <= 1e-9:
        return np.nan
    hit = np.where(v <= base - 0.5 * (base - top))[0]
    return float(e[hit[0]]) if hit.size else np.nan


def loo_mae(X, y):
    X = np.asarray(X, float).reshape(len(y), -1)
    y = np.asarray(y, float)
    X = np.c_[X, np.ones(len(y))]
    err = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        err.append(abs(X[i] @ beta - y[i]))
    return float(np.mean(err))


def partial_spearman(x, y, z):
    """rho(x, y | z) via rank-regression residuals."""
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        A = np.c_[b, np.ones(len(b))]
        beta, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ beta
    return spearmanr(resid(rx, rz), resid(ry, rz)).statistic


def main():
    D = pd.read_csv(OUT / "v3_within_series.csv")
    # corpus runs + the random-label controls only: the v3 overnight cells (new Z59 seeds,
    # narrow widths) are analysed separately in the T2 sections and must not shift these numbers
    inv = set(pd.read_csv(OUT / "_inventory.csv").dir)
    D = D[D.dir.isin(inv) | D.dir.str.contains("randlab")]
    D = D[D.layer == "last_hidden"].copy()
    say(f"=== T1.1: within-class spectral compression across the corpus ===")
    say(f"    {D.dir.nunique()} MLP runs, last hidden layer\n")

    # per-run terminal summary
    rows = []
    for d, g in D.groupby("dir"):
        g = g.sort_values("epoch")
        fin_test = g.test_acc.values[-3:].mean()
        fin_train = g.train_acc.values[-3:].mean()
        grokked = fin_test >= 0.9
        tg = np.nan
        if fin_test >= 0.5:
            hit = g[g.test_acc >= 0.5 * fin_test]
            tg = float(hit.epoch.min()) if len(hit) else np.nan
        w0, w1 = float(g.within_eff_rank.iloc[0]), float(g.within_eff_rank.iloc[-1])
        gl0, gl1 = float(g.global_eff_rank.iloc[0]), float(g.global_eff_rank.iloc[-1])
        rows.append({
            "dir": d, "task": g.task.iloc[0], "p": int(g.p.iloc[0]),
            "tf": float(g.tf.iloc[0]), "seed": int(g.seed.iloc[0]),
            "random_labels": bool(g.random_labels.iloc[0]),
            "fin_test": fin_test, "fin_train": fin_train, "grokked": grokked, "t_grok": tg,
            "w_er_init": w0, "w_er_fin": w1, "w_er_rel": (w1 - w0) / w0,
            "g_er_rel": (gl1 - gl0) / gl0,
            "sil_fin": float(g.sil_true.iloc[-1]),
            "cen_er_fin": float(g.centroid_eff_rank.iloc[-1]),
            "t_wcomp": half_fall_epoch(g.epoch, g.within_eff_rank),
            "t_gcomp": half_fall_epoch(g.epoch, g.global_eff_rank),
            "t_sil": half_fall_epoch(g.epoch, -g.sil_true),
        })
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "v3_t11_run_table.csv", index=False)

    RL = R[R.random_labels]
    R = R[~R.random_labels]
    gr, ng = R[R.grokked], R[(~R.grokked) & (R.fin_train >= 0.95)]

    say("--- Q1: does terminal within-class compression separate grokked / non-grokked? ---")
    say(f"  grokked     (n={len(gr)}): within-eff-rank change {gr.w_er_rel.median():+.1%} "
        f"[{gr.w_er_rel.quantile(.25):+.1%}, {gr.w_er_rel.quantile(.75):+.1%}]")
    say(f"  non-grokked memorisers (n={len(ng)}): {ng.w_er_rel.median():+.1%} "
        f"[{ng.w_er_rel.quantile(.25):+.1%}, {ng.w_er_rel.quantile(.75):+.1%}]")
    if len(gr) >= 5 and len(ng) >= 5:
        u = mannwhitneyu(gr.w_er_rel, ng.w_er_rel, alternative="less")
        thr = -0.30
        tp = (gr.w_er_rel <= thr).mean(); fp = (ng.w_er_rel <= thr).mean()
        say(f"  Mann-Whitney (grokked more compressed): p = {u.pvalue:.2e}")
        say(f"  single threshold {thr:.0%}: detects {tp:.0%} of grokked at {fp:.0%} of "
            f"non-grokked")
    if len(RL):
        say(f"  random-label memorisers (n={len(RL)}): {RL.w_er_rel.median():+.1%}")
    say("")

    say("--- Q2: onset of within-class compression vs t_grok (grokked runs) ---")
    q = gr.dropna(subset=["t_wcomp", "t_grok"])
    say(f"  n = {len(q)} with measurable onsets")
    r_raw = spearmanr(q.t_wcomp, q.t_grok)
    say(f"  Spearman(t_wcomp, t_grok) = {r_raw.statistic:+.3f} (p = {r_raw.pvalue:.1e})")
    q2 = q.dropna(subset=["t_sil"])
    if len(q2) >= 10:
        pr = partial_spearman(q2.t_wcomp.values, q2.t_grok.values, q2.t_sil.values)
        say(f"  partial, controlling separation onset t_sil: {pr:+.3f}  (n = {len(q2)})")
        say(f"  lead over t_grok: median {float((q.t_grok - q.t_wcomp).median()):+.0f} epochs "
            f"({(q.t_wcomp < q.t_grok).mean():.0%} of runs lead)")
    say("")

    say("--- Q3: co-emergence with the equivariance order parameter ---")
    try:
        E = pd.read_csv(OUT / "temporal_equiv_series.csv")
        E["dir"] = E.run
        rho = []
        for d, g in D.groupby("dir"):
            e = E[E.dir == d]
            if len(e) < 8:
                continue
            m = pd.merge_asof(g.sort_values("epoch"), e.sort_values("epoch")[["epoch", "equiv_R2"]],
                              on="epoch", direction="nearest", tolerance=100).dropna(
                                  subset=["equiv_R2"])
            if len(m) >= 8 and m.within_eff_rank.std() > 0 and m.equiv_R2.std() > 0:
                rho.append(spearmanr(m.within_eff_rank, m.equiv_R2).statistic)
        rho = np.array(rho)
        say(f"  within-run Spearman(within-eff-rank, equiv_R2): median {np.median(rho):+.3f} "
            f"over {len(rho)} runs ({(rho < 0).mean():.0%} negative)")
    except FileNotFoundError:
        say("  temporal_equiv_series.csv missing")
    say("")

    say("--- Q4: across tasks (terminal relative change, grokked runs, last hidden) ---")
    for (task, p), g in gr.groupby(["task", "p"]):
        say(f"  {task}/p{p:<4} n={len(g):>2}  within {g.w_er_rel.median():+.1%}   "
            f"global {g.g_er_rel.median():+.1%}")
    DL = pd.read_csv(OUT / "v3_within_series.csv")
    first = DL[DL.layer == "first_hidden"]
    fr = []
    for d, g in first.groupby("dir"):
        g = g.sort_values("epoch")
        if g.test_acc.values[-3:].mean() >= 0.9 and not g.random_labels.iloc[0]:
            w0, w1 = g.within_eff_rank.iloc[0], g.within_eff_rank.iloc[-1]
            fr.append((w1 - w0) / w0)
    say(f"  first hidden layer, grokked runs: within change median {np.median(fr):+.1%} "
        f"(n = {len(fr)})")
    say("")

    say("--- Q5: GeomDR regulariser (v3_geomdr_within.csv) ---")
    try:
        G = pd.read_csv(OUT / "v3_geomdr_within.csv")
        for lam, g in G.groupby("lam"):
            per = g.groupby("seed").agg(tg=("t_grok", "first"),
                                        w0=("within_eff_rank", "first"),
                                        wmin=("within_eff_rank", "min"))
            e2 = g[g.epoch == 2000].within_eff_rank.median()
            say(f"  lam={lam:<6} t_grok median {per.tg.median():>6.0f}  "
                f"within-eff-rank @2000 {e2:6.1f}  min {per.wmin.median():6.1f}")
        onset = {}
        for (lam, sd), g in G.groupby(["lam", "seed"]):
            onset[(lam, sd)] = half_fall_epoch(g.epoch, g.within_eff_rank)
        o = pd.Series(onset).groupby(level=0).median()
        say(f"  within-compression onset by lam: " +
            ", ".join(f"{l}: {v:.0f}" for l, v in o.items()))
    except FileNotFoundError:
        say("  not finished yet")
    say("")

    say("--- Q6: onset prediction, within vs global effective rank ---")
    q = gr.dropna(subset=["t_grok"])
    for col, lab in [("t_wcomp", "onset of WITHIN-class compression"),
                     ("t_gcomp", "onset of GLOBAL compression"),
                     ("t_sil", "onset of separation (silhouette)")]:
        qq = q.dropna(subset=[col])
        if len(qq) >= 8:
            r = spearmanr(qq[col], qq.t_grok)
            say(f"  {lab:<38} rho = {r.statistic:+.3f}  LOO-MAE = "
                f"{loo_mae(qq[col].values, qq.t_grok.values):.0f}  (n = {len(qq)})")
    # early-value variant (the p1_3 protocol): value at the 10% point of the run
    say("  early-value variant (value at the 10% epoch, p1_3 protocol):")
    ev = []
    for d, g in D.groupby("dir"):
        g = g.sort_values("epoch")
        if g.random_labels.iloc[0]:
            continue
        fin = g.test_acc.values[-3:].mean()
        if fin < 0.9:
            continue
        tg = g[g.test_acc >= 0.5 * fin].epoch.min()
        e_ep = g.epoch.iloc[int(len(g) * 0.10)]
        if not np.isfinite(tg) or e_ep >= tg:
            continue
        r0 = g[g.epoch == e_ep].iloc[0]
        ev.append({"dir": d, "t_grok": tg, "w_er": r0.within_eff_rank,
                   "g_er": r0.global_eff_rank, "sil": r0.sil_true})
    EV = pd.DataFrame(ev)
    if len(EV) >= 8:
        for col, lab in [("w_er", "early within-eff-rank"), ("g_er", "early global-eff-rank"),
                         ("sil", "early silhouette")]:
            r = spearmanr(EV[col], EV.t_grok)
            say(f"    {lab:<26} rho = {r.statistic:+.3f}  LOO-MAE = "
                f"{loo_mae(EV[col].values, EV.t_grok.values):.0f}  (n = {len(EV)})")

    (OUT / "v3_t11_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("\nwrote v3_t11_summary.txt, v3_t11_run_table.csv")


if __name__ == "__main__":
    main()
