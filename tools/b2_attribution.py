"""
B2 -- attribution over the range real networks actually traverse, and B3's statistics.

The headline fold changes in the paper run the intervention to its endpoint (alpha = 8), which puts
the cloud at label-silhouette ~0.96 -- far tighter than any trained network. The fair objection is
that a 12x response measured outside the natural range says little about the 53% PH-dim drop
actually reported through grokking. This file answers it directly:

  1. take each real run's OWN silhouette at initialisation and at convergence;
  2. invert the intervention's silhouette-vs-alpha curve to find the alpha values that reproduce
     exactly those two geometries;
  3. read the estimator change over that MATCHED alpha interval;
  4. compare it with the estimator change the same run shows through training.

The quotient is the attribution fraction: how much of the observed training-time movement in a
statistic is reproduced by moving class separation alone, holding the network's own within-class
geometry fixed. No extrapolation is involved -- the matched interval is interior to the swept grid.

Also reports B3: the seed-replicated response of the canonical cell, with a paired bootstrap over
runs and a discovery/confirmation split by alternating seeds.

Analysis-only. Inputs: b3_seed_alpha.csv (dense alpha grid, 9 seeds), e1_centroid.csv (the 7-run
breadth design), operating_point.csv and comeasure.csv (real training trajectories).
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd

OUT = REPO / "RESULTS"
ESTS = ["ph_dim", "H1_total", "eff_rank", "pca_pr", "twonn", "mle_k10", "corr_dim"]
# name of the same statistic in the training-trajectory files
TRAJ = {"ph_dim": "ph_dim", "eff_rank": "eff_rank", "pca_pr": "pca_pr",
        "twonn": "twonn", "mle_k10": "mle_id_k10", "corr_dim": "corr_dim"}


def alpha_at(sub, target_sil):
    """Invert the run's own silhouette-vs-alpha curve. Monotone in alpha, so interpolation is
    well defined; targets outside the swept range are clipped and flagged."""
    s = sub.sort_values("alpha")
    x, a = s.sil_true.values.astype(float), s.alpha.values.astype(float)
    ok = np.isfinite(x)
    x, a = x[ok], a[ok]
    if len(x) < 3:
        return np.nan, True
    o = np.argsort(x)
    x, a = x[o], a[o]
    clipped = bool(target_sil < x[0] or target_sil > x[-1])
    return float(np.interp(target_sil, x, a)), clipped


def value_at(sub, est, alpha):
    s = sub.sort_values("alpha")
    y = s[est].values.astype(float)
    a = s.alpha.values.astype(float)
    ok = np.isfinite(y)
    return float(np.interp(alpha, a[ok], y[ok])) if ok.sum() >= 3 else np.nan


def bootstrap(v, n=4000, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return np.nan, np.nan, np.nan
    g = np.random.default_rng(seed)
    bs = [np.median(g.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------------------------------------------------------------- B3
def b3(D):
    print("=== B3: seed-replicated centroid intervention, canonical Z_23 add MLP cell ===")
    print(f"    {D.dir.nunique()} distinct seeds, alpha grid {sorted(D.alpha.unique())}\n")
    rows = []
    for est in ESTS:
        folds = []
        for d, g in D.groupby("dir"):
            v = g.set_index("alpha")[est].dropna()
            if len(v) > 3 and abs(v.min()) > 1e-12:
                folds.append(v.max() / abs(v.min()))
        m, lo, hi = bootstrap(folds)
        # discovery / confirmation by alternating seeds, decided before looking
        seeds = sorted(D.seed.unique())
        disc, conf = set(seeds[::2]), set(seeds[1::2])
        f_by_seed = {}
        for d, g in D.groupby("dir"):
            v = g.set_index("alpha")[est].dropna()
            if len(v) > 3 and abs(v.min()) > 1e-12:
                f_by_seed[int(g.seed.iloc[0])] = v.max() / abs(v.min())
        fd = [f for s, f in f_by_seed.items() if s in disc]
        fc = [f for s, f in f_by_seed.items() if s in conf]
        rows.append({"estimator": est, "n_runs": len(folds), "fold_median": m,
                     "ci_lo": lo, "ci_hi": hi,
                     "fold_min": float(np.min(folds)) if folds else np.nan,
                     "fold_max": float(np.max(folds)) if folds else np.nan,
                     "n_disc": len(fd), "fold_disc": float(np.median(fd)) if fd else np.nan,
                     "n_conf": len(fc), "fold_conf": float(np.median(fc)) if fc else np.nan})
    B = pd.DataFrame(rows)
    print(f"  {'estimator':<12}{'n':>4}{'median':>9}{'  95% CI':>18}{'range':>16}"
          f"{'disc':>9}{'conf':>9}")
    for _, r in B.iterrows():
        print(f"  {r.estimator:<12}{r.n_runs:>4}{r.fold_median:>9.2f}"
              f"   [{r.ci_lo:>5.2f}, {r.ci_hi:>5.2f}]"
              f"{r.fold_min:>8.2f}-{r.fold_max:<7.2f}"
              f"{r.fold_disc:>9.2f}{r.fold_conf:>9.2f}")
    B.to_csv(OUT / "b3_summary.csv", index=False)
    return B


# ---------------------------------------------------------------- B2
def real_trajectory_change():
    """
    Per-run estimator change through the transition, and the run's silhouette endpoints.

    The silhouette MUST come from operating_point.csv, not from comeasure.csv's
    `act_silhouette`: that column is the label-free silhouette swept only to k <= 6, which cannot
    resolve p = 23 classes and reads 0.04 -> 0.20 where the true-label value is -0.04 -> +0.65.
    That is documented failure 19 in the appendix, and matching the intervention against it would
    have compressed the real range to a fifth of its size and understated the attribution.
    The intervention's own `sil_true` is a true-label silhouette, so this is the like-for-like join.
    """
    C = pd.read_csv(OUT / "comeasure.csv")
    S = pd.read_csv(OUT / "operating_point.csv").drop_duplicates("dir").set_index("dir")
    tg = {}
    for d, x in C.groupby("dir"):
        x = x.sort_values("epoch")
        fin = x.test_acc.values[-3:].mean()
        tg[d] = int(x[x.test_acc >= 0.5 * fin].t.min()) if fin >= 0.5 else -1
    C["t_grok"] = C.dir.map(tg)
    out = []
    for d, x in C.groupby("dir"):
        if x["t_grok"].iloc[0] < 0 or d not in S.index:
            continue
        x = x.sort_values("epoch")
        t = x["t_grok"].iloc[0]
        rec = {"dir": d, "sil_init": float(S.loc[d, "sil_init"]),
               "sil_final": float(S.loc[d, "sil_final"])}
        for est, col in TRAJ.items():
            if col not in x:
                continue
            a, b = x[x.t < t][col].dropna(), x[x.t > t][col].dropna()
            if len(a) >= 3 and len(b) >= 3:
                rec[f"pre_{est}"], rec[f"post_{est}"] = float(a.median()), float(b.median())
        out.append(rec)
    return pd.DataFrame(out)


def b2(D, label, cell=None):
    """
    `cell` restricts the runs that supply the OBSERVED training-time change, so the numerator and
    denominator describe the same population. The 9-seed arm is entirely Z_23 addition, so it is
    compared against the Z_23 addition runs of the trajectory corpus; the 7-run breadth arm spans
    cells, so it is compared against the whole corpus. Mixing the two would compare an intervention
    on one cell with a training-time change measured on another.
    """
    print(f"\n=== B2: attribution over the range real networks occupy ({label}) ===")
    T = real_trajectory_change()
    if cell:
        T = T[T.dir.str.contains(cell)]
    T = T.reset_index(drop=True)
    sil_lo, sil_hi = float(T.sil_init.median()), float(T.sil_final.median())
    print(f"    trajectory runs: {len(T)}"
          + (f" (restricted to {cell})" if cell else " (whole corpus)"))
    print(f"    real silhouette, median: {sil_lo:+.3f} -> {sil_hi:+.3f}")

    a_lo, a_hi, clip = [], [], 0
    for d, g in D.groupby("dir"):
        x0, c0 = alpha_at(g, sil_lo)
        x1, c1 = alpha_at(g, sil_hi)
        a_lo.append(x0); a_hi.append(x1); clip += int(c0 or c1)
    print(f"    matched alpha interval: {np.nanmedian(a_lo):.2f} -> {np.nanmedian(a_hi):.2f} "
          f"(swept range {D.alpha.min():g}-{D.alpha.max():g}; "
          f"{clip}/{D.dir.nunique()} runs needed clipping)")

    rows = []
    for est in ESTS:
        matched, endpoint = [], []
        for d, g in D.groupby("dir"):
            x0, _ = alpha_at(g, sil_lo)
            x1, _ = alpha_at(g, sil_hi)
            v0, v1 = value_at(g, est, x0), value_at(g, est, x1)
            if np.isfinite(v0) and np.isfinite(v1) and abs(v1) > 1e-12:
                matched.append((v0 - v1) / abs(v0) if abs(v0) > 1e-12 else np.nan)
            vv = g.set_index("alpha")[est].dropna()
            if len(vv) > 3 and abs(vv.min()) > 1e-12:
                endpoint.append(vv.max() / abs(vv.min()))
        mm, mlo, mhi = bootstrap(matched)
        em, _, _ = bootstrap(endpoint)
        rec = {"estimator": est, "matched_frac_change": mm, "ci_lo": mlo, "ci_hi": mhi,
               "endpoint_fold": em, "n_runs": len(matched)}
        # observed training-time change in the same statistic, same sign convention
        if est in TRAJ and f"pre_{est}" in T:
            obs = ((T[f"pre_{est}"] - T[f"post_{est}"]) / T[f"pre_{est}"].abs()).dropna()
            om, olo, ohi = bootstrap(obs.values)
            rec["observed_frac_change"] = om
            rec["attribution"] = mm / om if (np.isfinite(om) and abs(om) > 1e-9) else np.nan
            att = [x / om for x in matched] if abs(om) > 1e-9 else []
            _, alo, ahi = bootstrap(att)
            rec["attr_lo"], rec["attr_hi"] = alo, ahi
            rec["n_obs"] = len(obs)
        rows.append(rec)
    A = pd.DataFrame(rows)
    print(f"\n  {'estimator':<12}{'matched':>10}{'observed':>10}{'attributed':>12}"
          f"{'  95% CI':>18}{'endpoint fold':>15}")
    for _, r in A.iterrows():
        o = r.get("observed_frac_change", np.nan)
        a = r.get("attribution", np.nan)
        ci = (f"  [{r.get('attr_lo', np.nan):>5.2f}, {r.get('attr_hi', np.nan):>5.2f}]"
              if np.isfinite(a) else " " * 18)
        print(f"  {r.estimator:<12}{r.matched_frac_change:>9.1%}"
              f"{o:>10.1%}" if np.isfinite(o) else
              f"  {r.estimator:<12}{r.matched_frac_change:>9.1%}{'--':>10}", end="")
        print(f"{a:>11.0%}" if np.isfinite(a) else f"{'--':>11}", end="")
        print(f"{ci}{r.endpoint_fold:>14.1f}x")
    A["source"] = label
    return A


if __name__ == "__main__":
    D3 = pd.read_csv(OUT / "b3_seed_alpha.csv")
    b3(D3)
    A3 = b2(D3, "9 seeds, canonical cell, dense grid", cell="task-add__p-23")

    E1 = pd.read_csv(OUT / "e1_centroid.csv")
    E1 = E1[(E1.sweep == "alpha") & (E1.norm == "trace")]
    A1 = b2(E1, "7-run breadth design, E1 grid")

    pd.concat([A3, A1]).to_csv(OUT / "b2_attribution.csv", index=False)
    print("\nwrote b2_attribution.csv, b3_summary.csv")
    print("\n  'attributed' is the matched-range change divided by the observed training-time")
    print("  change in the same statistic: the share of the reported movement that varying class")
    print("  separation alone reproduces, with the network's own within-class geometry fixed.")
