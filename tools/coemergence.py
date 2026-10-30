"""
P5 -- do clustering and the group's representation geometry emerge together?

The estimators this paper is about respond to clustering. On the group-structured tasks the
representation is also acquiring the group's symmetry, and an equivariance order parameter
(prior work) rises through the transition. If the two rise together, then a "geometric progress
measure" that is really tracking clustering will nonetheless appear to track symmetry emergence --
which is exactly why these statistics are attractive and exactly why they cannot be read as
dimension.

This is DESCRIPTIVE AND CORRELATIONAL. Nothing here says symmetry drives grokking, or that
clustering drives symmetry; the claim is co-emergence in time on the same checkpoints.

Data, no new training:
  RESULTS/geom_series.csv           equiv_R2 co-measured with gap_ratio / nc_wb at the SAME
                                    checkpoints of the same model state (57 group-structured runs)
  RESULTS/comeasure.csv x
  RESULTS/temporal_equiv_series.csv an independent join giving the label-silhouette against true
                                    classes on a denser epoch grid (20 runs)

The paper's own warning applies to us: within a run nearly every quantity is monotone in training
time, so a raw within-run correlation is partly guaranteed. Every number below is therefore
reported raw AND under the two detrending controls used in Appendix E6 -- first differences and
residuals after regressing on log epoch.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

OUT = REPO / "RESULTS"
GROUP_TASKS = ("add", "perm")          # modular arithmetic on Z_p, composition in S_n
N_MIN = 6                              # checkpoints required before a run enters a correlation


def within_run(df, a, b, mode="raw"):
    """Per-run Spearman between two co-measured series, raw or detrended."""
    out = []
    for _, x in df.groupby("dir"):
        x = x.sort_values("epoch")
        u, v = x[a].values.astype(float), x[b].values.astype(float)
        le = np.log10(np.maximum(x["epoch"].values.astype(float), 1.0))
        m = np.isfinite(u) & np.isfinite(v)
        u, v, le = u[m], v[m], le[m]
        if len(u) < N_MIN:
            continue
        if mode == "diff":
            u, v = np.diff(u), np.diff(v)
        elif mode == "resid":
            A = np.vstack([le, np.ones(len(le))]).T
            u = u - A @ np.linalg.lstsq(A, u, rcond=None)[0]
            v = v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
        if len(u) < N_MIN or np.std(u) < 1e-12 or np.std(v) < 1e-12:
            continue
        out.append(spearmanr(u, v).statistic)
    return np.array(out)


def onset(t, y, frac=0.5):
    """First epoch at which a series has covered `frac` of its own rise over the run."""
    m = np.isfinite(y)
    t, y = t[m], y[m]
    if len(y) < N_MIN:
        return np.nan
    lo, hi = y[0], np.median(y[-3:])
    if hi - lo <= 1e-9:
        return np.nan
    i = np.where(y >= lo + frac * (hi - lo))[0]
    return float(t[i[0]]) if len(i) else np.nan


def row(name, r):
    p = float(wilcoxon(r).pvalue) if len(r) > 5 else np.nan
    return {"check": name, "n": len(r), "median_rho": float(np.median(r)),
            "q25": float(np.percentile(r, 25)), "q75": float(np.percentile(r, 75)),
            "frac_positive": float(np.mean(r > 0)), "wilcoxon_p": p}


def main():
    G = pd.read_csv(OUT / "geom_series.csv")
    G = G[G.task.isin(GROUP_TASKS)].copy()
    # gap_ratio FALLS as the cloud clusters; flip it so the order parameter and the clustering
    # statistic both rise, and a positive rho means "they emerge together".
    G["nn_clusteredness"] = 1.0 - G["gap_ratio"]
    G["grokked"] = G.final_test >= 0.9

    C = pd.read_csv(OUT / "comeasure.csv")
    T = pd.read_csv(OUT / "temporal_equiv_series.csv")
    J = C.merge(T[["run", "epoch", "equiv_R2"]], left_on=["dir", "epoch"],
                right_on=["run", "epoch"], how="inner")
    J = J[J.task.isin(GROUP_TASKS)].copy()

    print(f"co-measured corpus: {G.dir.nunique()} group-structured runs, {len(G)} checkpoints "
          f"({int(G.groupby('dir').size().median())} per run)")
    print(f"independent join:   {J.dir.nunique()} runs, {len(J)} checkpoints\n")

    rows = []
    print("=== NN-clusteredness (1 - gap ratio) vs equivariance order parameter ===")
    print(f"  {'check':44s} {'n':>4s} {'median rho':>11s} {'frac>0':>7s}")
    for lbl, sub in [("all group-structured runs", G), ("grokked runs only", G[G.grokked])]:
        for mode, tag in [("raw", "raw within-run"), ("diff", "first differences"),
                          ("resid", "residuals on log epoch")]:
            r = within_run(sub, "nn_clusteredness", "equiv_R2", mode)
            nm = f"{lbl}, {tag}"
            rows.append(dict(source="geom_series", statistic="nn_clusteredness", **row(nm, r)))
            print(f"  {nm:44s} {len(r):>4d} {np.median(r):>+11.3f} {np.mean(r > 0):>7.2f}")

    print("\n=== label-silhouette vs equivariance order parameter (independent join) ===")
    for mode, tag in [("raw", "raw within-run"), ("diff", "first differences"),
                      ("resid", "residuals on log epoch")]:
        r = within_run(J, "act_silhouette", "equiv_R2", mode)
        nm = f"label-silhouette, {tag}"
        rows.append(dict(source="comeasure x temporal_equiv", statistic="act_silhouette",
                         **row(nm, r)))
        print(f"  {nm:44s} {len(r):>4d} {np.median(r):>+11.3f} {np.mean(r > 0):>7.2f}")

    print("\n=== by task (raw within-run, geom_series) ===")
    for tk, x in G.groupby("task"):
        r = within_run(x, "nn_clusteredness", "equiv_R2", "raw")
        rows.append(dict(source="geom_series", statistic="nn_clusteredness",
                         **row(f"task = {tk}, raw within-run", r)))
        print(f"  task {tk:5s} n={len(r):3d} median rho {np.median(r):+.3f}")

    # ---------------------------------------------------------------- onsets
    recs = []
    for d, x in G.groupby("dir"):
        x = x.sort_values("epoch")
        t = x.epoch.values.astype(float)
        recs.append(dict(dir=d, task=x.task.iloc[0], p=int(x.p.iloc[0]),
                         grokked=bool(x.grokked.iloc[0]),
                         t_equiv=onset(t, x.equiv_R2.values.astype(float)),
                         t_clust=onset(t, x.nn_clusteredness.values.astype(float)),
                         t_acc=onset(t, x.test_acc.values.astype(float))))
    O = pd.DataFrame(recs).dropna(subset=["t_equiv", "t_clust"])
    O.to_csv(OUT / "coemergence_onsets.csv", index=False)
    print("\n=== onset of each (50% of the run's own rise) ===")
    for lbl, D_ in [("all", O), ("grokked", O[O.grokked])]:
        if len(D_) < 6:
            continue
        s = spearmanr(D_.t_equiv, D_.t_clust)
        lg = np.log10(np.maximum(D_[["t_equiv", "t_clust"]].values, 1.0))
        rp = float(np.corrcoef(lg[:, 0], lg[:, 1])[0, 1])
        print(f"  [{lbl:8s}] n={len(D_):3d}  median t_equiv={D_.t_equiv.median():.0f}  "
              f"t_clust={D_.t_clust.median():.0f}  rho={s.statistic:+.3f} (p={s.pvalue:.3g})  "
              f"pearson on log10 = {rp:+.3f}")
        rows.append({"source": "geom_series", "statistic": "onset_epoch",
                     "check": f"rho(t_equiv, t_clust), {lbl}", "n": len(D_),
                     "median_rho": float(s.statistic), "q25": np.nan, "q75": np.nan,
                     "frac_positive": np.nan, "wilcoxon_p": float(s.pvalue)})
        rows.append({"source": "geom_series", "statistic": "onset_epoch",
                     "check": f"pearson log10(t_equiv, t_clust), {lbl}", "n": len(D_),
                     "median_rho": rp, "q25": np.nan, "q75": np.nan,
                     "frac_positive": np.nan, "wilcoxon_p": np.nan})

    pd.DataFrame(rows).to_csv(OUT / "coemergence.csv", index=False)
    print("\nwrote coemergence.csv, coemergence_onsets.csv")
    print("\n  Correlational only. Both series rise through training, so the detrended rows are")
    print("  the load-bearing ones; the raw row is reported for comparability with Appendix E6.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time()-t0:.0f}s")
