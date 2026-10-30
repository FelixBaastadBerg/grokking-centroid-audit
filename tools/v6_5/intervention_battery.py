"""
v6.5 -- the complete estimator battery for the centroid intervention, on the primary input
population.

Why this exists. The estimator-battery table for the intervention was computed on a fixed
400-point subsample of the training inputs, while the main text, the attribution tables and
Figure 1 use the complete input grid. Both are correct for their own population, but they were
printed with the same interval names, so the same quantity appeared twice with different values
(PH-dimension fold 11.5x against 10.4x). Recomputing the battery on the complete grid removes
the collision by giving the paper one primary population.

Adds the three statistics the existing full-grid sweep did not carry -- total and maximum H_1
persistence, and spectral entropy -- so the table needs no second source.

Reports both aggregations explicitly, because the two differ and the paper must name which it
quotes:
    ratio of medians        median over seeds at alpha=0, divided by median at alpha=8
    median of seed ratios   per-seed fold, then the median over seeds

Analysis-only: retained checkpoints, no training. Writes RESULTS_v6_5/.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from analysis import geom
from analysis.lsrun import LandscapeRun
from tools.linchpin import h1_stats
from tools.b3_seed_alpha import select
from tools.v6_4.fullgrid import full_grid_inputs, acts, trace_norm, leakage, ALPHAS

OUT = REPO / "RESULTS_v6_5"
OUT.mkdir(exist_ok=True)


def battery(Z, y):
    pdim, r2 = geom.ph_dim(Z)
    cd, _ = geom.correlation_dim(Z)
    out = {"ph_dim": pdim, "ph_dim_r2": r2, "twonn": geom.twonn(Z),
           "mle_k10": geom.mle_id(Z, 10), "corr_dim": cd,
           "pca_pr": geom.pca_participation_ratio(Z),
           "eff_rank": geom.effective_rank(Z)}
    Xc = Z - Z.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    s = s[s > 0]
    p = s / s.sum()
    out["spec_entropy"] = float(-(p * np.log(p)).sum())
    h = h1_stats(Z)
    out["H1_total"] = h["H1_total"]
    out["H1_max"] = h["H1_max"]
    for q in (2, 10):
        out[f"L{q}"] = leakage(Z, y, q)
    return out


def one_run(rec):
    d, seed = rec["dir"], int(rec["seed"])
    r = LandscapeRun(REPO / "runs_topo" / d)
    X, y = full_grid_inputs(r.meta)
    H = acts(r, len(r.epochs) - 1, X)
    mu = H.mean(0)
    cen = np.stack([{c: H[y == c].mean(0) for c in np.unique(y)}[c] for c in y])
    bet, wit = cen - mu, H - cen
    rows = []
    for a in ALPHAS:
        Z = trace_norm(mu + a * bet + wit)
        rows.append({"dir": d, "seed": seed, "alpha": a,
                     "n_points": len(y), "n_classes": int(len(np.unique(y))),
                     **battery(Z, y)})
    return rows


ESTS = ["H1_total", "ph_dim", "eff_rank", "pca_pr", "mle_k10", "H1_max",
        "spec_entropy", "corr_dim", "twonn"]
NAMES = {"H1_total": "total $H_1$ persistence", "ph_dim": "PH-dimension",
         "eff_rank": "effective rank", "pca_pr": "PCA participation ratio",
         "mle_k10": "MLE-ID $k{=}10$", "H1_max": "maximum $H_1$ persistence",
         "spec_entropy": "spectral entropy", "corr_dim": "correlation dimension",
         "twonn": "TwoNN"}


def summarise(D):
    rows = []
    for e in ESTS:
        p = D.pivot_table(index="alpha", columns="seed", values=e)
        med = p.median(axis=1)
        folds = (p.loc[0.0] / p.loc[8.0]).dropna()
        rows.append({
            "estimator": e, "name": NAMES[e], "n_seeds": p.shape[1],
            "median_at_0": med[0.0], "median_at_1": med[1.0], "median_at_8": med[8.0],
            "abs_diff_0_8": med[8.0] - med[0.0],
            "fold_ratio_of_medians_0_8": med[0.0] / med[8.0],
            "fold_median_of_ratios_0_8": float(np.median(folds)),
            "fold_seed_min_0_8": float(np.min(folds)),
            "fold_seed_max_0_8": float(np.max(folds)),
            "fold_ratio_of_medians_1_8": med[1.0] / med[8.0],
        })
    return pd.DataFrame(rows)


def main():
    from multiprocessing import Pool
    t0 = time.time()
    inv = pd.read_csv(REPO / "RESULTS" / "_inventory.csv")
    recs = [r for _, r in select(inv).iterrows()]
    print(f"=== v6.5 intervention battery, complete input grid, {len(recs)} primary seeds ===",
          flush=True)
    rows = []
    with Pool(processes=5) as pool:
        for i, rr in enumerate(pool.imap_unordered(one_run, recs), 1):
            rows += rr
            print(f"  [{i}/{len(recs)}] ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "intervention_battery_fullgrid.csv", index=False)
    S = summarise(D)
    S.to_csv(OUT / "intervention_battery_summary.csv", index=False)

    print(f"\n  n = {int(D.n_points.iloc[0])} points, K = {int(D.n_classes.iloc[0])} classes\n")
    print(f"  {'statistic':<26}{'a=0':>8}{'a=8':>8}{'ratio-of-med':>14}"
          f"{'med-of-ratios':>15}{'seed range':>18}")
    for _, r in S.iterrows():
        print(f"  {r.estimator:<26}{r.median_at_0:>8.2f}{r.median_at_8:>8.2f}"
              f"{r.fold_ratio_of_medians_0_8:>13.2f}x{r.fold_median_of_ratios_0_8:>14.2f}x"
              f"   [{r.fold_seed_min_0_8:.2f}, {r.fold_seed_max_0_8:.2f}]")
    print(f"\nwrote intervention_battery_fullgrid.csv, intervention_battery_summary.csv "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
