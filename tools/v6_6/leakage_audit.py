"""
v6.6 -- measured support leakage behind the two explanations that v6.5 rested on class size.

v6.5 explains the raised TwoNN between-class share on the Z_59 cell, and the raised TwoNN fold
on the held-out-only intervention, by "~7 points per class" and "m = 5" respectively. Small m
makes cross-class leakage more likely, but it is not itself the estimator-support condition:
the proposition of the paper needs L_2 = 0 at BOTH endpoints of the transformation. This script
measures L_2 directly on exactly the clouds those two claims are made about, so the wording can
rest on the measured quantity rather than on class size.

Two audits, both analysis-only re-evaluations of saved checkpoints (no training):

  crosscell   For the Z_59 and S_4 cells (and, for reference, the primary Z_23 cell measured
              the same way), rebuild the endpoint 2x2 recombination of tools/v3_t21_crosscell.py
              -- same 400-point subsample, same fixed rng seed, same trace normalisation -- and
              measure L_2 on all four recombined clouds. f(C0,R0) and f(C1,R1) are the network's
              real initial and converged states; f(C0,R1) and f(C1,R0) are the hybrids. The
              containment condition for the centroid factor is L_2 = 0 at the two clouds a
              one-factor step connects, so it is reported for both order steps.

  splits      For the primary cell, rerun the centroid intervention of tools/b5_splits.py on the
              train / test / full-grid inputs and measure L_2 at every alpha, so the held-out
              TwoNN response can be attributed to measured leakage rather than to m alone.

Outputs (RESULTS_v6_6/, nothing existing is touched):
  crosscell_leakage.csv          per run x cell: L_2 on all four recombination cells
  crosscell_leakage_summary.csv  per cell: median and seed range, containment verdicts
  splits_leakage.csv             per seed x split x alpha: L_2 and TwoNN
  splits_leakage_summary.csv     per split: m, L_2 at alpha = 1, TwoNN fold, leaky share
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
import torch
from sklearn.neighbors import NearestNeighbors

from analysis import geom
from analysis.lsrun import LandscapeRun
from tools.e1_centroid import decompose, recompose
from tools.b3_seed_alpha import select, ALPHAS

OUT = REPO / "RESULTS_v6_6"
OUT.mkdir(exist_ok=True)
RUNS = REPO / "runs_topo"
N_SUB = 400          # identical to v3_t21_crosscell.py and b5_splits.py

CELLS = {
    "Z59_add_mlp": lambda n: "task-add__p-59" in n and n.endswith("__hb"),
    "S4_perm_mlp": lambda n: ("arch-mlp__task-perm__p-24" in n and "tf-0.8" in n
                              and n.endswith("tf-0.8")),
}


def leakage(Z, y, q=2):
    """Fraction of points whose q nearest neighbours are not all in the point's own class."""
    nn = NearestNeighbors(n_neighbors=q + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    lab = np.asarray(y)[idx[:, 1:q + 1]]
    return float(np.mean((lab != np.asarray(y)[:, None]).any(1)))


def parts(H, y):
    cen = np.zeros_like(H)
    for c in np.unique(y):
        cen[y == c] = H[y == c].mean(0)
    return cen, H - cen


def trace_norm(Z):
    Zc = Z - Z.mean(0)
    return Zc / (np.sqrt((Zc ** 2).sum(1).mean()) + 1e-12)


def acts_pair(d):
    """(H_init, H_final, y) on the same fixed subsample v3_t21_crosscell.py uses."""
    r = LandscapeRun(d)
    n = len(r.epochs)

    def acts(i):
        r._load(r.flat(i)); r.model.eval()
        with torch.no_grad():
            r.model(r.Xtr, keep_activations=True)
            return r.model.activations[-2].numpy()

    y_all = r.Ytr.numpy()
    H0, H1 = acts(0), acts(n - 1)
    fin = float(r.curves.test_acc.values[-5:].mean())
    sub = np.sort(np.random.default_rng(0).choice(len(y_all), min(N_SUB, len(y_all)),
                                                  replace=False))
    return H0[sub], H1[sub], y_all[sub], fin


def crosscell():
    # the primary cell measured under the same protocol, as the reference row
    prim = set(select(pd.read_csv(REPO / "RESULTS" / "_inventory.csv")).dir)
    cells = dict(CELLS, Z23_add_mlp=lambda n: n in prim)

    rows, t0 = [], time.time()
    for cell, match in cells.items():
        dirs = [d for d in sorted(RUNS.iterdir())
                if d.is_dir() and match(d.name) and (d / "trajectory.npz").exists()]
        print(f"--- {cell}: {len(dirs)} runs ---", flush=True)
        for d in dirs:
            try:
                H0, H1, y, fin = acts_pair(d)
            except Exception as e:
                print(f"  SKIP {d.name}: {type(e).__name__} {e}", flush=True)
                continue
            if fin < 0.9:
                print(f"  skip (not grokked) {d.name[:55]}", flush=True)
                continue
            c0, r0 = parts(H0, y)
            c1, r1 = parts(H1, y)
            K = len(np.unique(y))
            rec = {"cell": cell, "dir": d.name, "final_test": fin, "n_points": len(y),
                   "n_classes": K, "points_per_class": len(y) / K,
                   "train_fraction": float(d.name.split("tf-")[1].split("__")[0])}
            for cs, rs, tag in ((c0, r0, "C0R0"), (c1, r0, "C1R0"),
                                (c0, r1, "C0R1"), (c1, r1, "C1R1")):
                Z = trace_norm(cs + rs)
                rec[f"L2_{tag}"] = leakage(Z, y, 2)
                rec[f"twonn_{tag}"] = geom.twonn(Z)
            rows.append(rec)
            print(f"  {d.name[:58]}  L2 {rec['L2_C0R0']:.3f}/{rec['L2_C1R1']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "crosscell_leakage.csv", index=False)

    summ = []
    for cell, g in D.groupby("cell"):
        rec = {"cell": cell, "n_runs": len(g),
               "points_per_class_median": g.points_per_class.median(),
               "points_per_class_min": g.points_per_class.min(),
               "points_per_class_max": g.points_per_class.max(),
               "train_fractions": ";".join(sorted({f"{v:g}" for v in g.train_fraction}))}
        for tag in ("C0R0", "C1R0", "C0R1", "C1R1"):
            rec[f"L2_{tag}_median"] = g[f"L2_{tag}"].median()
            rec[f"L2_{tag}_min"] = g[f"L2_{tag}"].min()
            rec[f"L2_{tag}_max"] = g[f"L2_{tag}"].max()
        # the centroid factor is the step C0 -> C1 at fixed residual state; the two-endpoint
        # containment condition needs L_2 = 0 at both clouds of that step
        rec["contained_centroid_step_Rinit"] = int(
            ((g.L2_C0R0 == 0) & (g.L2_C1R0 == 0)).sum())
        rec["contained_centroid_step_Rconv"] = int(
            ((g.L2_C0R1 == 0) & (g.L2_C1R1 == 0)).sum())
        rec["contained_converged_only"] = int((g.L2_C1R1 == 0).sum())
        summ.append(rec)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "crosscell_leakage_summary.csv", index=False)
    print("\n=== L_2 on the endpoint recombination cells (median over grokked runs) ===")
    print(S[["cell", "n_runs", "points_per_class_median", "L2_C0R0_median", "L2_C1R0_median",
             "L2_C0R1_median", "L2_C1R1_median", "contained_centroid_step_Rconv"]]
          .round(3).to_string(index=False))
    return D, S


def splits():
    inv = pd.read_csv(REPO / "RESULTS" / "_inventory.csv")
    sel = select(inv)
    rows, t0 = [], time.time()
    for i, (_, rec) in enumerate(sel.iterrows()):
        try:
            r = LandscapeRun(RUNS / rec["dir"])
            r._load(r.flat(len(r.epochs) - 1)); r.model.eval()
        except Exception as e:
            print(f"  skip {rec['dir'][:50]}: {type(e).__name__}", flush=True)
            continue
        Xf = torch.cat([r.Xtr, r.Xte]); Yf = torch.cat([r.Ytr, r.Yte])
        for split, X, Y in (("train", r.Xtr, r.Ytr), ("test", r.Xte, r.Yte), ("full", Xf, Yf)):
            with torch.no_grad():
                r.model(X, keep_activations=True)
                H = r.model.activations[-2].numpy()
            y = Y.numpy()
            n = min(N_SUB, len(y))
            sub = np.sort(np.random.default_rng(0).choice(len(y), n, replace=False))
            H, y = H[sub], y[sub]
            if len(np.unique(y)) < 2:
                continue
            mu, bet, wit = decompose(H, y)
            for a in ALPHAS:
                Z = recompose(mu, bet, wit, a, 1.0, norm="trace")
                rows.append({"dir": rec["dir"], "seed": int(rec["seed"]), "split": split,
                             "alpha": a, "n_points": len(y),
                             "points_per_class": len(y) / len(np.unique(y)),
                             "twonn": geom.twonn(Z), "ph_dim": geom.ph_dim(Z)[0],
                             "L2": leakage(Z, y, 2)})
        print(f"  [{i+1}/{len(sel)}] seed {int(rec['seed']):>2} ({time.time()-t0:.0f}s)",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT / "splits_leakage.csv", index=False)

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "splits_leakage.csv", index=False)

    summ = []
    for split, g in D.groupby("split"):
        p = g.pivot_table(index="alpha", columns="seed", values="L2")
        tw = g.pivot_table(index="alpha", columns="seed", values="twonn")
        ph = g.pivot_table(index="alpha", columns="seed", values="ph_dim")
        contained = [a for a in p.index if (p.loc[a].dropna() == 0).all()]
        summ.append({
            "split": split,
            "points_per_class": g.points_per_class.median(),
            "n_points": g.n_points.median(),
            "L2_at_alpha1_median": float(p.loc[1.0].median()),
            "L2_at_alpha1_min": float(p.loc[1.0].min()),
            "L2_at_alpha1_max": float(p.loc[1.0].max()),
            "L2_at_alpha8_median": float(p.loc[8.0].median()),
            "seeds_L2_zero_at_alpha1": int((p.loc[1.0] == 0).sum()),
            "seeds_L2_zero_at_alpha8": int((p.loc[8.0] == 0).sum()),
            "alpha_contained_all_seeds": min(contained) if contained else np.nan,
            "twonn_fold_0_8": float(tw.loc[0.0].median() / tw.loc[8.0].median()),
            "twonn_fold_0_1": float(tw.loc[0.0].median() / tw.loc[1.0].median()),
            "twonn_fold_1_8": float(tw.loc[1.0].median() / tw.loc[8.0].median()),
            "ph_fold_0_8": float(ph.loc[0.0].median() / ph.loc[8.0].median()),
            "ph_fold_0_1": float(ph.loc[0.0].median() / ph.loc[1.0].median()),
            "ph_fold_1_8": float(ph.loc[1.0].median() / ph.loc[8.0].median()),
        })
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "splits_leakage_summary.csv", index=False)
    print("\n=== the intervention by input split: measured leakage, not m alone ===")
    print(S.round(3).to_string(index=False))
    return D, S


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "crosscell"):
        crosscell()
    if which in ("all", "splits"):
        splits()
