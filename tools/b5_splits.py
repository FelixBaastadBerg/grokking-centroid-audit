"""
B5 -- is the confound a property of the training activations only?

Everything in the paper is measured on activations at the TRAINING inputs, which invites the fair
question: grokking is a statement about generalisation, so why look at the set the network fit?
Three answers are possible and all three are publishable, so the measurement is worth making:

  train      the p^2 * tf inputs the network was fit on          (what the paper reports)
  test       the held-out inputs                                  (generalisation set)
  full       every element of the group table, train and test     (the whole representation)

For each split we re-run the identical centroid intervention and record the response of each
estimator, plus the label-silhouette so the clustering itself can be compared across splits. If the
confound is the same on all three, the training-set choice is immaterial and the objection
dissolves; if it is markedly weaker off the training set, that is a scope boundary and we say so.

The full grid is the honest comparator for cloud size: the test split has fewer points per class
than the train split at tf = 0.8, and points-per-class is exactly what the paper shows drives the
effect, so any train/test difference has to be read against m. Both are reported.

Analysis-only: retained checkpoints, no training.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from analysis.lsrun import LandscapeRun
from tools.e1_centroid import decompose, recompose, measure
from tools.b3_seed_alpha import select, ALPHAS

OUT = REPO / "RESULTS"
ESTS = ["ph_dim", "H1_total", "twonn", "mle_k10", "corr_dim", "eff_rank", "pca_pr"]


def activations(r, X):
    with torch.no_grad():
        r.model(X, keep_activations=True)
        return r.model.activations[-2].numpy()


def main():
    inv = pd.read_csv(OUT / "_inventory.csv")
    sel = select(inv)
    print(f"=== B5: the intervention on train / test / full-grid activations ===")
    print(f"    {len(sel)} seeds of the canonical cell, alpha grid {ALPHAS}\n", flush=True)

    rows, t0 = [], time.time()
    for i, (_, rec) in enumerate(sel.iterrows()):
        try:
            r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
            r._load(r.flat(len(r.epochs) - 1)); r.model.eval()
        except Exception as e:
            print(f"  skip {rec['dir'][:50]}: {type(e).__name__}"); continue
        Xf = torch.cat([r.Xtr, r.Xte]); Yf = torch.cat([r.Ytr, r.Yte])
        for split, X, Y in (("train", r.Xtr, r.Ytr), ("test", r.Xte, r.Yte), ("full", Xf, Yf)):
            H, y = activations(r, X), Y.numpy()
            n = min(400, len(y))
            sub = np.sort(np.random.default_rng(0).choice(len(y), n, replace=False))
            H, y = H[sub], y[sub]
            if len(np.unique(y)) < 2:
                continue
            mu, bet, wit = decompose(H, y)
            for a in ALPHAS:
                Z = recompose(mu, bet, wit, a, 1.0, norm="trace")
                rows.append({"dir": rec["dir"], "seed": int(rec["seed"]), "split": split,
                             "alpha": a, "n_points": len(y),
                             "points_per_class": len(y) / len(np.unique(y)), **measure(Z, y)})
        pd.DataFrame(rows).to_csv(OUT / "b5_splits.csv", index=False)
        print(f"  [{i+1}/{len(sel)}] seed {int(rec['seed']):>2} ({time.time()-t0:.0f}s)", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "b5_splits.csv", index=False)

    print("\n--- fold change by split (median over seeds) ---")
    print(f"  {'split':<7}{'m':>6}{'n':>6}" + "".join(f"{e:>11}" for e in ESTS))
    summ = []
    for split, g in D.groupby("split"):
        r_ = {"split": split, "points_per_class": g.points_per_class.median(),
              "n_points": g.n_points.median()}
        for e in ESTS:
            folds = []
            for d, gg in g.groupby("dir"):
                v = gg.set_index("alpha")[e].dropna()
                if len(v) > 3 and abs(v.min()) > 1e-12:
                    folds.append(v.max() / abs(v.min()))
            r_[e] = float(np.median(folds)) if folds else np.nan
        summ.append(r_)
        print(f"  {split:<7}{r_['points_per_class']:>6.0f}{r_['n_points']:>6.0f}"
              + "".join(f"{r_[e]:>10.2f}x" for e in ESTS))
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "b5_splits_summary.csv", index=False)

    print("\n--- silhouette reached at alpha = 1 (the unmodified representation) ---")
    for split, g in D[D.alpha == 1.0].groupby("split"):
        print(f"  {split:<7} sil {g.sil_true.median():+.3f}")
    print(f"\nwrote b5_splits.csv, b5_splits_summary.csv   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
