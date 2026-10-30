"""
P1-4, matched operating point. Companion to p1_4_scale.py, run in parallel.

Why this exists. p1_4_scale.py measures the alpha sweep on a 400-point subsample, which is what the
corpus cells use. At p = 23 that is ~17 points per class, close to the real operating point (median
20). At p = 97 the same 400 points leave ~4 per class -- but the trained network's activation cloud
actually holds p^2*tf/p = 58 points per class, MORE than at p = 23. So a fixed 400-point cap does
not hold the operating point fixed as p grows; it drives points-per-cluster m down, and by our own
scale-locality account m is the variable that decides whether an estimator responds. Reading a
larger fold at p = 97 off that measurement would be reading our own subsampling.

This arm holds m fixed instead. It draws a random subset of CLASSES and a fixed number of points
per class, so the cloud has the corpus operating point's m at every p, and only the number of
classes present in the cloud differs. Both measurements are reported: together they separate
"the confound changes with p" from "the confound changes with m".

Final activations are saved to RESULTS/p1_4_acts/ so no further question about these networks
needs a retrain.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from sklearn.metrics import silhouette_score

from analysis import geom
from tools.linchpin import h1_stats
from tools.e1_centroid import decompose, recompose, ALPHAS
from tools.p1_4_scale import train

OUT = REPO / "RESULTS"
ACTS = OUT / "p1_4_acts"
ACTS.mkdir(parents=True, exist_ok=True)

PER_CLASS = 20        # the corpus operating point: median 20 points per class at convergence
N_CLASSES = 20        # so the cloud is 400 points, the same size as every other sweep


def matched_subset(H, y, per_class=PER_CLASS, n_classes=N_CLASSES, seed=0):
    """Random classes, fixed points per class: holds m at the real operating point."""
    g = np.random.default_rng(seed)
    usable = [c for c in np.unique(y) if (y == c).sum() >= per_class]
    if len(usable) < 2:
        return None, None
    take = g.choice(usable, min(n_classes, len(usable)), replace=False)
    idx = np.concatenate([g.choice(np.flatnonzero(y == c), per_class, replace=False) for c in take])
    return H[idx], y[idx]


def sweep(Z_src, ys, tag, task, p, seed, m):
    mu, bet, wit = decompose(Z_src, ys)
    rows = []
    for a in ALPHAS:
        Z = recompose(mu, bet, wit, a, 1.0)
        pdim, _ = geom.ph_dim(Z); cd, _ = geom.correlation_dim(Z)
        h = h1_stats(Z)
        rows.append({"arch": "mlp", "task": task, "p": p, "seed": seed, "norm": "trace",
                     "sweep": "alpha", "alpha": a, "beta": 1.0, "ph_dim": pdim, "corr_dim": cd,
                     "twonn": geom.twonn(Z), "mle_k10": geom.mle_id(Z, 10),
                     "pca_pr": geom.pca_participation_ratio(Z),
                     "eff_rank": geom.effective_rank(Z), "H1_max": h["H1_max"],
                     "H1_total": h["H1_total"], "points_per_class": m,
                     "subsample": tag, "source": "P1-4"})
    return rows


if __name__ == "__main__":
    t0 = time.time()
    e1_rows, op_rows = [], []
    for task, p in [("add", 97), ("add", 113)]:
        for seed in range(3):
            try:
                curve, snaps = train(task, p, seed, max_points=4000)
            except TypeError:
                curve, snaps = train(task, p, seed)
            except Exception as e:
                print(f"  FAIL {task} p={p} seed{seed}: {type(e).__name__} {e}", flush=True)
                continue
            H, ys = snaps["final"]
            np.savez_compressed(ACTS / f"{task}_p{p}_seed{seed}.npz", H=H, y=ys)

            Hm, ym = matched_subset(H, ys, seed=seed)
            if Hm is None:
                print(f"  {task} p={p} seed{seed}: too few points per class to match", flush=True)
                continue
            m = PER_CLASS
            fin = curve.test_acc.values[-3:].mean()
            op_rows.append({"arch": "mlp", "task": task, "p": p, "seed": seed, "tf": 0.6,
                            "sil_init": curve.silhouette.iloc[0],
                            "sil_final": float(silhouette_score(Hm, ym)),
                            "grokked": bool(fin >= 0.9), "final_test": fin,
                            "points_per_class": m, "subsample": "matched", "source": "P1-4"})
            e1_rows += sweep(Hm, ym, "matched", task, p, seed, m)
            print(f"  {task}/p{p}/seed{seed}: test={fin:.3f}  matched-m sil="
                  f"{op_rows[-1]['sil_final']:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(op_rows).to_csv(OUT / "p1_4_operating_matched.csv", index=False)
            pd.DataFrame(e1_rows).to_csv(OUT / "p1_4_centroid_matched.csv", index=False)
    print(f"\nwrote p1_4_*_matched.csv ({time.time()-t0:.0f}s)")
