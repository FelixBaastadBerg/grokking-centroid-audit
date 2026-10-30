"""
DIR A experiment 4 -- Betti numbers of the loss sublevel sets themselves.

Mode connectivity asks about a path between two chosen solutions. This asks about the shape
of the low-loss set near a checkpoint, independent of any endpoint: take a random 2D plane
through W_t, evaluate the train loss on a grid, and read off b0 and b1 of {loss <= level}
as the level sweeps. Both are computed with the estimator that passed the analytic gate in
dirA_validate.py (exact saddle heights, exact ring-valley b1).

Levels are set per checkpoint as quantiles of the observed loss on that slice, so a slice
whose losses are all large is not automatically declared empty -- the topology is compared
at matched *rank*, not at matched absolute loss.

Null: the same measurement on a plane through a norm-matched random weight vector, which
gives the b0/b1 profile of a generic slice at that scale.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from analysis.lsrun import LandscapeRun
from analysis.landscape import betti_sublevel

OUT = REPO / "RESULTS"
QS = np.array([0.05, 0.10, 0.20, 0.35, 0.50])


def slice_grid(r: LandscapeRun, w: np.ndarray, radius: float, n=31, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=w.shape); u /= np.linalg.norm(u)
    v = rng.normal(size=w.shape); v -= (v @ u) * u; v /= np.linalg.norm(v)
    ax = np.linspace(-radius, radius, n)
    F = np.empty((n, n))
    for i, b in enumerate(ax):
        for j, a in enumerate(ax):
            F[i, j] = r.evaluate(w + a * u + b * v)["train_loss"]
    return F


def profile(F: np.ndarray) -> dict:
    lv = np.quantile(F, QS)
    b0, b1 = betti_sublevel(F, lv)
    return {**{f"b0_q{int(q*100)}": int(x) for q, x in zip(QS, b0)},
            **{f"b1_q{int(q*100)}": int(x) for q, x in zip(QS, b1)},
            "loss_min": float(F.min()), "loss_med": float(np.median(F))}


if __name__ == "__main__":
    inv = pd.read_csv(OUT / "_inventory.csv")
    G = inv[(inv.arch == "mlp") & (inv.test >= 0.9)].drop_duplicates(["task", "p", "tf"])
    G = G[G.tf.isin([0.7, 0.8])].head(6)
    rows = []
    for _, rec in G.iterrows():
        r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
        tg, n = r.t_grok(), len(r.epochs)
        if tg < 0:
            continue
        lg = np.unique(np.round(np.geomspace(1, n - 1, 14)).astype(int))
        ts = np.unique(np.concatenate([[0], lg, np.arange(max(0, tg - 3), min(n, tg + 4)), [n - 1]]))
        # a length scale for the slice: typical checkpoint-to-checkpoint motion late in the run
        radius = float(np.linalg.norm(r.flat(n - 1) - r.flat(n // 2))) * 0.25
        for t in ts:
            w = r.flat(int(t))
            p = profile(slice_grid(r, w, radius))
            p.update({"t": int(t), "epoch": int(r.epochs[int(t)]), "t_grok": tg,
                      "test_acc": float(r.curves.test_acc.values[int(t)]),
                      "task": rec["task"], "p": rec["p"], "tf": rec["tf"], "kind": "real"})
            rows.append(p)
        rng = np.random.default_rng(0)
        for k in range(3):
            wr = rng.normal(size=r.flat(0).shape); wr *= np.linalg.norm(r.flat(n - 1)) / np.linalg.norm(wr)
            p = profile(slice_grid(r, wr, radius, seed=100 + k))
            p.update({"t": -1, "epoch": -1, "t_grok": tg, "test_acc": np.nan,
                      "task": rec["task"], "p": rec["p"], "tf": rec["tf"], "kind": "null"})
            rows.append(p)
        print(f"  done {rec['task']}/p{rec['p']}/tf{rec['tf']}  ({len(ts)} checkpoints)", flush=True)
    D = pd.DataFrame(rows); D.to_csv(OUT / "dirA_slices.csv", index=False)
    D["side"] = np.where(D.kind == "null", "null",
                         np.where(D.t < D.t_grok, "pre-grok",
                                  np.where(D.t > D.t_grok, "post-grok", "at-grok")))
    cols = [c for c in D.columns if c.startswith(("b0_", "b1_"))]
    print("\n=== Betti profile of train-loss sublevel sets on random 2D slices ===")
    print(D.groupby("side")[cols].median().round(2).to_string())
    print("\n  per task:")
    print(D.groupby(["task", "p", "side"])[["b0_q5", "b0_q20", "b1_q20", "b1_q50"]].median().round(2).to_string())
