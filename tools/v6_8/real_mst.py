"""
v6.8 -- test the two-term MST account on the REAL alpha-sweep clouds, not only on synthetic
mixtures.

The mechanism section derives

    E_eta(n) ~ A n^{1-eta/d} + B,     d_hat ~ eta (1+u) / (u + eta/d),   u = E_between/E_within,

and validates it on 288 synthetic mixtures with u measured from the same mixtures. That is a
conditional explanation. This script asks whether the same decomposition describes the real
converged activations of the nine primary seeds as the centroid scale alpha is swept.

At every swept alpha and every subsample size in PH-dimension's own frozen schedule, it records:

  * total MST edge energy, and its split into within-class and cross-class edges;
  * the cross-class edge COUNT, against the K-1 the separated-mixture assumption predicts;
  * the fitted PH-dimension (recomputed here so the fit and the energies share subsamples);
  * u at the window midpoint and the closed-form d_hat implied by it;
  * the residual between measured and closed-form d_hat.

The estimator's subsample stream is seeded exactly as analysis/geom.ph_dim seeds it, so the
energies are measured on the SAME subsamples the fit uses.

Output: RESULTS_v6_8/real_mst_sweep.csv (per seed x alpha x subsample size)
        RESULTS_v6_8/real_mst_summary.csv (per alpha, pooled over seeds)

Analysis-only: retained checkpoints, no training.
"""
from __future__ import annotations
import os
import sys
import time
import warnings
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
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from analysis.lsrun import LandscapeRun
from tools.b3_seed_alpha import select, ALPHAS
from tools.train_topology import build_task

OUT = REPO / "RESULTS_v6_8"
OUT.mkdir(exist_ok=True)

ETA = 1.0
N_REPEATS = 5        # identical to analysis/geom.ph_dim
PH_SEED = 0          # identical to analysis/geom.ph_dim


def schedule(n):
    """PH-dimension's own subsample sizes for a cloud of n points."""
    hi = min(n, 1000)
    lo = max(50, hi // 5)
    return [m for m in range(lo, hi + 1, max(1, (hi - lo) // 10)) if 5 <= m <= n]


def full_grid_inputs(meta):
    p = meta["p"]
    pairs, y, n_sym, _ = build_task(meta["task"], p, sym_n=5)
    N = len(y)
    X = torch.zeros(N, 2 * n_sym)
    X[torch.arange(N), torch.as_tensor(pairs[:, 0])] = 1.0
    X[torch.arange(N), n_sym + torch.as_tensor(pairs[:, 1])] = 1.0
    return X, np.asarray(y)


def parts(H, y):
    cen = np.zeros_like(H)
    for c in np.unique(y):
        cen[y == c] = H[y == c].mean(0)
    return cen, H - cen


def trace_norm(Z):
    Zc = Z - Z.mean(0)
    return Zc / (np.sqrt((Zc ** 2).sum(1).mean()) + 1e-12)


def mst_split(X, y):
    """Total, within-class and cross-class MST edge energy, and the cross-edge count."""
    D = squareform(pdist(X))
    m = minimum_spanning_tree(D).tocoo()
    w = m.data ** ETA
    same = y[m.row] == y[m.col]
    return float(w.sum()), float(w[same].sum()), float(w[~same].sum()), int((~same).sum())


def sweep_one(H0, y, alphas):
    """For one converged cloud, every alpha and every subsample size."""
    mu = H0.mean(0)
    cen, res = parts(H0, y)
    rows = []
    for a in alphas:
        Z = trace_norm(mu + a * (cen - mu) + res)
        rng = np.random.default_rng(PH_SEED)     # same stream the estimator uses
        xs, ys = [], []
        for n in schedule(len(Z)):
            tot = wit = bet = 0.0
            nx = 0
            for _ in range(N_REPEATS):
                idx = rng.choice(len(Z), n, replace=False)
                t, w_, b_, c_ = mst_split(Z[idx], y[idx])
                tot += t; wit += w_; bet += b_; nx += c_
            k = N_REPEATS
            rows.append({"alpha": a, "n": n, "E_total": tot / k, "E_within": wit / k,
                         "E_between": bet / k, "cross_edges": nx / k,
                         "K_minus_1": len(np.unique(y)) - 1})
            xs.append(np.log(n)); ys.append(np.log(tot / k))
        A = np.vstack([xs, np.ones_like(xs)]).T
        slope, _ = np.linalg.lstsq(A, np.array(ys), rcond=None)[0]
        d_hat = np.nan if slope >= 1 - 1e-6 else ETA / (1 - slope)
        for r in rows:
            if r["alpha"] == a:
                r["d_hat_fit"] = d_hat
    return rows


def main():
    inv = pd.read_csv(REPO / "RESULTS" / "_inventory.csv")
    sel = select(inv)
    rows, t0 = [], time.time()
    for i, (_, rec) in enumerate(sel.iterrows()):
        d = REPO / "runs_topo" / rec["dir"]
        try:
            r = LandscapeRun(d)
            r._load(r.flat(len(r.epochs) - 1)); r.model.eval()
            import json
            meta = json.loads((d / "run_meta.json").read_text())
            X, y = full_grid_inputs(meta)
            with torch.no_grad():
                r.model(X, keep_activations=True)
                H = r.model.activations[-2].numpy()
        except Exception as e:
            print(f"  skip {rec['dir'][:48]}: {type(e).__name__} {e}", flush=True)
            continue
        for row in sweep_one(H, y, ALPHAS):
            rows.append({"dir": rec["dir"], "seed": int(rec["seed"]), **row})
        print(f"  [{i+1}/{len(sel)}] seed {int(rec['seed']):>2} ({time.time()-t0:.0f}s)",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT / "real_mst_sweep.csv", index=False)

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "real_mst_sweep.csv", index=False)

    # u at the window midpoint, and the closed form it implies
    summ = []
    for (seed, a), g in D.groupby(["seed", "alpha"]):
        g = g.sort_values("n")
        mid = g.iloc[len(g) // 2]
        u = mid.E_between / max(mid.E_within, 1e-12)
        summ.append({"seed": seed, "alpha": a, "u_mid": u,
                     "d_hat_fit": g.d_hat_fit.iloc[0],
                     "cross_edges_mid": mid.cross_edges,
                     "K_minus_1": mid.K_minus_1,
                     "cross_over_Km1": mid.cross_edges / max(mid.K_minus_1, 1)})
    S = pd.DataFrame(summ)
    # the closed form needs a within-class d; take it from the least-clustered swept state,
    # where u is smallest and d_hat is closest to the within-class dimension
    base = S[S.alpha == min(S.alpha)].set_index("seed").d_hat_fit
    S["d_within_ref"] = S.seed.map(base)
    S["d_closed"] = ETA * (1 + S.u_mid) / (S.u_mid + ETA / S.d_within_ref)
    S["resid"] = S.d_hat_fit - S.d_closed
    S.to_csv(OUT / "real_mst_summary.csv", index=False)

    print("\n=== real converged clouds: does the separated-mixture account hold? ===")
    print(f"  {'alpha':>6} {'u':>8} {'cross/K-1':>10} {'d_hat':>8} {'closed':>8} {'resid':>8}")
    for a, g in S.groupby("alpha"):
        print(f"  {a:>6g} {g.u_mid.median():>8.3f} {g.cross_over_Km1.median():>10.2f} "
              f"{g.d_hat_fit.median():>8.2f} {g.d_closed.median():>8.2f} "
              f"{g.resid.median():>+8.2f}")
    con = S[S.alpha >= 1.5]
    print(f"\n  class-contained region (alpha >= 1.5), pooled over {con.seed.nunique()} seeds:")
    print(f"    cross-edge count / (K-1)  median {con.cross_over_Km1.median():.2f}  "
          f"[{con.cross_over_Km1.min():.2f}, {con.cross_over_Km1.max():.2f}]")
    print(f"    residual d_hat - closed   median {con.resid.median():+.3f}  "
          f"[{con.resid.min():+.3f}, {con.resid.max():+.3f}]")
    print("\nwrote RESULTS_v6_8/real_mst_sweep.csv, real_mst_summary.csv")


if __name__ == "__main__":
    main()
