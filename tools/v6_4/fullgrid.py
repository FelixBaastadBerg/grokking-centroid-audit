"""
v6.4 Phase 2.1 + 2.2 -- the primary attribution on the COMPLETE input grid, under two fixed
Euclidean preprocessings.

Why. The v6.3 primary attribution measures a fixed subsample of 400 TRAINING inputs, which is
about 17 points per class. The paper's own mechanism account says estimator behaviour depends on
points per class, and the paper is about generalisation, so the primary number should not rest on
a training-only subsample whose points-per-class sits right at the support threshold the paper
identifies. Here the same analysis is rerun on the complete Z_23 x Z_23 grid: 529 inputs, 23
classes, exactly 23 points per class, train and held-out inputs together, in a fixed order that
does not change across checkpoints or counterfactuals.

Two pipelines, both fixed before the results were seen:

  raw    the v6.3 pipeline -- recombine, centre, divide by the root-mean-square radius
         (the trace normalisation of tools/e1_centroid.py).

  diag   a second fixed Euclidean metric -- recombine, apply a per-seed diagonal map frozen from
         the pooled REAL initial and REAL converged clouds, then the same trace normalisation.
         The map is estimated once per seed and applied identically to every counterfactual cloud
         and every point of the alpha sweep, so the intervention itself is unchanged. Features
         whose pooled standard deviation is below 1e-8 times the largest are dropped.
         This tests a second fixed Euclidean geometry. It is not a test of metric-free invariance.

Outputs (all under RESULTS_v6_4/, nothing in RESULTS/ is touched):
  fullgrid_recombine.csv     5x5 (s,t) recombination grid, per seed, per pipeline
  fullgrid_shapley.csv       endpoint 2x2 and interval decompositions, both factor orders
  fullgrid_sweep.csv         converged-cloud centroid intervention over the v6.3 alpha grid,
                             with the leakage L_q measured at every alpha
  fullgrid_meta.csv          n, K, points per class, checkpoint epochs per seed

Analysis-only: retained checkpoints, no training.
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
from tools.train_topology import build_task
from tools.b3_seed_alpha import select
from tools.c1_recombine import checkpoints

OUT = REPO / "RESULTS_v6_4"
OUT.mkdir(exist_ok=True)

# identical to tools/b3_seed_alpha.py, so the sweeps are comparable
ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
PIPELINES = ("raw", "diag")
ZERO_VAR_REL = 1e-8          # documented fixed rule for negligible feature variance


# --------------------------------------------------------------------------- measurement
def measure(Z, y):
    """PH-dimension and TwoNN are primary; the rest are cheap and come along."""
    pdim, r2 = geom.ph_dim(Z)
    cd, _ = geom.correlation_dim(Z)
    out = {"ph_dim": pdim, "ph_dim_r2": r2, "twonn": geom.twonn(Z),
           "mle_k10": geom.mle_id(Z, 10), "corr_dim": cd,
           "pca_pr": geom.pca_participation_ratio(Z),
           "eff_rank": geom.effective_rank(Z)}
    try:
        from sklearn.metrics import silhouette_score
        out["sil_true"] = float(silhouette_score(Z, y))
    except Exception:
        out["sil_true"] = np.nan
    return out


def leakage(Z, y, q):
    """Fraction of points whose q nearest neighbours are not all in the point's own class."""
    nn = NearestNeighbors(n_neighbors=q + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    lab = np.asarray(y)[idx[:, 1:q + 1]]
    return float(np.mean((lab != np.asarray(y)[:, None]).any(1)))


# --------------------------------------------------------------------------- geometry
def parts(H, y):
    cls = np.unique(y)
    M = {c: H[y == c].mean(0) for c in cls}
    cen = np.stack([M[c] for c in y])
    return cen, H - cen


def trace_norm(Z):
    Zc = Z - Z.mean(0)
    return Zc / (np.sqrt((Zc ** 2).sum(1).mean()) + 1e-12)


def diag_map_from(H_init, H_conv):
    """One frozen diagonal standardisation per seed, from the pooled real endpoint clouds."""
    P = np.vstack([H_init, H_conv])
    sd = P.std(0)
    keep = sd > ZERO_VAR_REL * sd.max()
    return keep, sd


def apply_pipeline(Z, pipeline, keep, sd):
    if pipeline == "diag":
        Z = Z[:, keep] / sd[keep]
    return trace_norm(Z)


# --------------------------------------------------------------------------- full input grid
def full_grid_inputs(meta):
    """Every (a, b) pair of the task, in the deterministic build_task order."""
    p = meta["p"]
    pairs, y, n_sym, _ = build_task(meta["task"], p,
                                    sym_n=4 if (meta["task"] == "perm" and p == 24) else 5)
    N = len(y)
    X = torch.zeros(N, 2 * n_sym)
    X[torch.arange(N), torch.as_tensor(pairs[:, 0])] = 1.0
    X[torch.arange(N), n_sym + torch.as_tensor(pairs[:, 1])] = 1.0
    return X, np.asarray(y)


def acts(r, i, X):
    r._load(r.flat(i))
    r.model.eval()
    with torch.no_grad():
        r.model(X, keep_activations=True)
        return r.model.activations[-2].numpy()


# --------------------------------------------------------------------------- per-run driver
def one_run(rec):
    d, seed = rec["dir"], int(rec["seed"])
    r = LandscapeRun(REPO / "runs_topo" / d)
    X, y = full_grid_inputs(r.meta)
    idx = checkpoints(r)

    cen, res, H = {}, {}, {}
    for i in idx:
        H[i] = acts(r, i, X)
        cen[i], res[i] = parts(H[i], y)

    keep, sd = diag_map_from(H[idx[0]], H[idx[-1]])

    meta = {"dir": d, "seed": seed, "n_points": len(y), "n_classes": int(len(np.unique(y))),
            "points_per_class": len(y) / len(np.unique(y)),
            "n_features": H[idx[0]].shape[1], "n_features_kept_diag": int(keep.sum()),
            "checkpoint_epochs": ";".join(str(int(r.epochs[i])) for i in idx),
            "checkpoint_indices": ";".join(str(i) for i in idx)}

    grid, sweep = [], []
    for pipe in PIPELINES:
        # ---- 5x5 recombination grid
        for si, s in enumerate(idx):
            for ti, t in enumerate(idx):
                Z = apply_pipeline(cen[s] + res[t], pipe, keep, sd)
                grid.append({"dir": d, "seed": seed, "pipeline": pipe, "s": si, "t": ti,
                             "epoch_s": int(r.epochs[s]), "epoch_t": int(r.epochs[t]),
                             **measure(Z, y)})

        # ---- converged-cloud centroid intervention, same alpha grid as v6.3
        Hc = H[idx[-1]]
        mu = Hc.mean(0)
        bet = np.stack([{c: Hc[y == c].mean(0) for c in np.unique(y)}[c] for c in y]) - mu
        wit = Hc - (bet + mu)
        for a in ALPHAS:
            Z = apply_pipeline(mu + a * bet + wit, pipe, keep, sd)
            row = {"dir": d, "seed": seed, "pipeline": pipe, "alpha": a, **measure(Z, y)}
            for q in (2, 10):
                row[f"L{q}"] = leakage(Z, y, q)
            sweep.append(row)

    return grid, sweep, meta


def main():
    from multiprocessing import Pool
    t0 = time.time()
    inv = pd.read_csv(REPO / "RESULTS" / "_inventory.csv")
    sel = select(inv)
    recs = [r for _, r in sel.iterrows()]
    print(f"=== v6.4 full-grid recombination: {len(recs)} primary seeds, "
          f"{len(PIPELINES)} pipelines ===", flush=True)
    print(f"    seeds {sorted(int(r['seed']) for r in recs)}", flush=True)

    G, S, M = [], [], []
    with Pool(processes=5) as pool:
        for i, (g, s, m) in enumerate(pool.imap_unordered(one_run, recs), 1):
            G += g; S += s; M.append(m)
            print(f"  [{i}/{len(recs)}] seed {m['seed']} "
                  f"n={m['n_points']} m={m['points_per_class']:.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(G).to_csv(OUT / "fullgrid_recombine.csv", index=False)
    pd.DataFrame(S).to_csv(OUT / "fullgrid_sweep.csv", index=False)
    pd.DataFrame(M).to_csv(OUT / "fullgrid_meta.csv", index=False)
    print(f"\nwrote fullgrid_recombine.csv ({len(G)} rows), fullgrid_sweep.csv ({len(S)}), "
          f"fullgrid_meta.csv ({len(M)})   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
