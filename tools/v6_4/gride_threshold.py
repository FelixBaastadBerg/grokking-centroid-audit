"""
v6.4 Phase 2.6 -- corrected GRIDE forced-leakage threshold, on a dense order grid.

The v6.3 text says a neighbourhood of 2*n1 points "must leave the class once n1 > m/2". That is
off by the query point and by rounding. If a class holds m points INCLUDING the query, the query
has at most m-1 same-class neighbours, and GRIDE at order n1 reaches through its 2*n1-th
neighbour. Leakage is therefore forced when

    2*n1 > m - 1,      i.e. the smallest forced order is  n1 = ceil(m/2).

For even m this agrees with the old statement; for odd m it does not, and real clouds have
unequal class sizes, so there is no single global m at all. This script:

  1. computes the class-specific forced order ceil(m_c/2) for every class of every real cloud,
     and reports the distribution rather than one number;
  2. measures the ACTUAL leakage L_{2 n1} against n1, which can begin earlier than the forced
     order because of geometry, and never later;
  3. reruns the synthetic crossover on a DENSE order grid (n1 = 1..24) instead of powers of two,
     so the measured crossover is not quantised to the grid;
  4. reruns the real-activation sweep on the same dense grid, for both the 400-point training
     subsample of v6.3 and the complete input grid of v6.4.

The experiment validates the support interpretation. It is not a free-standing theorem: the
forced order is a counting bound, and where the response actually switches on is measured.

Analysis-only. Writes RESULTS_v6_4/.
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
from tools.b1_gride import cloud, D_TRUE
from tools.e1_centroid import decompose, recompose
from tools.b3_seed_alpha import select
from tools.v6_4.fullgrid import full_grid_inputs, acts

OUT = REPO / "RESULTS_v6_4"
OUT.mkdir(exist_ok=True)

N1S = list(range(1, 25))          # dense, not powers of two
SEEDS = (0, 1, 2)


def forced_order(class_sizes):
    """Smallest GRIDE order at which same-class neighbours run out, per class."""
    return {int(c): int(np.ceil(m / 2)) for c, m in class_sizes.items()}


def leak_curve(X, y, n1s):
    """Measured fraction of points whose 2*n1 nearest neighbours are not all same-class."""
    q = 2 * max(n1s)
    nn = NearestNeighbors(n_neighbors=min(q + 1, len(X))).fit(X)
    _, idx = nn.kneighbors(X)
    y = np.asarray(y)
    lab = y[idx[:, 1:]]
    same = lab == y[:, None]
    out = {}
    for n1 in n1s:
        k = 2 * n1
        out[n1] = float(np.mean(~same[:, :k].all(1))) if k <= same.shape[1] else np.nan
    return out


# --------------------------------------------------------------------------- synthetic arm
def synthetic():
    print("=== synthetic: dense-order GRIDE crossover against the corrected forced order ===")
    rows = []
    for m in (8, 16, 32, 64):
        for sigma, lab in ((0.05, "clustered"), (1.5, "unclustered")):
            for s in SEEDS:
                X, y = cloud(D_TRUE, 8, m, sigma, seed=s)
                lk = leak_curve(X, y, [n for n in N1S if 2 * n < len(X)])
                for n1 in N1S:
                    if 2 * n1 >= len(X):
                        continue
                    rows.append({"m": m, "sigma": sigma, "regime": lab, "seed": s, "n1": n1,
                                 "gride": geom.gride(X, n1), "leak_2n1": lk.get(n1, np.nan),
                                 "forced_n1": int(np.ceil(m / 2)),
                                 "old_rule_n1": m / 2.0})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "gride_synth_dense.csv", index=False)

    w = S.pivot_table(index=["m", "seed", "n1", "forced_n1"], columns="regime",
                      values="gride").reset_index()
    w["ratio"] = w["clustered"] / w["unclustered"]
    lk = S[S.regime == "clustered"].groupby(["m", "seed", "n1"]).leak_2n1.first().reset_index()
    w = w.merge(lk, on=["m", "seed", "n1"])
    w.to_csv(OUT / "gride_synth_ratio_dense.csv", index=False)

    out = []
    for (m, s), g in w.groupby(["m", "seed"]):
        g = g.sort_values("n1")
        base = g[g.n1 <= 2].ratio.mean()
        below = g[g.ratio <= 0.5 * base]
        first_leak = g[g.leak_2n1 > 0]
        out.append({"m": m, "seed": s, "base_ratio": base,
                    "n1_cross_measured": float(below.n1.min()) if len(below) else np.nan,
                    "n1_first_leak_measured": (float(first_leak.n1.min())
                                               if len(first_leak) else np.nan),
                    "n1_forced_corrected": int(np.ceil(m / 2)),
                    "n1_old_rule": m / 2.0})
    C = pd.DataFrame(out)
    C.to_csv(OUT / "gride_synth_crossover_dense.csv", index=False)
    print(C.groupby("m").agg(cross=("n1_cross_measured", "median"),
                             first_leak=("n1_first_leak_measured", "median"),
                             forced=("n1_forced_corrected", "first"),
                             old=("n1_old_rule", "first")).to_string())
    return S, C


# --------------------------------------------------------------------------- real arm
def real():
    print("\n=== real activations: class sizes, forced orders, and measured crossover ===")
    inv = pd.read_csv(REPO / "RESULTS" / "_inventory.csv")
    sel = select(inv)
    rows, sizes = [], []
    for _, rec in sel.iterrows():
        r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
        seed = int(rec["seed"])

        # population A: the v6.3 400-point training subsample
        y_all = r.Ytr.numpy()
        sub = np.sort(np.random.default_rng(0).choice(len(y_all), min(400, len(y_all)),
                                                      replace=False))
        r._load(r.flat(len(r.epochs) - 1)); r.model.eval()
        with torch.no_grad():
            r.model(r.Xtr, keep_activations=True)
            HA = r.model.activations[-2].numpy()[sub]
        yA = y_all[sub]

        # population B: the complete input grid
        X, yB = full_grid_inputs(r.meta)
        HB = acts(r, len(r.epochs) - 1, X)

        for pop, H, y in (("train400", HA, yA), ("fullgrid", HB, yB)):
            cs = pd.Series(y).value_counts().to_dict()
            fo = forced_order(cs)
            sizes.append({"seed": seed, "population": pop, "n_points": len(y),
                          "n_classes": len(cs),
                          "m_min": min(cs.values()), "m_median": float(np.median(list(cs.values()))),
                          "m_max": max(cs.values()),
                          "m_mean": len(y) / len(cs),
                          "forced_n1_min": min(fo.values()),
                          "forced_n1_median": float(np.median(list(fo.values()))),
                          "forced_n1_max": max(fo.values()),
                          "old_rule_n1": len(y) / len(cs) / 2.0})
            mu, bet, wit = decompose(H, y)
            for a in (0.0, 8.0):
                Z = recompose(mu, bet, wit, a, 1.0, norm="trace")
                lk = leak_curve(Z, y, N1S)
                for n1 in N1S:
                    rows.append({"seed": seed, "population": pop, "alpha": a, "n1": n1,
                                 "gride": geom.gride(Z, n1), "leak_2n1": lk[n1],
                                 "forced_n1_min": min(fo.values()),
                                 "forced_n1_median": float(np.median(list(fo.values())))})
        print(f"  seed {seed} done", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "gride_real_dense.csv", index=False)
    Z = pd.DataFrame(sizes)
    Z.to_csv(OUT / "gride_real_class_sizes.csv", index=False)

    w = R.pivot_table(index=["seed", "population", "n1", "forced_n1_min", "forced_n1_median"],
                      columns="alpha", values="gride").reset_index()
    w["ratio"] = w[8.0] / w[0.0]
    lk = R[R.alpha == 8.0].groupby(["seed", "population", "n1"]).leak_2n1.first().reset_index()
    w = w.merge(lk, on=["seed", "population", "n1"])
    w.to_csv(OUT / "gride_real_ratio_dense.csv", index=False)

    out = []
    for (s, pop), g in w.groupby(["seed", "population"]):
        g = g.sort_values("n1")
        base = g[g.n1 <= 2].ratio.mean()
        below = g[g.ratio <= 0.5 * base]
        fl = g[g.leak_2n1 > 0]
        out.append({"seed": s, "population": pop, "base_ratio": base,
                    "n1_cross_measured": float(below.n1.min()) if len(below) else np.nan,
                    "n1_first_leak_measured": float(fl.n1.min()) if len(fl) else np.nan,
                    "forced_n1_min": g.forced_n1_min.iloc[0],
                    "forced_n1_median": g.forced_n1_median.iloc[0]})
    C = pd.DataFrame(out)
    C.to_csv(OUT / "gride_real_crossover_dense.csv", index=False)
    print("\n  per-population summary (median over the 9 primary seeds):")
    print(C.groupby("population").agg(
        cross=("n1_cross_measured", "median"),
        first_leak=("n1_first_leak_measured", "median"),
        forced_min=("forced_n1_min", "median"),
        forced_median=("forced_n1_median", "median")).to_string())
    print("\n  class-size summary:")
    print(Z.groupby("population").agg(
        n=("n_points", "median"), K=("n_classes", "median"),
        m_min=("m_min", "median"), m_med=("m_median", "median"), m_max=("m_max", "median"),
        forced_min=("forced_n1_min", "median"),
        forced_med=("forced_n1_median", "median")).to_string())
    return R, C, Z


def main():
    t0 = time.time()
    synthetic()
    real()
    print(f"\ndone ({time.time()-t0:.0f}s); wrote RESULTS_v6_4/gride_*.csv")


if __name__ == "__main__":
    main()
