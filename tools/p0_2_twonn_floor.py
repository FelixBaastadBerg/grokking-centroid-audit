"""
P0-2 -- TwoNN's residual clustering-sensitivity, and a detection floor.

The recommendation rests on TwoNN being the least clustering-sensitive estimator, but it still
moves 1.2-1.4x under pure clustering, while the real dimensional signal we report is only -13%.
Those two numbers are uncomfortably close, so this quantifies the floor explicitly.

Definition used, stated so it can be argued with:

    At the operating point real networks occupy (points-per-class m taken from real activation
    clouds), sweep separation-to-scale gamma at FIXED true d and record TwoNN's spread -- that is
    the movement clustering alone can manufacture. Separately, at the same m and a mid-range
    gamma, measure TwoNN's sensitivity to a real change in d (the local slope dTwoNN/dd).

        detection floor  =  clustering-induced TwoNN spread  /  |dTwoNN / dd|

    i.e. the smallest true dimension change whose TwoNN signature exceeds what clustering alone
    can produce. Bootstrap CI over seeds.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from analysis import geom
from analysis.lsrun import LandscapeRun
from tools.intervention import make_cloud, clusteredness

OUT = REPO / "RESULTS"
MS = [4, 8, 16, 32, 64, 128]          # points per cluster
GAMMAS = [0.02, 0.05, 0.1, 0.25, 0.6, 1.5]   # separation-to-scale, via sigma at fixed R
SEEDS = range(5)


def cloud(d, m, sigma, K=8, amb=60, seed=0):
    return make_cloud(d, K, sigma, n=m * K, seed=seed)


def real_points_per_class():
    """Median points-per-class in real activation clouds at convergence."""
    inv = pd.read_csv(OUT / "_inventory.csv")
    sel = inv[(inv.arch == "mlp") & (inv.test >= 0.9)].drop_duplicates(["task", "p", "tf"])
    vals = []
    for _, rec in sel.iterrows():
        try:
            r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
        except Exception:
            continue
        y = r.Ytr.numpy()
        _, cnt = np.unique(y, return_counts=True)
        vals.append(float(np.median(cnt)))
    return float(np.median(vals)), len(vals)


if __name__ == "__main__":
    t0 = time.time()
    pd.set_option("display.width", 220)

    m_real, n_runs = real_points_per_class()
    print(f"=== operating point from real activation clouds ===")
    print(f"    median points per class = {m_real:.0f}  (over {n_runs} grokked MLP runs)\n")

    print("=== P0-2a: TwoNN under pure clustering, true d fixed at 5 ===")
    rows = []
    for m in MS:
        for gsig in GAMMAS:
            for seed in SEEDS:
                X = cloud(5, m, gsig, seed=seed)
                rows.append({"m": m, "sigma": gsig, "seed": seed,
                             "silhouette": clusteredness(X), "twonn": geom.twonn(X),
                             "ph_dim": geom.ph_dim(X)[0]})
        print(f"  m={m} done ({time.time()-t0:.0f}s)", flush=True)
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "twonn_floor.csv", index=False)
    piv = A.groupby(["m", "sigma"]).twonn.mean().unstack()
    print("\n  TwoNN by points-per-cluster (rows) and sigma (cols), true d = 5:")
    print(piv.round(2).to_string())
    fold = A.groupby("m").apply(lambda g: g.groupby("sigma").twonn.mean().max()
                                / max(g.groupby("sigma").twonn.mean().min(), 1e-9))
    print("\n  TwoNN fold change over the clustering sweep, by points-per-cluster:")
    print(fold.round(3).to_string())

    print("\n\n=== P0-2b: TwoNN sensitivity to a REAL dimension change at the same m ===")
    m_use = min(MS, key=lambda x: abs(x - m_real))
    print(f"    using m = {m_use} (nearest grid point to the real {m_real:.0f})\n")
    rows = []
    for d in (2, 3, 4, 5, 6, 8):
        for seed in SEEDS:
            X = cloud(d, m_use, 0.25, seed=seed)
            rows.append({"true_d": d, "seed": seed, "twonn": geom.twonn(X)})
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "twonn_floor_sensitivity.csv", index=False)
    print(B.groupby("true_d").twonn.agg(["mean", "std"]).round(3).to_string())

    print("\n\n=== the detection floor ===")
    g = np.random.default_rng(0)
    floors, spreads, slopes = [], [], []
    for _ in range(2000):
        take = g.choice(list(SEEDS), len(list(SEEDS)), replace=True)
        a = A[(A.m == m_use) & (A.seed.isin(take))]
        mm = a.groupby("sigma").twonn.mean()
        spread = mm.max() - mm.min()
        b = B[B.seed.isin(take)]
        bb = b.groupby("true_d").twonn.mean()
        slope = np.polyfit(bb.index.values.astype(float), bb.values, 1)[0]
        if abs(slope) > 1e-9:
            floors.append(spread / abs(slope)); spreads.append(spread); slopes.append(slope)
    a0 = A[A.m == m_use].groupby("sigma").twonn.mean()
    b0 = B.groupby("true_d").twonn.mean()
    slope0 = np.polyfit(b0.index.values.astype(float), b0.values, 1)[0]
    spread0 = a0.max() - a0.min()
    floor0 = spread0 / abs(slope0)
    print(f"  clustering-induced TwoNN spread at m={m_use}: {spread0:.3f}")
    print(f"  TwoNN sensitivity to true d:                 {slope0:.3f} per unit d")
    print(f"  DETECTION FLOOR: {floor0:.2f} dimensions "
          f"[{np.percentile(floors,2.5):.2f}, {np.percentile(floors,97.5):.2f}] (95% CI)")
    print()
    print("  Comparison with the real signal reported in the paper:")
    print("    real activations: TwoNN falls 16.96 -> 14.05 through the transition, i.e. -2.91")
    print(f"    in dimension units that is {2.91/abs(slope0):.2f} dimensions")
    verdict = ("EXCEEDS the floor -- a real (small) dimensional change is established"
               if 2.91 / abs(slope0) > floor0 else
               "does NOT exceed the floor -- we can establish only that the large reported "
               "collapses are confounded, not that any real dimensional change occurs")
    print(f"    verdict: {verdict}")
    pd.DataFrame([{"m_real": m_real, "m_used": m_use, "spread": spread0, "slope": slope0,
                   "floor_dims": floor0, "floor_lo": np.percentile(floors, 2.5),
                   "floor_hi": np.percentile(floors, 97.5),
                   "real_signal_dims": 2.91 / abs(slope0)}]).to_csv(
        OUT / "twonn_floor_summary.csv", index=False)
    print(f"\ndone in {time.time()-t0:.0f}s")
