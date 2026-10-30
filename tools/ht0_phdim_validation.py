"""
H-T PREREQUISITE -- is the trajectory PH-dim estimator trustworthy at our point counts?

Our sliding-window fits came out at r^2 = 0.33-0.53, which cannot support an "event"
claim. Before fixing anything we need to know whether that is (a) the estimator being
noisy at small n, (b) the fit window, or (c) the trajectory genuinely not having a
clean scaling regime.

So: run the estimator on synthetic trajectories of KNOWN fractal dimension, at the
point counts we actually have.

Known-answer objects (all are *paths*, i.e. the same kind of object as a training
trajectory -- not blobs):
  smooth helix           dim 1
  fBm path, H=0.75       dim = 1/H = 1.33   (trace of fBm in R^m, m > 1/H)
  fBm path, H=0.5        dim = 2.0          (Brownian motion)
  fBm path, H=0.33       dim = 3.0
  Lorenz attractor       dim ~ 2.05         (correlation dimension, literature)

Birdal's own code samples n in [200, 1000] step 50 -- it needs >= 1000 points. Our
runs have ~300 checkpoints. That mismatch is the first suspect.
"""
from __future__ import annotations

import os, sys, warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

OUT = REPO / "RESULTS"; OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------- synthetic paths
def helix(n, rng, amb=64):
    t = np.linspace(0, 8 * np.pi, n)
    X = np.c_[np.cos(t), np.sin(t), 0.3 * t]
    return _embed(X, amb, rng), 1.0


def fbm_path(n, rng, H=0.5, m=3, amb=64):
    """Davies-Harte-ish via cumulative fractional noise (spectral synthesis)."""
    cols = []
    for _ in range(m):
        f = np.fft.rfftfreq(n * 2)
        f[0] = f[1]
        amp = f ** (-(H + 0.5))
        ph = rng.uniform(0, 2 * np.pi, len(f))
        # spectral density |amp|^2 = f^-(2H+1) IS fBm already -- an extra cumsum
        # integrates a second time and yields H+1 (a smooth curve). Do not cumsum.
        s = np.fft.irfft(amp * np.exp(1j * ph), n=n * 2)[:n]
        cols.append(s - s.mean())
    X = np.column_stack(cols)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return _embed(X, amb, rng), min(m, 1.0 / H)


def lorenz(n, rng, amb=64, dt=0.01):
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    xyz = np.array([1.0, 1.0, 1.0]); pts = []
    for i in range(n * 5 + 2000):
        x, y, z = xyz
        xyz = xyz + dt * np.array([s * (y - x), x * (r - z) - y, x * y - b * z])
        if i >= 2000 and i % 5 == 0:
            pts.append(xyz.copy())
    X = np.array(pts)[:n]
    X = (X - X.mean(0)) / X.std(0)
    return _embed(X, amb, rng), 2.05


def _embed(X, amb, rng):
    n, d = X.shape
    if amb <= d:
        return X
    Z = np.zeros((n, amb)); Z[:, :d] = X
    Q, _ = np.linalg.qr(rng.normal(size=(amb, amb)))
    return Z @ Q


# ---------------------------------------------------------------- estimator
def mst_lifetime_sum(X, alpha=1.0):
    D = squareform(pdist(X))
    T = minimum_spanning_tree(D)
    e = np.asarray(T.data, float)
    e = e[np.isfinite(e) & (e > 0)]
    return float((e ** alpha).sum()) if e.size else np.nan


def ph_dim(X, n_list, n_repeats=5, alpha=1.0, seed=0):
    """Birdal's estimator: slope of log E_alpha vs log n, dim = alpha/(1-slope)."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    ns = [n for n in n_list if n <= N]
    if len(ns) < 3:
        return np.nan, np.nan, 0
    xs, ys = [], []
    for n in ns:
        for _ in range(n_repeats):
            v = mst_lifetime_sum(X[rng.choice(N, n, replace=False)], alpha)
            if np.isfinite(v) and v > 0:
                xs.append(np.log(n)); ys.append(np.log(v))
    if len(xs) < 6:
        return np.nan, np.nan, 0
    xs, ys = np.array(xs), np.array(ys)
    m, c = np.polyfit(xs, ys, 1)
    yh = m * xs + c
    ss_res = float(((ys - yh) ** 2).sum()); ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = np.nan if ss_tot <= 1e-12 else 1 - ss_res / ss_tot
    d = np.inf if m >= 1 else alpha / (1 - m)
    return float(d), float(r2), len(ns)


SCHEDULES = {
    "birdal_200_1000":  list(range(200, 1001, 50)),
    "wide_50_1000":     [50, 75, 100, 150, 200, 300, 400, 600, 800, 1000],
    "ours_16_256":      [16, 24, 32, 48, 64, 96, 128, 192, 256],
    "mid_32_512":       [32, 48, 64, 96, 128, 192, 256, 384, 512],
    "dense_64_1024":    [64, 96, 128, 192, 256, 384, 512, 768, 1024],
}


def main() -> int:
    shapes = [
        ("helix",      lambda n, r: helix(n, r)),
        ("fbm_H0.75",  lambda n, r: fbm_path(n, r, H=0.75)),
        ("fbm_H0.50",  lambda n, r: fbm_path(n, r, H=0.50)),
        ("fbm_H0.33",  lambda n, r: fbm_path(n, r, H=0.33)),
        ("lorenz",     lambda n, r: lorenz(n, r)),
    ]
    rows = []
    for name, fn in shapes:
        for N in (300, 1000, 3000):
            for seed in (0, 1, 2):
                rng = np.random.default_rng(100 * seed + 7)
                X, d_true = fn(N, rng)
                for sname, sched in SCHEDULES.items():
                    if min(sched) > N:
                        continue
                    d, r2, k = ph_dim(X, sched, n_repeats=5, seed=seed)
                    rows.append({"shape": name, "d_true": round(d_true, 3), "N": N,
                                 "seed": seed, "schedule": sname, "n_levels": k,
                                 "d_est": d, "r2": r2,
                                 "abs_err": abs(d - d_true) if np.isfinite(d) else np.nan})
            print(f"  {name:<11} N={N}", flush=True)
    D = pd.DataFrame(rows); D.to_csv(OUT / "ht0_phdim_validation.csv", index=False)
    pd.set_option("display.width", 240)

    print("\n=== PH-dim accuracy and fit quality, by point count and sampling schedule ===")
    g = D.groupby(["N", "schedule"]).agg(
        n_cfg=("d_est", "size"), mean_r2=("r2", "mean"),
        mean_abs_err=("abs_err", "mean"), med_abs_err=("abs_err", "median")).round(3)
    print(g.to_string())

    print("\n=== per shape, at OUR point count (N=300) ===")
    s = D[(D.N == 300)].groupby(["shape", "d_true", "schedule"]).agg(
        d_est=("d_est", "mean"), sd=("d_est", "std"), r2=("r2", "mean")).round(3)
    print(s.to_string())

    print("\n=== the direct comparison: Birdal's schedule needs N>=1000 ===")
    b = D[(D.schedule == "birdal_200_1000")].groupby(["shape", "d_true", "N"]).agg(
        d_est=("d_est", "mean"), r2=("r2", "mean")).round(3)
    print(b.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
