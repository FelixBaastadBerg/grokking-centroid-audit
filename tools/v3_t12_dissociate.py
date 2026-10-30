"""
T1.2 -- do the geometric progress measures respond to genuine group structure, or to generic
class separation?

Construction: K classes, m points per class, within-class isotropic d=5 manifold at fixed scale
in R^60. The CENTROID ARRANGEMENT varies by family; every family's centroid cloud is centred and
normalised to unit RMS norm, then scaled by one common separation factor, so total separation and
centroid scale are matched and only the arrangement differs.

  fourier3        c -> sum of 3 random-frequency Fourier pairs (the arrangement grokked networks
                  actually form; rank 6, circulant distance structure)
  circular        single frequency (rank 2)
  simplex         regular simplex, the neural-collapse ETF (rank K-1, all distances equal)
  random_full     i.i.d. Gaussian centroids (rank min(K-1, 60))
  random_rank6    i.i.d. Gaussian centroids in a random 6-d subspace (RANK-matched to fourier3)
  fourier3_phase  fourier3 with re-randomised phases. Phases enter pairwise distances only
                  through differences, so this is an ISOMETRY of the centroid set: every
                  label-free statistic must be exactly invariant. Numerical check of the claim.
  fourier3_perm   fourier3 with the class -> centroid correspondence permuted. The union cloud
                  is identical; only which element sits where changes. Label-free statistics
                  and silhouette are invariant BY CONSTRUCTION; the equivariance order
                  parameter is the only statistic that can respond.

Alongside the estimator battery we compute the equivariance order parameter of the centroid
matrix itself (max of left/right R2 against the Z_K difference table), which is the one
statistic that sees group correspondence.

Output: RESULTS/v3_dissociate.csv
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from tools.e1_centroid import measure
from analysis.group_structure import diff_table_for, right_diff_table, equivariance_r2

OUT = REPO / "RESULTS"
K, M, D_WITHIN, AMB = 23, 17, 5, 60
SEP_LEVELS = {"low": 2.0, "matched": 4.0, "high": 10.0}   # x within-scale; realised sil reported
N_SEEDS = 8


def fourier_centroids(K, freqs, phases, amps, rng):
    cols = []
    for f, ph, a in zip(freqs, phases, amps):
        t = 2 * np.pi * f * np.arange(K) / K + ph
        cols += [a * np.cos(t), a * np.sin(t)]
    C = np.stack(cols, 1)                                   # K x 2F
    Q = np.linalg.qr(rng.normal(size=(AMB, C.shape[1])))[0]  # isometric embed into R^AMB
    return C @ Q.T


def make_centroids(family, rng):
    freqs = rng.choice(np.arange(1, K // 2 + 1), 3, replace=False)
    phases = rng.uniform(0, 2 * np.pi, 3)
    amps = np.ones(3)
    if family == "fourier3":
        C = fourier_centroids(K, freqs, phases, amps, rng)
    elif family == "fourier3_phase":
        C = fourier_centroids(K, freqs, rng.uniform(0, 2 * np.pi, 3), amps, rng)
    elif family == "fourier3_perm":
        C = fourier_centroids(K, freqs, phases, amps, rng)[rng.permutation(K)]
    elif family == "circular":
        C = fourier_centroids(K, [1], [phases[0]], [1.0], rng)
    elif family == "simplex":
        C = np.eye(K) - 1.0 / K
        Q = np.linalg.qr(rng.normal(size=(AMB, K)))[0][:, :K]
        C = C @ Q.T
    elif family == "random_full":
        C = rng.normal(size=(K, AMB))
    elif family == "random_rank6":
        B = np.linalg.qr(rng.normal(size=(AMB, 6)))[0]
        C = rng.normal(size=(K, 6)) @ B.T
    else:
        raise ValueError(family)
    C = C - C.mean(0)
    return C / (np.sqrt((C ** 2).sum(1).mean()) + 1e-12)     # unit RMS: scale matched


def cloud(C, sep, rng):
    P = np.linalg.qr(rng.normal(size=(AMB, D_WITHIN)))[0]    # within-class d-frame
    y = np.repeat(np.arange(K), M)
    W = rng.normal(size=(K * M, D_WITHIN)) @ P.T             # unit-scale within noise
    return sep * C[y] + W, y


def centroid_stats(C):
    from scipy.spatial.distance import pdist
    d = pdist(C)
    s = np.linalg.svd(C - C.mean(0), compute_uv=False) ** 2
    s = s[s > 1e-12]; p = s / s.sum()
    return {"cen_dist_mean": float(d.mean()), "cen_dist_cv": float(d.std() / d.mean()),
            "cen_eff_rank": float(np.exp(-(p * np.log(p)).sum()))}


def main():
    Dl, Dr = diff_table_for("add", K), right_diff_table("add", K)
    fams = ["fourier3", "circular", "simplex", "random_full", "random_rank6",
            "fourier3_phase", "fourier3_perm"]
    rows, t0 = [], time.time()
    for fam in fams:
        for sep_name, sep in SEP_LEVELS.items():
            for sd in range(N_SEEDS):
                rng = np.random.default_rng(1000 * sd + hash(fam) % 997)
                C = make_centroids(fam, rng)
                X, y = cloud(C, sep, rng)
                Xc = X - X.mean(0)
                Z = Xc / (np.sqrt((Xc ** 2).sum(1).mean()) + 1e-12)
                eq = float(np.nanmax([equivariance_r2(C, Dl),
                                      equivariance_r2(C, Dr) if Dr is not None else np.nan]))
                rows.append({"family": fam, "sep": sep_name, "sep_factor": sep, "seed": sd,
                             "equiv_R2_centroids": eq, **centroid_stats(C), **measure(Z, y)})
        print(f"  {fam} done ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "v3_dissociate.csv", index=False)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "v3_dissociate.csv", index=False)

    print("\n=== medians at the matched separation level ===")
    g = D[D.sep == "matched"].groupby("family")
    cols = ["sil_true", "ph_dim", "twonn", "eff_rank", "H1_total", "H1_max",
            "equiv_R2_centroids", "cen_eff_rank", "cen_dist_cv"]
    print(g[cols].median().round(3).to_string())
    print(f"\nwrote v3_dissociate.csv {D.shape} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
