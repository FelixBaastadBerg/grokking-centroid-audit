"""
C1 stress test -- is the estimator-triage result an artifact of ONE synthetic construction?

The headline numbers (37x clustering response at fixed dimension; inversion of corr_dim and
Monod H1_max at the operating point) were measured on a single cloud family: isotropic Gaussian
patches, K=8 centres, shared random linear embedding. If any of those choices is doing the work,
the claim is about my generator, not about the estimators.

So the same two arms are re-run across a grid of DELIBERATELY different constructions:

  manifold   how each patch is shaped      gauss | ball | sphere | cube | curved (swiss-roll-like)
  layout     how the patches are arranged  random | grid | hierarchical | unequal-sizes
  K          number of clusters            4 | 8 | 16
  d range    true dimension swept          2..10

True intrinsic dimension is d in every case by construction. If the conclusion is real it should
survive all of them; anything that flips identifies the artifact.
"""
from __future__ import annotations
import os, sys, warnings, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import spearmanr
from tools.intervention import clusteredness, estimators, EST

OUT = REPO / "RESULTS"
N, AMB = 400, 60


def patch(kind, m, d, g):
    """A patch of TRUE intrinsic dimension d, in d coordinates."""
    if kind == "gauss":
        return g.normal(size=(m, d))
    if kind == "ball":
        z = g.normal(size=(m, d)); z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z * g.uniform(size=(m, 1)) ** (1 / d)
    if kind == "sphere":                       # d-sphere lives in d+1 coords, ID = d
        z = g.normal(size=(m, d + 1)); return z / np.linalg.norm(z, axis=1, keepdims=True)
    if kind == "cube":
        return g.uniform(-1, 1, size=(m, d))
    if kind == "curved":                       # curved d-manifold: swiss-roll in first 2 coords
        z = g.uniform(size=(m, d))
        t = 1.5 * np.pi * (1 + 2 * z[:, 0])
        out = np.c_[t * np.cos(t) / 8, t * np.sin(t) / 8]
        return np.c_[out, z[:, 2:]] if d > 2 else out
    raise ValueError(kind)


def centres(layout, K, amb, g, R=1.0):
    if layout == "random":
        return g.normal(size=(K, amb)) * R
    if layout == "grid":
        side = int(np.ceil(K ** (1 / 3)))
        pts = np.array([[i, j, k] for i in range(side) for j in range(side)
                        for k in range(side)], float)[:K]
        pts = (pts - pts.mean(0)) / (pts.std() + 1e-9) * R
        M = np.zeros((K, amb)); M[:, :3] = pts
        return M @ np.linalg.qr(g.normal(size=(amb, amb)))[0]
    if layout == "hier":                        # clusters of clusters
        n1 = max(2, K // 4)
        top = g.normal(size=(n1, amb)) * R * 2
        out = [top[i % n1] + g.normal(size=amb) * R * 0.3 for i in range(K)]
        return np.array(out)
    raise ValueError(layout)


def make(kind, layout, K, d, sigma, n=N, amb=AMB, seed=0, unequal=False):
    g = np.random.default_rng(seed)
    cen = centres(layout, K, amb, g) if K > 1 else np.zeros((1, amb))
    dd = d + 1 if kind == "sphere" else d
    P = g.normal(size=(dd, amb))
    if unequal:
        w = g.uniform(0.3, 3.0, size=K); w = w / w.sum()
        sizes = np.maximum(6, (w * n).astype(int))
    else:
        sizes = np.full(K, int(np.ceil(n / K)))
    pts = []
    for c, m in zip(cen, sizes):
        pts.append(c + (patch(kind, m, d, g) @ P) * sigma)
    X = np.vstack(pts)
    return X[g.permutation(len(X))[:n]]


def tune(kind, layout, K, d, target, seed=0, unequal=False, iters=14):
    lo, hi = 1e-3, 3.0
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        s = clusteredness(make(kind, layout, K, d, mid, seed=seed, unequal=unequal))
        if s > target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


CONSTRUCTIONS = [
    ("gauss", "random", 8, False),      # the original
    ("ball", "random", 8, False),
    ("sphere", "random", 8, False),
    ("cube", "random", 8, False),
    ("curved", "random", 8, False),
    ("gauss", "grid", 8, False),
    ("gauss", "hier", 8, False),
    ("gauss", "random", 4, False),
    ("gauss", "random", 16, False),
    ("gauss", "random", 8, True),       # unequal cluster sizes
]

if __name__ == "__main__":
    t0 = time.time()
    pd.set_option("display.width", 230)

    print("=== ARM A across constructions: TRUE d FIXED at 5, clusteredness swept ===")
    print("    fold change = max/min of the estimator over the clusteredness sweep\n")
    rowsA = []
    for kind, layout, K, uneq in CONSTRUCTIONS:
        recs = []
        for sigma in (1.5, 0.6, 0.25, 0.1, 0.03, 0.01):
            for seed in (0, 1, 2):
                X = make(kind, layout, K, 5, sigma, seed=seed, unequal=uneq)
                recs.append({"sigma": sigma, "seed": seed, "sil": clusteredness(X), **estimators(X)})
        A = pd.DataFrame(recs)
        nm = f"{kind}/{layout}/K{K}" + ("/uneq" if uneq else "")
        r = {"construction": nm, "sil_range": f"{A.sil.min():.2f}-{A.sil.max():.2f}"}
        for e in EST:
            m = A.groupby("sigma")[e].mean()
            r[e] = m.max() / max(abs(m.min()), 1e-9)
        rowsA.append(r)
        print(f"  {nm:26s} done ({time.time()-t0:.0f}s)", flush=True)
    RA = pd.DataFrame(rowsA)
    RA.to_csv(OUT / "robust_c1_armA.csv", index=False)
    print("\n  FOLD CHANGE with true dimension held at 5:")
    print(RA.set_index("construction").round(1).to_string())
    print("\n  median fold change across constructions:")
    print(RA[EST].median().round(1).to_string())

    print("\n\n=== ARM B across constructions: TRUE d swept, clusteredness FIXED at 0.70 ===")
    print("    rho(estimator, true d); negative = INVERTED, near 0 = FLAT\n")
    rowsB = []
    for kind, layout, K, uneq in CONSTRUCTIONS:
        recs = []
        for d in (2, 4, 6, 8, 10):
            for seed in (0, 1, 2):
                sig = tune(kind, layout, K, d, 0.70, seed=seed, unequal=uneq)
                X = make(kind, layout, K, d, sig, seed=seed, unequal=uneq)
                recs.append({"true_d": d, "seed": seed, "sil": clusteredness(X), **estimators(X)})
        B = pd.DataFrame(recs)
        nm = f"{kind}/{layout}/K{K}" + ("/uneq" if uneq else "")
        r = {"construction": nm, "sil": B.sil.mean()}
        for e in EST:
            r[e] = spearmanr(B[e], B.true_d).statistic
        rowsB.append(r)
        print(f"  {nm:26s} done ({time.time()-t0:.0f}s)", flush=True)
    RB = pd.DataFrame(rowsB)
    RB.to_csv(OUT / "robust_c1_armB.csv", index=False)
    print("\n  rho(estimator, TRUE d) at silhouette 0.70:")
    print(RB.set_index("construction").round(2).to_string())
    print("\n  median rho across constructions:")
    print(RB[EST].median().round(2).to_string())
    print("\n  fraction of constructions where the estimator INVERTS (rho < -0.3):")
    print((RB[EST] < -0.3).mean().round(2).to_string())
    print(f"\ndone in {time.time()-t0:.0f}s")
