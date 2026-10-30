"""
DIR B, via density slices rather than the full 2-parameter module.

`multipers` 2.6.1 segfaults in this environment inside its own constructors (both
SimplexTreeMulti(gudhi_tree) and filtrations.RipsCodensity crash before returning), so the
full bipersistence module is not computable here. That blocks the *summary*, not the
*scientific premise*, so this tests the premise directly.

The premise of DIR B is that a second filtration parameter disentangles the multi-frequency
superposition that made single-parameter Rips report b1 = 0. The (Rips x codensity)
bifiltration restricted to a fixed codensity level IS an ordinary Rips complex on the density
sublevel, and those restrictions are exactly the axis-parallel slices of the bifiltration.
So: sweep the codensity threshold, run ordinary PH on each dense subset, and ask whether ANY
slice recovers structure that the full cloud misses.

If no slice ever beats the full cloud on data where the answer is known, the second parameter
is not buying anything and DIR B is dead regardless of which summary one would have used.
If some slice does, that is a real finding and justifies getting the module computed properly.

Known answers used here:
  clean circle          -> b1 = 1 at some slice
  genuine 2-torus       -> b1 = 2 (the case 1-parameter PH already fails, reading b1 = 3)
  k superposed modes    -> b1 should reveal k circles; 1-parameter PH gives 0 for k >= 3
  pure noise            -> nothing, at every slice
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from ripser import ripser
from sklearn.neighbors import NearestNeighbors

OUT = REPO / "RESULTS"
QUANTILES = [1.0, 0.8, 0.6, 0.45, 0.3, 0.2]


def codensity(X, k=15):
    k = min(k + 1, len(X))
    d, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    return d[:, -1]


def betti_at(X, maxdim=1, frac=0.15):
    """b_i counted as bars longer than `frac` of the cloud diameter -- the calibrated rule."""
    if len(X) < 6:
        return {f"b{i}": 0 for i in range(1, maxdim + 1)} | {"H1max": 0.0}
    D = np.sqrt(np.maximum(((X[:, None] - X[None]) ** 2).sum(-1), 0))
    diam = D.max()
    if diam <= 0:
        return {f"b{i}": 0 for i in range(1, maxdim + 1)} | {"H1max": 0.0}
    dg = ripser(D / diam, distance_matrix=True, maxdim=maxdim)["dgms"]
    out = {}
    for i in range(1, maxdim + 1):
        d = dg[i]
        d = d[np.isfinite(d[:, 1])] if len(d) else d
        life = (d[:, 1] - d[:, 0]) if len(d) else np.array([])
        out[f"b{i}"] = int((life > frac).sum())
        if i == 1:
            out["H1max"] = float(life.max()) if len(life) else 0.0
    return out


def slice_sweep(X, k=15, maxdim=1):
    cod = codensity(X, k)
    rows = []
    for q in QUANTILES:
        keep = cod <= np.quantile(cod, q)
        Xs = X[keep]
        r = betti_at(Xs, maxdim=maxdim)
        r.update({"q": q, "n": int(keep.sum())})
        rows.append(r)
    return pd.DataFrame(rows)


def torus(n, R=3.0, r=1.0, seed=0, amb=0):
    g = np.random.default_rng(seed)
    u, v = g.uniform(0, 2 * np.pi, n), g.uniform(0, 2 * np.pi, n)
    X = np.c_[(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)]
    if amb:
        X = X @ g.normal(size=(3, amb))
    return X


def modes(p, freqs, amb=60, seed=0, n_rep=1, jit=0.02):
    g = np.random.default_rng(seed)
    n = np.arange(p)
    E = np.zeros((p, amb))
    for k in freqs:
        E += np.c_[np.cos(2 * np.pi * k * n / p), np.sin(2 * np.pi * k * n / p)] @ g.normal(size=(2, amb))
    if n_rep > 1:
        E = np.repeat(E, n_rep, axis=0) + g.normal(size=(p * n_rep, amb)) * jit * E.std()
    return E


if __name__ == "__main__":
    print("=== DIR B PREMISE TEST: does ANY density slice beat the full cloud? ===")
    print("    (q=1.00 is the full cloud, i.e. ordinary single-parameter PH)\n")
    rows = []
    cases = [("clean circle n=529", np.c_[np.cos(np.linspace(0, 2 * np.pi, 529, endpoint=False)),
                                          np.sin(np.linspace(0, 2 * np.pi, 529, endpoint=False))], 1, 0),
             ("2-torus n=529", torus(529, seed=0), 2, 1),
             ("2-torus n=529 (60-dim)", torus(529, seed=1, amb=60), 2, 1),
             ("noise n=529 (60-dim)", np.random.default_rng(2).normal(size=(529, 60)), 0, 0)]
    for k in (1, 2, 3, 5):
        fr = sorted(np.random.default_rng(7).choice(np.arange(1, 11), k, replace=False).tolist())
        cases.append((f"{k}-mode embedding n=529", modes(23, fr, n_rep=23, seed=0), k, None))
    for nm, X, b1_true, b2_true in cases:
        S = slice_sweep(X)
        S["case"] = nm; S["b1_true"] = b1_true
        rows.append(S)
        best = S.loc[S.b1.idxmax()]
        print(f"  {nm:28s} b1_true={b1_true}")
        print(f"      q:      " + "  ".join(f"{q:5.2f}" for q in S.q))
        print(f"      n:      " + "  ".join(f"{int(x):5d}" for x in S.n))
        print(f"      b1:     " + "  ".join(f"{int(x):5d}" for x in S.b1))
        print(f"      b2:     (not computed at n=529; H2 on 529 pts is O(n^3))")
        print(f"      H1max:  " + "  ".join(f"{x:5.2f}" for x in S.H1max))
        print(f"      -> full cloud b1={int(S[S.q==1.0].b1.iloc[0])}, best slice b1={int(best.b1)} at q={best.q}")
        print()
    pd.concat(rows, ignore_index=True).to_csv(OUT / "dirB_slices.csv", index=False)
