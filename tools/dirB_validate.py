"""
DIR B gate -- 2-parameter persistence over (Rips scale x codensity).

The rationale for trying it at all: single-parameter Rips reported b1 = 0 on every cyclic run,
and the synthetic diagnosis was that a superposition of ~5 Fourier modes has no single scale at
which one clean cycle exists. A second filtration parameter is exactly the standard remedy for
"no single scale works" -- so the question is whether it recovers structure that 1-parameter PH
provably misses.

Gate, in order (a failure at any step kills the direction):
  1. multipers actually runs and returns a module we can summarise
  2. on a CLEAN circle at these point counts, the 2-parameter summary finds 1 dominant class
  3. on a 5-frequency embedding -- where 1-parameter b1 = 0 is already established -- does it
     find anything? and does the count match the known number of modes?
  4. on pure noise, it must find nothing (per-cloud null)

Point counts are set to the pair-representation regime (n ~ 529), not |G|, since escaping the
|G| ceiling is the entire reason this direction is on the list.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np

# ------------------------------------------------------------------ API discovery
import multipers as mp
import gudhi
print("multipers", mp.__version__)
print("top-level:", [a for a in dir(mp) if not a.startswith("_")])
try:
    import multipers.ml.signed_measures as sm
    print("signed_measures:", [a for a in dir(sm) if not a.startswith("_")][:20])
except Exception as e:
    print("signed_measures import:", type(e).__name__, e)


def codensity(X, k=15):
    """Codensity = distance to the k-th nearest neighbour. Low where the cloud is dense."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    d, _ = nn.kneighbors(X)
    return d[:, -1]


def synth_modes(p, freqs, amb=60, seed=0, noise=0.0, n_rep=1):
    """The same construction whose 5-frequency case gives ambient b1 = 0."""
    r = np.random.default_rng(seed)
    n = np.arange(p)
    E = np.zeros((p, amb))
    for k in freqs:
        E += np.c_[np.cos(2 * np.pi * k * n / p), np.sin(2 * np.pi * k * n / p)] @ r.normal(size=(2, amb))
    if n_rep > 1:                      # densify: jitter copies, to reach pair-representation counts
        E = np.repeat(E, n_rep, axis=0) + r.normal(size=(p * n_rep, amb)) * 0.02 * E.std()
    if noise:
        E = E + r.normal(size=E.shape) * noise * E.std()
    return E


def two_param(X, degree=1, k=15, max_edge=None, verbose=False):
    """(Rips scale x codensity) module; returns the summary we will actually use."""
    from scipy.spatial.distance import pdist
    D = pdist(X)
    me = float(np.quantile(D, 0.35)) if max_edge is None else max_edge
    stg = gudhi.RipsComplex(points=X, max_edge_length=me).create_simplex_tree(max_dimension=degree + 1)
    st = mp.SimplexTreeMulti(stg, num_parameters=2)
    st.fill_lowerstar(codensity(X, k), parameter=1)
    mod = mp.module_approximation(st)
    bars = mod.get_module_of_degree(degree)
    vols = []
    for b in bars:
        try:
            births = np.asarray(b.get_birth_list(), float)
            deaths = np.asarray(b.get_death_list(), float)
            fin = np.isfinite(deaths).all(axis=1) if deaths.ndim > 1 else np.isfinite(deaths)
            if deaths.ndim > 1 and fin.any():
                vols.append(float(np.max(deaths[fin].min(axis=1) - births[:len(deaths[fin])].min(axis=1))))
            else:
                vols.append(0.0)
        except Exception:
            vols.append(0.0)
    vols = np.sort(np.array(vols))[::-1]
    return {"n_bars": len(bars), "vols": vols, "max_edge": me}


if __name__ == "__main__":
    print("\n=== GATE 2: clean circle (known: exactly 1 class) ===")
    for n in (120, 529):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        X = np.c_[np.cos(t), np.sin(t)] + np.random.default_rng(0).normal(size=(n, 2)) * 0.02
        r = two_param(X)
        print(f"  n={n:4d}  n_bars={r['n_bars']:4d}  top volumes {np.round(r['vols'][:5], 4)}")

    print("\n=== GATE 3: multi-frequency embeddings (1-param PH gives b1=0 at k>=3) ===")
    for k in (1, 2, 3, 5):
        for n_rep, p in ((1, 529), (23, 23)):
            X = synth_modes(p, sorted(np.random.default_rng(7).choice(np.arange(1, 11), k, replace=False).tolist()),
                            seed=0, n_rep=n_rep)
            r = two_param(X)
            tag = f"p={p} x{n_rep} -> n={X.shape[0]}"
            print(f"  k_true={k}  {tag:22s} n_bars={r['n_bars']:4d}  top volumes {np.round(r['vols'][:6], 4)}")

    print("\n=== GATE 4: pure noise (must find nothing structured) ===")
    for n in (120, 529):
        X = np.random.default_rng(1).normal(size=(n, 60))
        r = two_param(X)
        print(f"  n={n:4d}  n_bars={r['n_bars']:4d}  top volumes {np.round(r['vols'][:5], 4)}")
