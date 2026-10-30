"""
T3.2 -- is part of the PH-dimension collapse an artifact of how mixture structure changes along
the subsample-size schedule?

The estimator fits log E against log n over subsamples n in [~80, 400]. On a K-class mixture a
RANDOM subsample of size n has ~n/K points per class, so the schedule sweeps points-per-class as
well as n -- exactly the variable the estimator is sensitive to. Three subsampling schemes on the
same clouds separate the readings:

  random          the estimator's own scheme (baseline);
  balanced        class-stratified: exactly n/K per class at every n (removes the class-count
                  fluctuation but keeps the m sweep);
  fixed_m         vary K, hold m per class fixed: subsample whole classes (n = m K', K' <= K);
  fixed_K         vary m, hold K fixed (identical class set at every n) -- the pure m sweep.

Computed on the converged real activations of the 9 canonical seeds and on matched synthetic
mixtures. If the collapse were an artifact of the schedule's changing class balance, `balanced`
would differ from `random`; if it is carried by the m-dependence of the mixture energy (the T1.3
account), `fixed_K` shows the same fall and `fixed_m` does not.

Output: RESULTS/v3_stratified.csv
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from analysis.lsrun import LandscapeRun
from tools.b3_seed_alpha import select

OUT = REPO / "RESULTS"
ALPHA = 1.0
N_REP = 5


def E_alpha(X):
    mst = minimum_spanning_tree(squareform(pdist(X)))
    return float(np.sum(mst.data ** ALPHA))


def fit_dim(ns, Es):
    xs, ys = np.log(ns), np.log(Es)
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope = np.linalg.lstsq(A, ys, rcond=None)[0][0]
    return float(ALPHA / (1 - slope)) if slope < 1 - 1e-6 else np.nan


def scheme_dims(X, y, rng):
    n = len(X)
    K = len(np.unique(y))
    hi = min(n, 1000); lo = max(50, hi // 5)
    n_list = list(range(lo, hi + 1, max(1, (hi - lo) // 10)))
    out = {}
    # random and balanced share the n schedule
    for scheme in ("random", "balanced"):
        Es, ns_real = [], []
        for m_ in n_list:
            v, sz = [], []
            for _ in range(N_REP):
                if scheme == "random":
                    idx = rng.choice(n, m_, replace=False)
                else:
                    # stratified draws quantise the realised size to ~K*per: the fit must use
                    # the REAL size, not the nominal m_, or the slope is misspecified
                    per = max(1, m_ // K)
                    idx = np.concatenate([rng.choice(np.where(y == c)[0],
                                                     min(per, (y == c).sum()), replace=False)
                                          for c in np.unique(y)])
                v.append(E_alpha(X[idx])); sz.append(len(idx))
            Es.append(np.mean(v)); ns_real.append(np.mean(sz))
        # drop duplicate realised sizes (balanced quantisation makes neighbours identical)
        S = pd.DataFrame({"n": ns_real, "E": Es}).groupby("n").E.mean()
        out[scheme] = fit_dim(S.index.values, S.values) if len(S) >= 4 else np.nan
    # fixed m per class, vary K: subsample whole classes
    cls = np.unique(y)
    m_full = int(np.median([(y == c).sum() for c in cls]))
    Ks = [k for k in (4, 8, 12, 16, len(cls)) if k <= len(cls)]
    Es, ns = [], []
    for k in Ks:
        v = []
        for _ in range(N_REP):
            pick = rng.choice(cls, k, replace=False)
            idx = np.concatenate([np.where(y == c)[0] for c in pick])
            v.append(E_alpha(X[idx]))
        Es.append(np.mean(v)); ns.append(k * m_full)
    out["fixed_m_varyK"] = fit_dim(ns, Es)
    # fixed K (all classes), vary m per class (capped at each class's actual count)
    m_min = int(min((y == c).sum() for c in cls))
    ms = sorted({m for m in (3, 5, 8, 12, m_min) if m <= m_min})
    Es, ns = [], []
    for m_ in ms:
        v = []
        for _ in range(N_REP):
            idx = np.concatenate([rng.choice(np.where(y == c)[0], m_, replace=False)
                                  for c in cls])
            v.append(E_alpha(X[idx]))
        Es.append(np.mean(v)); ns.append(m_ * len(cls))
    out["fixed_K_varym"] = fit_dim(ns, Es)
    return out


def main():
    t0 = time.time()
    inv = pd.read_csv(OUT / "_inventory.csv")
    sel = select(inv)
    rows = []
    for _, rec in sel.iterrows():
        r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
        r._load(r.flat(len(r.epochs) - 1)); r.model.eval()
        with torch.no_grad():
            r.model(r.Xtr, keep_activations=True)
            H = r.model.activations[-2].numpy()
        y = r.Ytr.numpy()
        rng = np.random.default_rng(0)
        d0 = scheme_dims(H, y, rng)
        rows.append({"source": "real_converged", "dir": rec["dir"], **d0})
        # and at initialisation, to express the fold
        r._load(r.flat(0)); r.model.eval()
        with torch.no_grad():
            r.model(r.Xtr, keep_activations=True)
            H0 = r.model.activations[-2].numpy()
        rows.append({"source": "real_init", "dir": rec["dir"],
                     **scheme_dims(H0, y, np.random.default_rng(0))})
        print(f"  {rec['dir'][:50]} ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "v3_stratified.csv", index=False)

    print("\n=== median PH-dim by scheme ===")
    med = D.groupby("source")[["random", "balanced", "fixed_m_varyK", "fixed_K_varym"]].median()
    print(med.round(2).to_string())
    f = med.loc["real_init"] / med.loc["real_converged"]
    print("\nfold init/converged by scheme:")
    print(f.round(2).to_string())


if __name__ == "__main__":
    main()
