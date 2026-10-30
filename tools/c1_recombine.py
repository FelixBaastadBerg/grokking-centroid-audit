"""
C1 -- counterfactual recombination of checkpoints: which half of the geometry carries the signal?

The centroid intervention shows that MOVING class separation by hand reproduces the reported
change. It does not show that the change the network actually undergoes is a between-class change,
because the network alters both factors at once. This closes that gap using only retained
checkpoints, by building clouds the network never produced:

    X~_{i; s,t}  =  mu_{y_i, s}  +  (x_{i,t} - mu_{y_i, t})

-- class centroids taken from checkpoint s, class-centred residuals from checkpoint t. The diagonal
s = t is the network's own trajectory. Off-diagonal cells hold one factor at its early value while
the other advances, so over an (s, t) grid each estimator's training-time change decomposes into

    between-class main effect   change along s at fixed t
    within-class main effect    change along t at fixed s
    interaction                 the remainder

If the between-class main effect accounts for essentially all of the diagonal change, the reported
"dimensional collapse" is a statement about where the classes sit, not about the geometry inside
them -- and that is an attribution about the real trajectory rather than about a hand-made sweep.

Analysis-only: retained checkpoints, no training.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from analysis.lsrun import LandscapeRun
from tools.e1_centroid import measure
from tools.b3_seed_alpha import select

OUT = REPO / "RESULTS"
N_CK = 5          # checkpoints spanning the run, t_grok included
N_SUB = 400
ESTS = ["ph_dim", "H1_total", "twonn", "mle_k10", "eff_rank", "pca_pr", "corr_dim", "sil_true"]


def checkpoints(r):
    """Indices spanning init -> convergence, with t_grok forced in."""
    n = len(r.epochs)
    tg = r.t_grok()
    idx = sorted(set([0, max(1, tg // 2), tg if tg > 0 else n // 2,
                      (tg + n) // 2 if tg > 0 else (3 * n) // 4, n - 1]))
    while len(idx) < N_CK:
        idx.append(min(n - 1, idx[-1] + 1))
    return sorted(set(idx))[:N_CK]


def acts(r, i, X):
    r._load(r.flat(i)); r.model.eval()
    with torch.no_grad():
        r.model(X, keep_activations=True)
        return r.model.activations[-2].numpy()


def parts(H, y):
    """Class centroids per point, and the class-centred residual."""
    cls = np.unique(y)
    M = {c: H[y == c].mean(0) for c in cls}
    cen = np.stack([M[c] for c in y])
    return cen, H - cen


def one_run(rec):
    r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
    y_all = r.Ytr.numpy()
    sub = np.sort(np.random.default_rng(0).choice(len(y_all), min(N_SUB, len(y_all)),
                                                  replace=False))
    y = y_all[sub]
    idx = checkpoints(r)
    cen, res = {}, {}
    for i in idx:
        H = acts(r, i, r.Xtr)[sub]
        cen[i], res[i] = parts(H, y)
    rows = []
    for s in idx:
        for t in idx:
            Z = cen[s] + res[t]
            Zc = Z - Z.mean(0)
            Z = Zc / (np.sqrt((Zc ** 2).sum(1).mean()) + 1e-12)   # same trace normalisation as E1
            rows.append({"dir": rec["dir"], "seed": int(rec["seed"]),
                         "s": idx.index(s), "t": idx.index(t),
                         "epoch_s": int(r.epochs[s]), "epoch_t": int(r.epochs[t]),
                         **measure(Z, y)})
    return rows


def decompose_effects(D):
    """Share of the diagonal change explained by moving each factor alone."""
    out = []
    for est in ESTS:
        per = []
        for d, g in D.groupby("dir"):
            lo, hi = g.s.min(), g.s.max()
            def v(s, t):
                q = g[(g.s == s) & (g.t == t)][est]
                return float(q.iloc[0]) if len(q) else np.nan
            diag = v(lo, lo) - v(hi, hi)                 # the network's own change
            betw = v(lo, lo) - v(hi, lo)                 # advance centroids only
            with_ = v(lo, lo) - v(lo, hi)                # advance residuals only
            if not np.isfinite(diag) or abs(diag) < 1e-9:
                continue
            per.append({"dir": d, "diag": diag, "between": betw, "within": with_,
                        "between_share": betw / diag, "within_share": with_ / diag,
                        "interaction_share": 1 - (betw + with_) / diag})
        if not per:
            continue
        P = pd.DataFrame(per)
        out.append({"estimator": est, "n_runs": len(P),
                    "diag_change": P.diag.median(),
                    "between_share": P.between_share.median(),
                    "within_share": P.within_share.median(),
                    "interaction_share": P.interaction_share.median(),
                    "between_lo": P.between_share.quantile(.25),
                    "between_hi": P.between_share.quantile(.75)})
    return pd.DataFrame(out)


def main():
    from multiprocessing import Pool
    t0 = time.time()
    inv = pd.read_csv(OUT / "_inventory.csv")
    sel = select(inv)
    print(f"=== C1: counterfactual checkpoint recombination, {len(sel)} seeds, "
          f"{N_CK}x{N_CK} (s,t) grid ===\n", flush=True)
    recs = [r for _, r in sel.iterrows()]
    rows = []
    with Pool(processes=5) as pool:
        for i, rr in enumerate(pool.imap_unordered(one_run, recs), 1):
            rows += rr
            print(f"  [{i}/{len(recs)}] ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "c1_recombine.csv", index=False)

    E = decompose_effects(D)
    E.to_csv(OUT / "c1_effects.csv", index=False)
    print("\n--- share of the network's own training-time change reproduced by moving ONE factor ---")
    print(f"  {'estimator':<12}{'n':>4}{'diag change':>13}{'between':>10}{'within':>10}"
          f"{'interact':>10}{'  between IQR':>18}")
    for _, r in E.iterrows():
        print(f"  {r.estimator:<12}{r.n_runs:>4}{r.diag_change:>13.3f}"
              f"{r.between_share:>9.0%}{r.within_share:>10.0%}{r.interaction_share:>10.0%}"
              f"   [{r.between_lo:>5.0%},{r.between_hi:>5.0%}]")
    print(f"\nwrote c1_recombine.csv, c1_effects.csv   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
