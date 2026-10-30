"""
Item 1 -- the missing direct baseline: measure class-conditional dimension DIRECTLY.

The paper diagnoses estimators that cannot separate class-conditional dimension from class
separation, and then declines to say whether within-class geometry actually compresses, hiding
behind a detection floor. But the labels are available, so the quantity can simply be measured.

CONDITIONAL ESTIMATORS. Each restricts an estimator's support to a point's own class:
  cond TwoNN     r1, r2 taken among same-class points only; the r2/r1 ratios pooled over all
                 classes and fitted exactly as TwoNN is fitted globally
  cond MLE-ID    the k-NN of each point restricted to same-class examples, k in {5, 10}
  within-class   effective rank / PCA participation ratio / spectral entropy of the residuals
  spectral       x_i - mu_{y_i} rather than of the global cloud
  per-class d_c  conditional TwoNN fitted separately inside each class, so a shared d is not
                 assumed; we report the median and spread over classes

These are invariant to the alpha intervention BY CONSTRUCTION -- moving class centroids cannot
change a within-class neighbour set -- which is exactly why they are the right instrument, and is
checked here rather than assumed.

VALIDATION FIRST, at the real operating point. A conditional estimator sees only m points per class,
so it must be shown calibrated at m ~ 17 before any trajectory is read off it. Synthetic clouds of
known d at matched m do that, and the answer sets how large a real change would have to be to count.

Then the actual question: does within-class geometry compress from init -> t_grok -> convergence?

Analysis-only: retained checkpoints.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from sklearn.neighbors import NearestNeighbors
from analysis import geom
from analysis.lsrun import LandscapeRun
from tools.b3_seed_alpha import select

OUT = REPO / "RESULTS"
D_TRUE, AMB = 5, 60


# ---------------------------------------------------------------- conditional estimators
def _twonn_from_mu(mu, discard=0.1):
    """The TwoNN fit of analysis/geom.py, applied to an externally supplied set of ratios."""
    mu = np.sort(np.asarray([m for m in mu if np.isfinite(m) and m > 1], float))
    if len(mu) < 10:
        return np.nan
    n = len(mu)
    keep = int(n * (1 - discard))
    F = np.arange(1, n + 1) / n
    x, y = np.log(mu[:keep]), -np.log(1 - F[:keep])
    return float(np.sum(x * y) / np.sum(x * x))


def cond_twonn(X, y, pooled=True):
    """r2/r1 among same-class points only. pooled=False returns one estimate per class."""
    X = np.asarray(X, float)
    ratios, per_class = [], {}
    for c in np.unique(y):
        Z = X[y == c]
        if len(Z) < 4:
            continue
        d, _ = NearestNeighbors(n_neighbors=3).fit(Z).kneighbors(Z)
        r1, r2 = d[:, 1], d[:, 2]
        ok = r1 > 0
        r = list(r2[ok] / r1[ok])
        ratios += r
        per_class[int(c)] = _twonn_from_mu(r)
    return (_twonn_from_mu(ratios) if pooled else per_class)


def cond_mle(X, y, k=10):
    """Levina-Bickel with the k-NN restricted to a point's own class."""
    X = np.asarray(X, float)
    vals = []
    for c in np.unique(y):
        Z = X[y == c]
        kk = min(k, len(Z) - 1)
        if kk < 3:
            continue
        d, _ = NearestNeighbors(n_neighbors=kk + 1).fit(Z).kneighbors(Z)
        d = d[:, 1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(d[:, [-1]] / d[:, :-1])
            v = (kk - 1) / np.maximum(lg.sum(1), 1e-12)
        vals += list(v[np.isfinite(v)])
    return float(np.mean(vals)) if vals else np.nan


def residuals(X, y):
    X = np.asarray(X, float)
    R = X.copy()
    for c in np.unique(y):
        R[y == c] -= X[y == c].mean(0)
    return R


def spectral(Z):
    Zc = np.asarray(Z, float) - np.mean(Z, 0)
    s = np.linalg.svd(Zc, compute_uv=False) ** 2
    s = s[s > 1e-14]
    if len(s) < 2:
        return {"eff_rank": np.nan, "pca_pr": np.nan, "spec_entropy": np.nan}
    p = s / s.sum()
    return {"eff_rank": float(np.exp(-(p * np.log(p)).sum())),
            "pca_pr": float(s.sum() ** 2 / (s ** 2).sum()),
            "spec_entropy": float(-(p * np.log(p)).sum())}


def conditional_battery(X, y):
    out = {"cond_twonn": cond_twonn(X, y), "cond_mle_k5": cond_mle(X, y, 5),
           "cond_mle_k10": cond_mle(X, y, 10)}
    R = residuals(X, y)
    for k, v in spectral(R).items():
        out["within_" + k] = v
    for k, v in spectral(X).items():
        out["global_" + k] = v
    pc = cond_twonn(X, y, pooled=False)
    vals = [v for v in pc.values() if np.isfinite(v)]
    out["per_class_twonn_median"] = float(np.median(vals)) if vals else np.nan
    out["per_class_twonn_iqr"] = (float(np.percentile(vals, 75) - np.percentile(vals, 25))
                                  if len(vals) > 3 else np.nan)
    out["global_twonn"] = geom.twonn(X)
    return out


# ---------------------------------------------------------------- validation
def cloud(d, K, m, sigma, seed=0, amb=AMB):
    g = np.random.default_rng(seed)
    P = g.normal(size=(d, amb))
    cen = g.normal(size=(K, amb)) * 3
    X = np.vstack([cen[k] + (g.normal(size=(m, d)) @ P) * sigma for k in range(K)])
    return X, np.repeat(np.arange(K), m)


def validate():
    """Are the conditional estimators calibrated at the m real networks actually have?"""
    print("=== VALIDATION: conditional estimators at matched points-per-class ===")
    print("    K = 23 classes (as at p=23), true class-conditional d known, 3 seeds\n")
    rows = []
    for m in (8, 17, 32, 64):
        for d in (2, 5, 8):
            for s in (0, 1, 2):
                X, y = cloud(d, 23, m, 0.05, seed=s)
                r = conditional_battery(X, y)
                rows.append({"m": m, "true_d": d, "seed": s, **r})
    V = pd.DataFrame(rows)
    V.to_csv(OUT / "v2_conditional_validation.csv", index=False)
    P = V.pivot_table(index="m", columns="true_d", values="cond_twonn")
    print("  conditional TwoNN (rows = points per class, cols = true d):")
    print("   m \\ d " + "".join(f"{d:>9}" for d in P.columns))
    for m_, r in P.iterrows():
        print(f"   {m_:>5} " + "".join(f"{v:>9.2f}" for v in r.values))
    at17 = V[V.m == 17]
    err = float(np.median(np.abs(at17.cond_twonn - at17.true_d)))
    bias = float(np.median(at17.cond_twonn - at17.true_d))
    print(f"\n  at m = 17 (the corpus operating point): median |error| = {err:.2f}, "
          f"median bias = {bias:+.2f}")
    # sensitivity: how much does the estimate move per unit of true d at m = 17?
    sl = np.polyfit(at17.true_d, at17.cond_twonn, 1)[0]
    print(f"  slope vs true d at m = 17: {sl:.3f}  (1.0 would be perfectly calibrated)")
    return V, err, sl


def invariance_check():
    """Conditional estimators must be EXACTLY invariant to the alpha intervention."""
    print("\n=== INVARIANCE: conditional estimators under the centroid sweep ===")
    from tools.e1_centroid import decompose, recompose
    X, y = cloud(D_TRUE, 23, 17, 0.05, seed=0)
    mu, bet, wit = decompose(X, y)
    rows = []
    for a in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        Z = recompose(mu, bet, wit, a, 1.0, norm="trace")
        rows.append({"alpha": a, **conditional_battery(Z, y)})
    I = pd.DataFrame(rows)
    I.to_csv(OUT / "v2_conditional_invariance.csv", index=False)
    for c in ("cond_twonn", "cond_mle_k10", "within_eff_rank", "global_twonn"):
        v = I[c].values
        print(f"  {c:22s} {v.min():7.3f} -> {v.max():7.3f}   spread "
              f"{v.max()/max(abs(v.min()),1e-9):.4f}x")
    return I


# ---------------------------------------------------------------- the real question
def trajectory():
    print("\n=== TRAJECTORY: does within-class geometry compress through grokking? ===", flush=True)
    inv = pd.read_csv(OUT / "_inventory.csv")
    sel = select(inv)
    rows, t0 = [], time.time()
    for i, (_, rec) in enumerate(sel.iterrows()):
        try:
            r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
        except Exception as e:
            print(f"  skip {rec['dir'][:40]}: {type(e).__name__}"); continue
        n = len(r.epochs); tg = r.t_grok()
        marks = {"init": 0, "pre": max(1, tg // 2), "t_grok": tg if tg > 0 else n // 2,
                 "post": (tg + n) // 2 if tg > 0 else (3 * n) // 4, "final": n - 1}
        y_all = r.Ytr.numpy()
        sub = np.sort(np.random.default_rng(0).choice(len(y_all), min(400, len(y_all)),
                                                      replace=False))
        y = y_all[sub]
        for name, idx in marks.items():
            r._load(r.flat(idx)); r.model.eval()
            with torch.no_grad():
                r.model(r.Xtr, keep_activations=True)
                H = r.model.activations[-2].numpy()[sub]
            rows.append({"dir": rec["dir"], "seed": int(rec["seed"]), "stage": name,
                         "epoch": int(r.epochs[idx]),
                         "points_per_class": len(y) / len(np.unique(y)),
                         **conditional_battery(H, y)})
        print(f"  [{i+1}/{len(sel)}] seed {int(rec['seed']):>2} ({time.time()-t0:.0f}s)", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "v2_conditional_trajectory.csv", index=False)

    order = ["init", "pre", "t_grok", "post", "final"]
    cols = ["cond_twonn", "cond_mle_k10", "within_eff_rank", "within_pca_pr",
            "per_class_twonn_median", "global_twonn"]
    print(f"\n  median over {T.dir.nunique()} seeds:")
    print(f"  {'stage':<9}" + "".join(f"{c:>24}" for c in cols))
    for st in order:
        g = T[T.stage == st]
        print(f"  {st:<9}" + "".join(f"{g[c].median():>24.3f}" for c in cols))

    print("\n  per-run change init -> final (median, with a paired bootstrap 95% interval):")
    rng = np.random.default_rng(0)
    summ = []
    for c in cols:
        ch = []
        for d, g in T.groupby("dir"):
            a = g[g.stage == "init"][c]
            b = g[g.stage == "final"][c]
            if len(a) and len(b) and np.isfinite(a.iloc[0]) and np.isfinite(b.iloc[0]):
                ch.append((b.iloc[0] - a.iloc[0], (b.iloc[0] - a.iloc[0]) / abs(a.iloc[0])))
        if not ch:
            continue
        abs_, rel = np.array([x[0] for x in ch]), np.array([x[1] for x in ch])
        bs = [np.median(rng.choice(abs_, len(abs_), replace=True)) for _ in range(4000)]
        summ.append({"metric": c, "n": len(abs_), "init": float(T[T.stage == 'init'][c].median()),
                     "final": float(T[T.stage == 'final'][c].median()),
                     "abs_change": float(np.median(abs_)),
                     "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                     "rel_change": float(np.median(rel))})
        print(f"    {c:<24} {np.median(abs_):+7.3f}  "
              f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  "
              f"({100*np.median(rel):+.1f}%)")
    pd.DataFrame(summ).to_csv(OUT / "v2_conditional_summary.csv", index=False)
    return T


if __name__ == "__main__":
    t0 = time.time()
    validate()
    invariance_check()
    trajectory()
    print(f"\nwrote v2_conditional_*.csv   ({time.time()-t0:.0f}s)")
