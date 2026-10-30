"""
Items 5, 6, 8, 9 -- the covariance decomposition, the support-leakage axis, and the geometry
robustness sweeps. All synthetic plus retained activations; no training.

ITEM 5. Under the intervention x -> mu + alpha(mu_y - mu) + beta(x - mu_y), the total covariance
decomposes EXACTLY:

    Sigma_total(alpha, beta) = beta^2 Sigma_W + alpha^2 Sigma_B

because the residuals x - mu_y have zero mean within each class, so the cross term
E[(mu_y - mu)(x - mu_y)^T] vanishes by construction. Every spectral statistic is a function of the
eigenvalues of that sum, so their response to alpha is not estimator failure at all -- between-class
variance is legitimately part of the global covariance. The question for spectral metrics is
therefore one of INTERPRETATION: a fall in global effective rank is not evidence of within-class
compression unless Sigma_W is looked at separately. We check the identity numerically, then compute
the spectral statistics of Sigma_W and Sigma_B separately.

ITEM 6. Support leakage L_q = fraction of points whose first q neighbours include another class.
At L_q = 0 the exact proposition of the paper applies, so a ratio statistic on those q neighbours is
invariant; the claim here is only that L_q is the right single axis, which is an empirical question,
so we plot estimator response against it across very different constructions.

ITEMS 8 and 9. Centroid arrangements (random, simplex, circular, Fourier, equivariant) and
adversarial within-class geometries (anisotropic, per-class dimension, unequal density, unequal
size, nearly touching). Reported against L_q rather than case by case.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from analysis import geom
from tools.v2_conditional import residuals, spectral, cond_twonn

OUT = REPO / "RESULTS"
AMB, D_TRUE = 60, 5


# ---------------------------------------------------------------- item 6: L_q
def leakage(X, y, q=2):
    """Fraction of points whose first q neighbours are not all in the point's own class."""
    X = np.asarray(X, float)
    q = min(q, len(X) - 1)
    _, idx = NearestNeighbors(n_neighbors=q + 1).fit(X).kneighbors(X)
    nb = y[idx[:, 1:q + 1]]
    return float(np.mean((nb != y[:, None]).any(1)))


# ---------------------------------------------------------------- constructions
def _basis(d, amb, g):
    return g.normal(size=(d, amb))


def centroids(kind, K, amb, g, scale=3.0):
    """Item 8: between-class arrangements at matched scale."""
    if kind == "random":
        C = g.normal(size=(K, amb))
    elif kind == "simplex":
        C = np.eye(K, amb) - 1.0 / K
    elif kind == "circular":
        t = 2 * np.pi * np.arange(K) / K
        C = np.zeros((K, amb)); C[:, 0], C[:, 1] = np.cos(t), np.sin(t)
    elif kind == "fourier":
        t = 2 * np.pi * np.arange(K) / K
        C = np.zeros((K, amb))
        for j, f in enumerate([1, 2, 3, 5]):
            C[:, 2 * j], C[:, 2 * j + 1] = np.cos(f * t), np.sin(f * t)
    elif kind == "equivariant":
        # a genuine regular-representation embedding of Z_K: rows are shifted copies, so the
        # arrangement carries the group action rather than merely being spread out
        t = 2 * np.pi * np.outer(np.arange(K), np.arange(1, K // 2 + 1)) / K
        C = np.zeros((K, amb))
        w = min(K // 2, amb // 2)
        C[:, :2 * w] = np.concatenate([np.cos(t[:, :w]), np.sin(t[:, :w])], 1)
    else:
        raise ValueError(kind)
    C = C - C.mean(0)
    n = np.linalg.norm(C, axis=1).mean()
    return C / max(n, 1e-12) * scale


def make(kind_c="random", kind_w="iso", K=8, m=32, sigma=0.05, d=D_TRUE, amb=AMB, seed=0,
         scale=3.0):
    """Items 8/9: a cloud with a chosen between-class arrangement and within-class geometry."""
    g = np.random.default_rng(seed)
    C = centroids(kind_c, K, amb, g, scale)
    blocks, ys = [], []
    for k in range(K):
        if kind_w == "iso":
            P, mk, s = _basis(d, amb, g), m, sigma
        elif kind_w == "aniso":
            P = _basis(d, amb, g) * (np.geomspace(1, 0.05, d)[:, None]); mk, s = m, sigma
        elif kind_w == "perclass_d":
            dk = 2 + (k % 7); P, mk, s = _basis(dk, amb, g), m, sigma
        elif kind_w == "density":
            P, mk, s = _basis(d, amb, g), m, sigma * (0.2 + 1.6 * (k % 5) / 4)
        elif kind_w == "size":
            P, mk, s = _basis(d, amb, g), max(4, int(m * (0.2 + 1.6 * (k % 5) / 4))), sigma
        elif kind_w == "touching":
            P, mk, s = _basis(d, amb, g), m, sigma * 8
        else:
            raise ValueError(kind_w)
        blocks.append(C[k] + (g.normal(size=(mk, P.shape[0])) @ P) * s)
        ys.append(np.full(mk, k))
    return np.vstack(blocks), np.concatenate(ys)


def battery(X, y):
    R = residuals(X, y)
    out = {"ph_dim": geom.ph_dim(X)[0], "twonn": geom.twonn(X), "corr_dim": geom.correlation_dim(X)[0],
           "cond_twonn": cond_twonn(X, y)}
    for k, v in spectral(X).items():
        out["global_" + k] = v
    for k, v in spectral(R).items():
        out["within_" + k] = v
    for q in (2, 5, 10):
        out[f"L{q}"] = leakage(X, y, q)
    return out


# ---------------------------------------------------------------- item 5
def item5():
    print("=== Item 5: Sigma_total(alpha,beta) = beta^2 Sigma_W + alpha^2 Sigma_B ===")
    from tools.e1_centroid import decompose
    X, y = make(K=8, m=32, sigma=0.15, seed=0)
    mu, bet, wit = decompose(X, y)
    SW = np.cov(wit, rowvar=False)
    SB = np.cov(bet, rowvar=False)
    worst, rows = 0.0, []
    for a in (0.0, 0.5, 1.0, 2.0, 4.0):
        for b in (0.5, 1.0, 2.0):
            Z = mu + a * bet + b * wit
            S = np.cov(Z - Z.mean(0), rowvar=False)
            pred = b ** 2 * SW + a ** 2 * SB
            err = float(np.linalg.norm(S - pred) / max(np.linalg.norm(S), 1e-12))
            worst = max(worst, err)
            rows.append({"alpha": a, "beta": b, "rel_err": err})
    print(f"  identity verified numerically: worst relative error over the (alpha,beta) grid = "
          f"{worst:.2e}")
    pd.DataFrame(rows).to_csv(OUT / "v2_cov_identity.csv", index=False)

    # spectra of the parts, and what each implies for the global statistics
    out = []
    for name, S in (("Sigma_W", SW), ("Sigma_B", SB)):
        w = np.linalg.eigvalsh(S); w = w[w > 1e-14]; p = w / w.sum()
        out.append({"matrix": name, "rank": int(len(w)),
                    "eff_rank": float(np.exp(-(p * np.log(p)).sum())),
                    "pca_pr": float(w.sum() ** 2 / (w ** 2).sum()),
                    "spec_entropy": float(-(p * np.log(p)).sum())})
    S5 = pd.DataFrame(out)
    print(S5.round(3).to_string(index=False))
    print("  Sigma_B has rank <= K-1 by construction, so adding it to Sigma_W can only pull the")
    print("  global spectrum towards a low-rank one as alpha grows -- an identity, not a failure.")
    S5.to_csv(OUT / "v2_cov_parts.csv", index=False)
    return worst


# ---------------------------------------------------------------- items 6, 8, 9
def sweeps():
    print("\n=== Items 6/8/9: response against support leakage, across constructions ===",
          flush=True)
    rows, t0 = [], time.time()
    grid_sigma = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5]
    # item 8: between-class arrangement, at matched centroid scale
    for kc in ("random", "simplex", "circular", "fourier", "equivariant"):
        for sc in (1.5, 3.0, 6.0):
            for s in (0, 1, 2):
                for sig in grid_sigma:
                    X, y = make(kind_c=kc, K=8, m=32, sigma=sig, seed=s, scale=sc)
                    rows.append({"family": "centroid_shape", "centroids": kc, "within": "iso",
                                 "scale": sc, "sigma": sig, "seed": s, **battery(X, y)})
    print(f"  item 8 done ({time.time()-t0:.0f}s)", flush=True)
    # item 9: adversarial within-class geometry
    for kw in ("iso", "aniso", "perclass_d", "density", "size", "touching"):
        for s in (0, 1, 2):
            for sig in grid_sigma:
                X, y = make(kind_c="random", kind_w=kw, K=8, m=32, sigma=sig, seed=s)
                rows.append({"family": "within_geometry", "centroids": "random", "within": kw,
                             "scale": 3.0, "sigma": sig, "seed": s, **battery(X, y)})
    print(f"  item 9 done ({time.time()-t0:.0f}s)", flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "v2_geometry_sweeps.csv", index=False)

    # fold change per (family, construction, scale, seed), against the least-clustered cell
    def folds(g, est):
        v = g.sort_values("sigma")[est].values
        base = v[-1]
        return v / base if np.isfinite(base) and abs(base) > 1e-9 else v * np.nan

    frows = []
    keys = ["family", "centroids", "within", "scale", "seed"]
    for k, g in S.groupby(keys):
        g = g.sort_values("sigma")
        rec = dict(zip(keys, k))
        for est in ("ph_dim", "twonn", "corr_dim", "cond_twonn", "global_eff_rank",
                    "within_eff_rank"):
            v = g[est].values
            base = v[-1]
            if np.isfinite(base) and abs(base) > 1e-9:
                for i, sg in enumerate(g.sigma.values):
                    frows.append({**rec, "sigma": sg, "estimator": est,
                                  "fold": float(base / v[i]) if np.isfinite(v[i]) and abs(v[i]) > 1e-9 else np.nan,
                                  "L2": float(g.L2.values[i]), "L5": float(g.L5.values[i]),
                                  "L10": float(g.L10.values[i])})
    F = pd.DataFrame(frows)
    F.to_csv(OUT / "v2_geometry_folds.csv", index=False)

    print("\n  item 8 -- does the between-class ARRANGEMENT matter, at matched scale?")
    print(f"  {'centroids':<13}{'L2 at sigma=0.05':>18}{'PH-dim fold':>14}{'TwoNN fold':>13}")
    for kc, g in F[(F.family == "centroid_shape") & (F.sigma == 0.05) & (F.scale == 3.0)
                   ].groupby("centroids"):
        ph = g[g.estimator == "ph_dim"].fold.median()
        tw = g[g.estimator == "twonn"].fold.median()
        print(f"  {kc:<13}{g.L2.median():>18.3f}{ph:>14.2f}{tw:>13.2f}")

    print("\n  item 9 -- adversarial within-class geometry, at sigma = 0.05")
    print(f"  {'within':<13}{'L2':>8}{'PH-dim fold':>14}{'TwoNN fold':>13}{'cond TwoNN fold':>18}")
    for kw, g in F[(F.family == "within_geometry") & (F.sigma == 0.05)].groupby("within"):
        ph = g[g.estimator == "ph_dim"].fold.median()
        tw = g[g.estimator == "twonn"].fold.median()
        ct = g[g.estimator == "cond_twonn"].fold.median()
        print(f"  {kw:<13}{g.L2.median():>8.3f}{ph:>14.2f}{tw:>13.2f}{ct:>18.2f}")

    print("\n  item 6 -- is L_q the single axis? TwoNN fold binned by L2, pooled over ALL "
          "constructions")
    F["L2bin"] = pd.cut(F.L2, [-0.001, 0.001, 0.05, 0.15, 0.35, 0.65, 1.001])
    print(f"  {'L2 bin':<16}{'n cells':>9}{'TwoNN fold':>13}{'IQR':>18}{'PH-dim fold':>14}")
    for b, g in F.groupby("L2bin", observed=True):
        tw = g[g.estimator == "twonn"].fold
        ph = g[g.estimator == "ph_dim"].fold
        if not len(tw):
            continue
        print(f"  {str(b):<16}{len(tw):>9}{tw.median():>13.2f}"
              f"   [{tw.quantile(.25):>5.2f},{tw.quantile(.75):>5.2f}]{ph.median():>14.2f}")
    return S, F


if __name__ == "__main__":
    t0 = time.time()
    item5()
    sweeps()
    print(f"\nwrote v2_cov_*.csv, v2_geometry_*.csv   ({time.time()-t0:.0f}s)")
