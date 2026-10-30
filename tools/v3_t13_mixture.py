"""
T1.3 -- a mixture account of PH-dimension: derivation and numerical validation.

THE CLAIM. The repo's PH-dimension fits the slope s of log E(n) against log n, where
E(n) = sum of MST edge lengths^alpha over an n-point subsample, and reports
d_hat = alpha/(1-s). For a K-component mixture (m points per component, within-component
dimension d, within scale sigma, centroid scale R) the MST of a well-separated union
decomposes as

    E(n)  =  E_within(n) + E_between(n)
          ~  C_d sigma^alpha K^{alpha/d} n^{1-alpha/d}   +   R^alpha B_K(n),

with exactly K-1 between-component connector edges. Writing u(n) = E_between/E_within, the
local log-log slope is s = (1-alpha/d)/(1+u), hence the fitted dimension

    d_hat = alpha (1+u) / (u + alpha/d)        (*)

which interpolates d (u -> 0) to alpha (u -> infinity): a fully collapsed mixture reads
EXACTLY alpha, which is why the real-activation sweeps bottom out at PH-dim ~ 1.0 with the
conventional alpha = 1. Asymptotically u ~ (B_K/(C_d K)) gamma^alpha m^{-(1-alpha/d)}, which
derives the existence of the paper's empirical collapse variable u = gamma m^-kappa; at finite
m the connector lengths shrink as clusters fill out (boundary attachment), so the effective m
exponent is smaller than 1-alpha/d -- measured here rather than hand-waved.

VALIDATION on synthetic mixtures over (d, K, m, gamma):
  1. the MST cross-edge count is ~K-1 once components separate;
  2. E_within follows n^(1-alpha/d); E_between is nearly flat in n;
  3. (*) with u MEASURED from the same MSTs predicts the estimator's d_hat;
  4. the a-priori law u = c gamma^a m^-b: fitted (a, b) against the asymptotic (alpha, 1-alpha/d).

Output: RESULTS/v3_mixture_grid.csv, v3_mixture_summary.txt
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import coo_matrix

OUT = REPO / "RESULTS"
ALPHA = 1.0
AMB = 60
DS = (2, 5, 8)
KS = (8, 23)
MS = (8, 16, 32, 64)
GAMMAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SEEDS = (0, 1)


def cloud(d, K, m, gamma, seed):
    """b1_gride-style: centres ~ N(0,9 I), within-scale sigma set from gamma = 1/(sigma sqrt d)."""
    sigma = 1.0 / (gamma * np.sqrt(d))
    g = np.random.default_rng(seed)
    P = g.normal(size=(d, AMB))
    cen = g.normal(size=(K, AMB)) * 3
    X = np.vstack([cen[k] + (g.normal(size=(m, d)) @ P) * sigma for k in range(K)])
    return X, np.repeat(np.arange(K), m)


def mst_split(X, y):
    """(E_within, E_between, n_cross) of the MST at alpha = ALPHA."""
    D = squareform(pdist(X))
    T = coo_matrix(minimum_spanning_tree(D))
    w = T.data ** ALPHA
    cross = y[T.row] != y[T.col]
    return float(w[~cross].sum()), float(w[cross].sum()), int(cross.sum())


def analyse(X, y, seed=0, n_repeats=5):
    """geom.ph_dim's exact subsample schedule and fit, with the within/between split logged."""
    n = len(X)
    hi = min(n, 1000)
    lo = max(50, hi // 5)
    n_list = [m for m in range(lo, hi + 1, max(1, (hi - lo) // 10)) if 5 <= m <= n]
    if len(n_list) < 4:
        return None
    rng = np.random.default_rng(seed)
    rows = []
    for m_ in n_list:
        ew, eb, nc = [], [], []
        for _ in range(n_repeats):
            idx = rng.choice(n, m_, replace=False)
            a, b, c = mst_split(X[idx], y[idx])
            ew.append(a); eb.append(b); nc.append(c)
        rows.append({"n_sub": m_, "E_w": np.mean(ew), "E_b": np.mean(eb),
                     "n_cross": np.mean(nc)})
    S = pd.DataFrame(rows)
    xs, ys = np.log(S.n_sub.values), np.log((S.E_w + S.E_b).values)
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope = np.linalg.lstsq(A, ys, rcond=None)[0][0]
    d_hat = ALPHA / (1 - slope) if slope < 1 - 1e-6 else np.nan
    # within-only slope (the Steele check)
    sw = np.linalg.lstsq(A, np.log(S.E_w.values), rcond=None)[0][0]
    sb = np.linalg.lstsq(A, np.log(np.maximum(S.E_b.values, 1e-30)), rcond=None)[0][0]
    # measured u at the geometric mid of the window, and the closed-form prediction
    mid = len(S) // 2
    u_mid = float(S.E_b.iloc[mid] / S.E_w.iloc[mid])
    return {"d_hat": float(d_hat), "slope": float(slope), "slope_w": float(sw),
            "slope_b": float(sb), "u_mid": u_mid,
            "n_cross_mean": float(S.n_cross.mean()),
            "u_series": S}


def predict_dhat_from_u(u, d):
    return ALPHA * (1 + u) / (u + ALPHA / d)


def main():
    rows, t0 = [], time.time()
    for d in DS:
        for K in KS:
            for m in MS:
                for gamma in GAMMAS:
                    for sd in SEEDS:
                        X, y = cloud(d, K, m, gamma, sd)
                        r = analyse(X, y, seed=sd)
                        if r is None:
                            continue
                        S = r.pop("u_series")
                        # model prediction: fit the two-term model through the same lsq.
                        # A_hat from the largest window size, B_hat = median E_b.
                        n_big = S.n_sub.iloc[-1]
                        A_hat = S.E_w.iloc[-1] / n_big ** (1 - ALPHA / d)
                        B_hat = float(S.E_b.median())
                        xs = np.log(S.n_sub.values)
                        ys_model = np.log(A_hat * S.n_sub.values ** (1 - ALPHA / d) + B_hat)
                        Amat = np.vstack([xs, np.ones_like(xs)]).T
                        sl_model = np.linalg.lstsq(Amat, ys_model, rcond=None)[0][0]
                        d_model = ALPHA / (1 - sl_model) if sl_model < 1 - 1e-6 else np.nan
                        rows.append({"d": d, "K": K, "m": m, "gamma": gamma, "seed": sd,
                                     **r, "d_pred_closedform": predict_dhat_from_u(r["u_mid"], d),
                                     "d_pred_model": float(d_model)})
            print(f"  d={d} K={K} done ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "v3_mixture_grid.csv", index=False)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "v3_mixture_grid.csv", index=False)

    lines = []
    def say(s):
        print(s); lines.append(s)

    sep = D[D.gamma >= 4]
    say("=== T1.3 mixture-theorem validation ===")
    say(f"1. cross edges ~ K-1 when separated (gamma >= 4): "
        f"median n_cross/(K-1) = {float((sep.n_cross_mean/(sep.K-1)).median()):.2f}")
    say(f"2. Steele within-slope vs 1 - alpha/d, by d:")
    for d, g in D.groupby("d"):
        say(f"     d={d}:  slope_w median {g.slope_w.median():+.3f}  "
            f"(asymptotic {1-ALPHA/d:+.3f});  slope_b median {g.slope_b.median():+.3f} (~0)")
    ok = np.isfinite(D.d_hat) & np.isfinite(D.d_pred_model)
    r2m = 1 - ((D.d_hat - D.d_pred_model)[ok] ** 2).sum() / ((D.d_hat[ok] - D.d_hat[ok].mean()) ** 2).sum()
    okc = np.isfinite(D.d_hat) & np.isfinite(D.d_pred_closedform)
    r2c = 1 - ((D.d_hat - D.d_pred_closedform)[okc] ** 2).sum() / ((D.d_hat[okc] - D.d_hat[okc].mean()) ** 2).sum()
    say(f"3. estimator d_hat predicted by the two-term model fit: R^2 = {r2m:.3f} "
        f"(n = {int(ok.sum())})")
    say(f"   closed-form (*) with u measured at the window midpoint:  R^2 = {r2c:.3f}")
    say(f"   endpoint check: most-collapsed cells (gamma=16, m=8): "
        f"d_hat median = {D[(D.gamma==16)&(D.m==8)].d_hat.median():.2f} (theory -> alpha = {ALPHA})")
    # 4. a-priori u-law: log u = log c + a log gamma - b log m (per d)
    say("4. u-law fit  u = c gamma^a m^-b  (theory: a = alpha = 1, b = 1 - alpha/d):")
    for d, g in D.groupby("d"):
        g = g[np.isfinite(g.u_mid) & (g.u_mid > 0)]
        Xd = np.c_[np.log(g.gamma), -np.log(g.m), np.ones(len(g))]
        coef, *_ = np.linalg.lstsq(Xd, np.log(g.u_mid), rcond=None)
        pred = Xd @ coef
        r2 = 1 - ((np.log(g.u_mid) - pred) ** 2).sum() / ((np.log(g.u_mid) - np.log(g.u_mid).mean()) ** 2).sum()
        say(f"     d={d}:  a = {coef[0]:.2f}  b = {coef[1]:.2f} "
            f"(asymptotic b = {1-ALPHA/d:.2f})  R^2 = {r2:.3f}")
    (OUT / "v3_mixture_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"\nwrote v3_mixture_grid.csv {D.shape}, v3_mixture_summary.txt ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
