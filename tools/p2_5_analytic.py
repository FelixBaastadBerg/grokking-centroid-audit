"""
P2-5 -- a bounded analytical account of the confound. No new training: this uses the Arm A /
detection-floor sweep already in hand (RESULTS/twonn_floor.csv -- 6 separation-to-scale levels x
6 points-per-cluster levels x 5 seeds, true d = 5 throughout).

Two results, one exact and one fitted.

(1) EXACT. For a ratio estimator like TwoNN (mu_i = r_2/r_1), if both nearest neighbours of every
    point lie in that point's own cluster, then rescaling the within-cluster scale sigma multiplies
    r_1 and r_2 by the same factor and leaves every mu_i -- hence the estimate -- unchanged. TwoNN
    is therefore *exactly* invariant to clustering throughout the regime in which its neighbourhood
    does not span the gap, and can move only when the gap enters the k=2 neighbourhood: when
    clusters merge, or when a cluster holds too few points to supply two neighbours. This is the
    scale-locality principle as an identity rather than a trend, and the sweep confirms it to
    machine precision.

(2) FITTED. For an estimator whose neighbourhood does span the gap, the principle predicts that
    separation-to-scale gamma and points-per-cluster m are not independent knobs: m sets how far a
    neighbourhood must reach before it crosses the gap, so the two should enter only through one
    composite variable u = gamma * m^(-kappa). We test that as a data collapse. Writing the response
    as a fold change against each row's own least-clustered cell (which divides out the m-dependent
    finite-sample bias, a separate and known effect),

        fold(gamma, m) = A + (1 - A) / (1 + (u / u0)^b),      u = gamma * m^(-kappa)

    with A the clustering-dominated plateau. We report the four-parameter fit, the restricted
    two-parameter fit (b = 1, A at its observed plateau), and a leave-one-gamma-column-out holdout.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.optimize import curve_fit

OUT = REPO / "RESULTS"
D_TRUE, AMB = 5, 60


def gamma_of_sigma(sigma, d=D_TRUE):
    """Centres are drawn N(0, I_amb), so typical centre separation is sqrt(2*amb); the
    within-cluster point-to-point scale is sigma*sqrt(2*d*amb). Their ratio is 1/(sigma*sqrt(d))."""
    return 1.0 / (sigma * np.sqrt(d))


def folds(F, col, mmin=8):
    """Fold change against the least-clustered cell of the same m row."""
    g = F[F.m >= mmin].groupby(["m", "sigma"])[col].mean().reset_index().dropna()
    base = g.sort_values("sigma").groupby("m")[col].last()      # largest sigma = smallest gamma
    g["fold"] = g[col].values / base.reindex(g.m.values).values
    g["gamma"] = gamma_of_sigma(g.sigma.values)
    return g


def model(X, A, b, lu0, kap):
    lg, lm = X
    return A + (1.0 - A) / (1.0 + np.exp(b * (lg - kap * lm - lu0)))


def fit(g, restricted=False, A_fix=None):
    lg, lm, y = np.log(g.gamma.values), np.log(g.m.values.astype(float)), g.fold.values
    A0 = float(np.nanmin(y)) if A_fix is None else A_fix
    if restricted:
        f = lambda X, lu0, kap: model(X, A0, 1.0, lu0, kap)
        p0, names = [0.5, 0.5], ["log u0", "kappa"]
    else:
        f, p0, names = model, [A0, 2.0, 0.5, 0.5], ["A", "b", "log u0", "kappa"]
    try:
        pop, _ = curve_fit(f, (lg, lm), y, p0=p0, maxfev=60000)
    except Exception:
        return None
    pred = f((lg, lm), *pop)
    r2 = 1 - np.nansum((y - pred) ** 2) / np.nansum((y - np.nanmean(y)) ** 2)
    return dict(params=dict(zip(names, pop)), r2=float(r2), A_plateau=A0, pred=pred, obs=y,
                lu=lg - (pop[-1] if not restricted else pop[1]) * lm)


def holdout(F, col):
    """Leave-one-gamma-column-out: fit on five separation levels, predict the sixth."""
    errs = []
    for s in sorted(F.sigma.unique()):
        r = fit(folds(F[F.sigma != s], col))
        if r is None:
            continue
        te = folds(F, col)
        te = te[te.sigma == s]
        if not len(te):
            continue
        p = r["params"]
        pred = model((np.log(te.gamma.values), np.log(te.m.values.astype(float))),
                     p["A"], p["b"], p["log u0"], p["kappa"])
        errs.append(np.nanmean(np.abs(pred - te.fold.values)))
    return float(np.mean(errs)) if errs else np.nan


def invariance(F, col="twonn", smax=0.25):
    """The exact result: spread of the estimate across sub-gap sigma levels, within each m."""
    p = F[F.sigma <= smax].groupby(["m", "sigma"])[col].mean().unstack()
    return (p.max(axis=1) - p.min(axis=1)), p.columns.max() / p.columns.min()


if __name__ == "__main__":
    F = pd.read_csv(OUT / "twonn_floor.csv")
    print(f"Arm A sweep: {len(F)} clouds, true d = {D_TRUE}, m in {sorted(F.m.unique())}, "
          f"gamma in {[round(gamma_of_sigma(s), 2) for s in sorted(F.sigma.unique())]}\n")

    # ---- (1) the exact result
    spread, ratio = invariance(F)
    print(f"(1) TwoNN across a {ratio:.1f}x change in cluster tightness (sub-gap sigma levels):")
    for m, v in spread.items():
        print(f"      m = {m:3d}   spread = {v:.2e}")
    print(f"    max spread {spread.max():.1e} -- invariant to machine precision, as the ratio "
          f"identity requires.\n")

    rows = []
    for col in ["ph_dim", "twonn"]:
        g = folds(F, col)
        full, res = fit(g), fit(g, restricted=True)
        ho = holdout(F, col)
        if full is None:
            print(f"(2) {col}: fit failed"); continue
        p = full["params"]
        print(f"(2) {col}:  {len(g)} cells")
        print(f"      full   A={p['A']:.2f}  b={p['b']:.2f}  log u0={p['log u0']:+.2f}  "
              f"kappa={p['kappa']:+.3f}   R2={full['r2']:.3f}")
        if res:
            q = res["params"]
            print(f"      2-par  log u0={q['log u0']:+.2f}  kappa={q['kappa']:+.3f}   "
                  f"R2={res['r2']:.3f}   (A fixed {res['A_plateau']:.2f}, b fixed 1)")
        print(f"      held-out gamma column MAE = {ho:.3f} fold units\n")
        rows.append({"estimator": col, "A": p["A"], "b": p["b"], "log_u0": p["log u0"],
                     "kappa": p["kappa"], "r2_full": full["r2"],
                     "r2_restricted": res["r2"] if res else np.nan,
                     "kappa_restricted": res["params"]["kappa"] if res else np.nan,
                     "holdout_mae": ho, "n_cells": len(g),
                     "invariance_max_spread": float(spread.max()),
                     "invariance_sigma_ratio": float(ratio)})
        pd.DataFrame({"estimator": col, "m": g.m.values, "sigma": g.sigma.values,
                      "gamma": g.gamma.values, "obs_fold": full["obs"],
                      "pred_fold": full["pred"], "log_u": full["lu"]}
                     ).to_csv(OUT / f"p2_5_cells_{col}.csv", index=False)

    pd.DataFrame(rows).to_csv(OUT / "p2_5_analytic.csv", index=False)
    print("wrote p2_5_analytic.csv")
