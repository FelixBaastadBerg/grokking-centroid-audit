"""
Item 7 -- how much of the attribution is an artefact of the path we chose?

The attribution inverts ONE statistic (label silhouette) on ONE counterfactual path (centroids
scaled by alpha around the converged within-class geometry). Both are choices. A different matching
statistic picks a different alpha interval, and residuals taken from a different checkpoint give a
different path between the same endpoints. If the attribution percentage swings with those choices,
the headline is about the path rather than about the representation.

So we vary both:

  RESIDUAL SOURCE   within-class geometry taken from init, mid-plateau, grokking onset or
                    convergence, with the centroid direction always the converged one
  MATCHING STATISTIC  label silhouette, NC1, centroid RMS distance, nearest-neighbour gap ratio,
                    median centroid-to-centroid distance, and the effective rank of the centroid
                    covariance -- six ways to say "the counterfactual now looks as clustered as
                    the real representation did"

and report the attribution for each combination, plus its spread. Analysis-only.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from sklearn.metrics import silhouette_score
from analysis import geom
from analysis.lsrun import LandscapeRun
from tools.e1_centroid import measure
from tools.b3_seed_alpha import select
from tools.b2_attribution import real_trajectory_change

OUT = REPO / "RESULTS"
ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
N_SUB = 400
MATCHERS = ["sil_true", "nc1", "centroid_rms", "gap_ratio", "centroid_dist_med",
            "centroid_eff_rank"]


def match_stats(X, y):
    """Six different ways of saying how separated the classes are."""
    X = np.asarray(X, float)
    cls = np.unique(y)
    M = np.stack([X[y == c].mean(0) for c in cls])
    W = np.concatenate([X[y == c] - M[i] for i, c in enumerate(cls)])
    out = {}
    try:
        out["sil_true"] = float(silhouette_score(X, y))
    except Exception:
        out["sil_true"] = np.nan
    sw = float((W ** 2).sum(1).mean())
    Mc = M - M.mean(0)
    sb = float((Mc ** 2).sum(1).mean())
    out["nc1"] = sw / max(sb, 1e-12)
    out["centroid_rms"] = float(np.sqrt(sb))
    out["gap_ratio"] = geom.clusteredness(X)["gap_ratio"]
    from scipy.spatial.distance import pdist
    out["centroid_dist_med"] = float(np.median(pdist(M))) if len(M) > 2 else np.nan
    s = np.linalg.svd(Mc, compute_uv=False) ** 2
    s = s[s > 1e-14]
    p = s / s.sum() if len(s) else np.array([1.0])
    out["centroid_eff_rank"] = float(np.exp(-(p * np.log(p)).sum()))
    return out


def invert(curve_x, curve_a, target):
    """Interpolate the alpha at which a (monotone) matching statistic hits `target`."""
    x = np.asarray(curve_x, float); a = np.asarray(curve_a, float)
    ok = np.isfinite(x) & np.isfinite(a)
    x, a = x[ok], a[ok]
    if len(x) < 3:
        return np.nan
    o = np.argsort(x)
    return float(np.interp(target, x[o], a[o]))


def main():
    t0 = time.time()
    inv_ = pd.read_csv(OUT / "_inventory.csv")
    sel = select(inv_)
    T = real_trajectory_change()
    T = T[T.dir.str.contains("task-add__p-23")]
    print("=== Item 7: path-dependence of the attribution ===")
    print(f"    {len(sel)} seeds x 4 residual sources x {len(MATCHERS)} matching statistics\n",
          flush=True)

    rows, real_rows = [], []
    for i, (_, rec) in enumerate(sel.iterrows()):
        r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
        n = len(r.epochs); tg = r.t_grok()
        marks = {"init": 0, "plateau": max(1, tg // 2), "onset": tg if tg > 0 else n // 2,
                 "final": n - 1}
        y_all = r.Ytr.numpy()
        sub = np.sort(np.random.default_rng(0).choice(len(y_all), min(N_SUB, len(y_all)),
                                                      replace=False))
        y = y_all[sub]
        H = {}
        for name, k in marks.items():
            r._load(r.flat(k)); r.model.eval()
            with torch.no_grad():
                r.model(r.Xtr, keep_activations=True)
                H[name] = r.model.activations[-2].numpy()[sub]
            real_rows.append({"dir": rec["dir"], "stage": name, **match_stats(H[name], y)})

        # centroid direction always from convergence; residuals from each source
        Xf = H["final"]
        cls = np.unique(y)
        Mf = {c: Xf[y == c].mean(0) for c in cls}
        muf = Xf.mean(0)
        bet = np.stack([Mf[c] for c in y]) - muf
        for src in marks:
            Xs = H[src]
            Ms = {c: Xs[y == c].mean(0) for c in cls}
            wit = Xs - np.stack([Ms[c] for c in y])
            for a in ALPHAS:
                Z = muf + a * bet + wit
                Zc = Z - Z.mean(0)
                Z = Zc / (np.sqrt((Zc ** 2).sum(1).mean()) + 1e-12)
                rows.append({"dir": rec["dir"], "seed": int(rec["seed"]), "resid_src": src,
                             "alpha": a, **match_stats(Z, y), **measure(Z, y)})
        print(f"  [{i+1}/{len(sel)}] ({time.time()-t0:.0f}s)", flush=True)

    P = pd.DataFrame(rows); P.to_csv(OUT / "v2_path_sweeps.csv", index=False)
    RS = pd.DataFrame(real_rows); RS.to_csv(OUT / "v2_path_real.csv", index=False)

    # observed training-time change, per estimator, on the same runs
    obs = {}
    for est, col in (("ph_dim", "ph_dim"), ("eff_rank", "eff_rank"), ("twonn", "twonn")):
        v = ((T[f"pre_{est}"] - T[f"post_{est}"]) / T[f"pre_{est}"].abs()).dropna()
        obs[est] = float(v.median()) if len(v) else np.nan

    print("\n  attribution for PH-dimension, by residual source x matching statistic")
    print(f"  {'resid source':<12}" + "".join(f"{m[:11]:>13}" for m in MATCHERS))
    out = []
    for src in ("init", "plateau", "onset", "final"):
        line = f"  {src:<12}"
        for mt in MATCHERS:
            vals = []
            for d, g in P[P.resid_src == src].groupby("dir"):
                g = g.sort_values("alpha")
                lo_t = RS[(RS.dir == d) & (RS.stage == "init")][mt]
                hi_t = RS[(RS.dir == d) & (RS.stage == "final")][mt]
                if not len(lo_t) or not len(hi_t):
                    continue
                a0 = invert(g[mt].values, g.alpha.values, float(lo_t.iloc[0]))
                a1 = invert(g[mt].values, g.alpha.values, float(hi_t.iloc[0]))
                v0 = float(np.interp(a0, g.alpha.values, g.ph_dim.values))
                v1 = float(np.interp(a1, g.alpha.values, g.ph_dim.values))
                if np.isfinite(v0) and abs(v0) > 1e-12:
                    vals.append((v0 - v1) / abs(v0))
            att = (np.median(vals) / obs["ph_dim"]) if vals and abs(obs["ph_dim"]) > 1e-9 else np.nan
            out.append({"resid_src": src, "matcher": mt, "n": len(vals),
                        "matched_change": float(np.median(vals)) if vals else np.nan,
                        "attribution": float(att)})
            line += f"{att:>12.0%}" if np.isfinite(att) else f"{'--':>13}"
        print(line)
    A = pd.DataFrame(out); A.to_csv(OUT / "v2_path_attribution.csv", index=False)

    v = A.attribution.dropna()
    print(f"\n  spread over all {len(v)} path/matcher combinations: median {v.median():.0%}, "
          f"range [{v.min():.0%}, {v.max():.0%}], IQR [{v.quantile(.25):.0%}, "
          f"{v.quantile(.75):.0%}]")
    print(f"  by matcher (median over residual sources): "
          + ", ".join(f"{m}: {A[A.matcher==m].attribution.median():.0%}" for m in MATCHERS))
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
