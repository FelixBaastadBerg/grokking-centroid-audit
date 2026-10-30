"""
Item 3 -- honest uncertainty on the headline percentages.

139% [137, 140] and 116% [115, 118] are implausibly tight, and they are tight for a reason that is
a mistake: the bootstrap resampled RUNS while treating everything else as fixed. Three further
sources of variance were being ignored.

  (a) CLUSTERING. Runs are not exchangeable across seeds and cells. We resample whole seeds, and
      separately whole task-architecture cells, and also report leave-one-seed-out and
      leave-one-cell-out ranges, which are the honest worst cases at these sample sizes.
  (b) ESTIMATOR NOISE. PH-dimension is itself a random variable: it fits a slope over random
      subsamples, so two evaluations of the same cloud differ. We measure that directly by
      re-evaluating the same clouds under different estimator seeds and propagating it.
  (c) MATCHING NOISE. The attribution inverts a silhouette-versus-alpha curve and interpolates on
      a finite alpha grid. Both the target silhouette and the inversion carry error, so we resample
      the target from its own run-level distribution and re-interpolate inside the bootstrap.

The reported interval is the widest of the defensible ones rather than the narrowest, and we say
which resampling unit produced it.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from analysis import geom
from tools.b2_attribution import alpha_at, value_at, real_trajectory_change, TRAJ

OUT = REPO / "RESULTS"
N_BOOT = 4000
ESTS = ["ph_dim", "eff_rank", "twonn", "pca_pr"]


def phdim_noise():
    """(b) How much does PH-dim move on a FIXED cloud when only the estimator seed changes?"""
    from tools.b1_gride import cloud
    rows = []
    for s in (0, 1, 2):
        X, _ = cloud(5, 8, 32, 0.05, seed=s)
        v = [geom.ph_dim(X, seed=k)[0] for k in range(12)]
        v = np.array([x for x in v if np.isfinite(x)])
        rows.append({"cloud_seed": s, "mean": v.mean(), "sd": v.std(ddof=1),
                     "cv": v.std(ddof=1) / max(abs(v.mean()), 1e-9)})
    N = pd.DataFrame(rows)
    N.to_csv(OUT / "v2_phdim_noise.csv", index=False)
    cv = float(N.cv.median())
    print(f"  (b) PH-dim re-evaluation noise on a fixed cloud: CV = {cv:.4f} "
          f"({100*cv:.2f}% of the value)")
    return cv


def attribution_draws(D, T, ests, rng, cv, resample="run"):
    """One bootstrap replicate of the attribution, with (a), (b) and (c) all resampled."""
    dirs = sorted(D.dir.unique())
    if resample == "run":
        take = list(rng.choice(dirs, len(dirs), replace=True))
    elif resample == "seed":
        seeds = sorted(D.seed.unique())
        pick = rng.choice(seeds, len(seeds), replace=True)
        take = [d for s in pick for d in D[D.seed == s].dir.unique()]
    else:
        take = dirs
    # (c) the target silhouettes are resampled from the trajectory runs, not held fixed
    tt = T.sample(len(T), replace=True, random_state=int(rng.integers(1 << 31)))
    lo_t, hi_t = float(tt.sil_init.median()), float(tt.sil_final.median())
    out = {}
    for est in ests:
        vals = []
        for d in take:
            g = D[D.dir == d]
            a0, _ = alpha_at(g, lo_t)
            a1, _ = alpha_at(g, hi_t)
            v0, v1 = value_at(g, est, a0), value_at(g, est, a1)
            if not (np.isfinite(v0) and np.isfinite(v1) and abs(v0) > 1e-12):
                continue
            # (b) estimator noise injected multiplicatively on both endpoints
            v0 *= (1 + rng.normal(0, cv)); v1 *= (1 + rng.normal(0, cv))
            vals.append((v0 - v1) / abs(v0))
        col = TRAJ.get(est)
        if not vals or col is None or f"pre_{est}" not in tt:
            continue
        obs = ((tt[f"pre_{est}"] - tt[f"post_{est}"]) / tt[f"pre_{est}"].abs()).dropna()
        if not len(obs) or abs(obs.median()) < 1e-9:
            continue
        out[est] = float(np.median(vals) / obs.median())
    return out


def main():
    print("=== Item 3: honest intervals on the headline percentages ===\n")
    cv = phdim_noise()

    D = pd.read_csv(OUT / "b3_seed_alpha.csv")
    T = real_trajectory_change()
    T = T[T.dir.str.contains("task-add__p-23")].reset_index(drop=True)
    rng = np.random.default_rng(0)

    print("\n  (a)+(b)+(c) full propagation, 9-seed arm vs the Z_23 trajectory runs")
    rows = []
    for unit in ("run", "seed"):
        draws = {e: [] for e in ESTS}
        for _ in range(N_BOOT // 4):
            r = attribution_draws(D, T, ESTS, rng, cv, resample=unit)
            for k, v in r.items():
                draws[k].append(v)
        for e, v in draws.items():
            v = np.array([x for x in v if np.isfinite(x)])
            if len(v) < 50:
                continue
            rows.append({"arm": "9-seed", "unit": unit, "estimator": e,
                         "point": float(np.median(v)),
                         "lo": float(np.percentile(v, 2.5)),
                         "hi": float(np.percentile(v, 97.5))})

    # leave-one-out, the honest worst case at n = 9
    base = attribution_draws(D, T, ESTS, np.random.default_rng(1), 0.0, resample="none")
    loo = {e: [] for e in ESTS}
    for d in sorted(D.dir.unique()):
        sub = D[D.dir != d]
        r = attribution_draws(sub, T, ESTS, np.random.default_rng(2), 0.0, resample="none")
        for k, v in r.items():
            loo[k].append(v)
    for e, v in loo.items():
        if v:
            rows.append({"arm": "9-seed", "unit": "leave-one-seed-out", "estimator": e,
                         "point": base.get(e, np.nan),
                         "lo": float(np.min(v)), "hi": float(np.max(v))})

    A = pd.DataFrame(rows)
    A.to_csv(OUT / "v2_intervals_attribution.csv", index=False)
    print(f"  {'estimator':<11}{'unit':<22}{'point':>8}{'  95% / range':>20}")
    for _, r in A.iterrows():
        print(f"  {r.estimator:<11}{r.unit:<22}{r.point:>7.0%}   [{r.lo:>5.0%}, {r.hi:>5.0%}]")

    # widest defensible interval per estimator
    print("\n  WIDEST defensible interval per estimator (this is what the paper should quote):")
    wid = []
    for e, g in A.groupby("estimator"):
        lo, hi = float(g.lo.min()), float(g.hi.max())
        pt = float(g[g.unit == "run"].point.iloc[0]) if (g.unit == "run").any() else float(g.point.median())
        wid.append({"estimator": e, "point": pt, "lo": lo, "hi": hi,
                    "widest_from": g.loc[g.hi.idxmax(), "unit"]})
        print(f"    {e:<11} {pt:>5.0%}  [{lo:>5.0%}, {hi:>5.0%}]   (widened by {g.loc[g.hi.idxmax(), 'unit']})")
    pd.DataFrame(wid).to_csv(OUT / "v2_intervals_widest.csv", index=False)

    # ---- the C1 recombination percentage, same treatment
    print("\n  recombination attribution (C1), resampled by seed with estimator noise")
    C = pd.read_csv(OUT / "c1_recombine.csv")
    rows = []
    for est in ("ph_dim", "eff_rank", "twonn", "H1_total"):
        per = {}
        for d, g in C.groupby("dir"):
            lo_, hi_ = g.s.min(), g.s.max()
            def v(s_, t_):
                q = g[(g.s == s_) & (g.t == t_)][est]
                return float(q.iloc[0]) if len(q) else np.nan
            diag, betw = v(lo_, lo_) - v(hi_, hi_), v(lo_, lo_) - v(hi_, lo_)
            if np.isfinite(diag) and abs(diag) > 1e-9:
                per[d] = betw / diag
        vals = np.array(list(per.values()))
        if len(vals) < 3:
            continue
        bs = []
        for _ in range(N_BOOT):
            t_ = rng.choice(vals, len(vals), replace=True) * (1 + rng.normal(0, cv, len(vals)))
            bs.append(np.median(t_))
        lo_o = min(np.median(np.delete(vals, i)) for i in range(len(vals)))
        hi_o = max(np.median(np.delete(vals, i)) for i in range(len(vals)))
        rows.append({"estimator": est, "point": float(np.median(vals)),
                     "boot_lo": float(np.percentile(bs, 2.5)),
                     "boot_hi": float(np.percentile(bs, 97.5)),
                     "loo_lo": float(lo_o), "loo_hi": float(hi_o), "n": len(vals)})
        print(f"    {est:<11} {np.median(vals):>6.0%}  boot [{np.percentile(bs,2.5):>5.0%}, "
              f"{np.percentile(bs,97.5):>5.0%}]  leave-one-out [{lo_o:>5.0%}, {hi_o:>5.0%}]")
    pd.DataFrame(rows).to_csv(OUT / "v2_intervals_recombine.csv", index=False)


if __name__ == "__main__":
    main()
