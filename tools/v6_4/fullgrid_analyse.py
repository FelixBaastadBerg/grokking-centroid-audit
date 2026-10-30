"""
v6.4 -- decompose the full-grid recombination, and compare it with the v6.3 400-point analysis.

Reads RESULTS_v6_4/fullgrid_recombine.csv (5x5 grid, 9 seeds, 2 preprocessing pipelines) and
RESULTS/c1_recombine.csv (the v6.3 400-training-point grid), and writes:

  fullgrid_shapley.csv    per estimator x interval x pipeline x seed: both factor orders, the
                          symmetric Shapley value, raw contributions, and the share
  fullgrid_summary.csv    per estimator x interval x pipeline: median share, seed range,
                          seed-bootstrap interval, and whether the interval's denominator is
                          appreciable
  fullgrid_vs_400.csv     paired per-seed differences between the two input populations

Conventions, all fixed before the results were inspected:

  * unit of replication is the TRAINING SEED. Bootstraps resample seeds, never points,
    checkpoints or grid cells.
  * an interval's share is interpreted only when |its own change| is at least 5% of the
    full-trajectory change, the same denominator rule v6.3 uses. Raw contributions are always
    reported, shares only when the rule passes.
  * phi_between = 1/2 [f(a,a) - f(b,a)] + 1/2 [f(a,b) - f(b,b)], so
    phi_between + phi_within = f(a,a) - f(b,b) exactly.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

OUT = REPO / "RESULTS_v6_4"
ESTS = ["ph_dim", "twonn", "mle_k10", "eff_rank", "pca_pr", "corr_dim"]
PHASES = [(0, 1, "init->pre-grok"), (1, 2, "pre-grok->t_grok"), (2, 3, "t_grok->post-mid"),
          (3, 4, "post-mid->conv"), (0, 2, "init->t_grok"), (2, 4, "t_grok->conv"),
          # pre-grok->conv excludes initialisation, the one checkpoint at which the
          # diagonal-standardised metric leaves PH-dimension's log-log fit outside its
          # validated regime; it is the interval on which the two metrics are comparable
          (1, 4, "pre-grok->conv"),
          (0, 4, "init->conv (endpoint)")]
APPRECIABLE = 0.05
N_BOOT = 10_000


def decompose(g, est, a, b):
    def f(s_, t_):
        q = g[(g.s == s_) & (g.t == t_)][est]
        return float(q.iloc[0]) if len(q) else np.nan
    faa, fba, fab, fbb = f(a, a), f(b, a), f(a, b), f(b, b)
    if not all(np.isfinite(v) for v in (faa, fba, fab, fbb)):
        return None
    diag = faa - fbb
    first = faa - fba          # centroids advanced first
    second = fab - fbb         # centroids advanced second
    phi_b = 0.5 * (first + second)
    full = f(0, 0) - f(4, 4)
    return {"f00": faa, "f10": fba, "f01": fab, "f11": fbb, "diag": diag,
            "centroid_first_raw": first, "centroid_second_raw": second,
            "phi_between_raw": phi_b, "phi_within_raw": diag - phi_b,
            "full_diag": full,
            "centroid_first_share": first / diag if abs(diag) > 1e-9 else np.nan,
            "centroid_second_share": second / diag if abs(diag) > 1e-9 else np.nan,
            "between_share": phi_b / diag if abs(diag) > 1e-9 else np.nan,
            "rel_change": abs(diag) / abs(full) if abs(full) > 1e-9 else np.nan}


def per_seed(D, pipelines):
    rows = []
    for pipe in pipelines:
        P = D[D.pipeline == pipe] if "pipeline" in D.columns else D
        for est in ESTS:
            if est not in P.columns:
                continue
            for a, b, label in PHASES:
                for d, g in P.groupby("dir"):
                    r = decompose(g, est, a, b)
                    if r is None:
                        continue
                    rows.append({"pipeline": pipe, "estimator": est, "a": a, "b": b,
                                 "phase": label, "dir": d, "seed": int(g.seed.iloc[0]), **r})
    return pd.DataFrame(rows)


def summarise(S, rng):
    rows = []
    for (pipe, est, phase), g in S.groupby(["pipeline", "estimator", "phase"], sort=False):
        v = g.between_share.dropna().values
        appreciable = float(np.median(g.rel_change.dropna())) >= APPRECIABLE \
            if g.rel_change.notna().any() else False
        if len(v) >= 3:
            boots = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(N_BOOT)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
        else:
            lo = hi = np.nan
        both_orders_majority = bool(
            (g.centroid_first_share.dropna() > 0.5).all() and
            (g.centroid_second_share.dropna() > 0.5).all()) if len(g) else False
        rows.append({
            "pipeline": pipe, "estimator": est, "phase": phase, "n_seeds": len(g),
            "median_diag": float(g.diag.median()),
            "median_rel_change": float(np.median(g.rel_change.dropna()))
            if g.rel_change.notna().any() else np.nan,
            "appreciable": appreciable,
            "median_between_share": float(np.median(v)) if len(v) else np.nan,
            "share_min": float(np.min(v)) if len(v) else np.nan,
            "share_max": float(np.max(v)) if len(v) else np.nan,
            "boot_lo": lo, "boot_hi": hi,
            "median_first_share": float(g.centroid_first_share.median()),
            "median_second_share": float(g.centroid_second_share.median()),
            "majority_both_orders_all_seeds": both_orders_majority,
            "median_phi_between_raw": float(g.phi_between_raw.median()),
            "median_phi_within_raw": float(g.phi_within_raw.median()),
            "phi_between_raw_min": float(g.phi_between_raw.min()),
            "phi_between_raw_max": float(g.phi_between_raw.max()),
        })
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(0)
    F = pd.read_csv(OUT / "fullgrid_recombine.csv")
    S = per_seed(F, sorted(F.pipeline.unique()))
    S.to_csv(OUT / "fullgrid_shapley.csv", index=False)
    G = summarise(S, rng)
    G.to_csv(OUT / "fullgrid_summary.csv", index=False)

    # ---- paired comparison with the v6.3 400-training-point analysis
    C = pd.read_csv(REPO / "RESULTS" / "c1_recombine.csv")
    C["pipeline"] = "train400"
    S4 = per_seed(C, ["train400"])
    S4.to_csv(OUT / "train400_shapley.csv", index=False)

    a = S[S.pipeline == "raw"][["estimator", "phase", "seed", "between_share",
                                "phi_between_raw", "diag"]]
    b = S4[["estimator", "phase", "seed", "between_share", "phi_between_raw", "diag"]]
    M = a.merge(b, on=["estimator", "phase", "seed"], suffixes=("_fullgrid", "_train400"))
    M["share_diff_pp"] = 100 * (M.between_share_fullgrid - M.between_share_train400)
    M.to_csv(OUT / "fullgrid_vs_400.csv", index=False)

    # ---- report
    for pipe in sorted(F.pipeline.unique()):
        print(f"\n================ pipeline: {pipe} ================")
        for est in ("ph_dim", "twonn"):
            print(f"\n=== {est} ===")
            print(f"  {'interval':<24}{'diag':>8}{'rel':>7}{'share':>8}"
                  f"{'  seed range':>17}{'  boot':>16}{'  1st':>7}{'  2nd':>7}  maj")
            q = G[(G.pipeline == pipe) & (G.estimator == est)]
            for _, r in q.iterrows():
                mark = "" if r.appreciable else "  (small denom)"
                print(f"  {r.phase:<24}{r.median_diag:>8.2f}{r.median_rel_change:>7.0%}"
                      f"{r.median_between_share:>8.0%}"
                      f"   [{r.share_min:>5.0%},{r.share_max:>6.0%}]"
                      f"   [{r.boot_lo:>5.0%},{r.boot_hi:>5.0%}]"
                      f"{r.median_first_share:>7.0%}{r.median_second_share:>7.0%}"
                      f"  {'yes' if r.majority_both_orders_all_seeds else 'no'}{mark}")

    print("\n\n=== paired full-grid minus 400-training-point, endpoint interval ===")
    E = M[M.phase == "init->conv (endpoint)"]
    for est in ("ph_dim", "twonn"):
        q = E[E.estimator == est]
        if not len(q):
            continue
        print(f"  {est:<10} full-grid {q.between_share_fullgrid.median():>6.0%}"
              f"   400-point {q.between_share_train400.median():>6.0%}"
              f"   paired diff {q.share_diff_pp.median():>+6.1f} pp"
              f"   [{q.share_diff_pp.min():+.1f}, {q.share_diff_pp.max():+.1f}]")
    print(f"\nwrote fullgrid_shapley.csv, fullgrid_summary.csv, fullgrid_vs_400.csv")


if __name__ == "__main__":
    main()
