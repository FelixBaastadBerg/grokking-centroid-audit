"""
v6 -- checkpoint-resolved Shapley attribution from the existing recombination grid.

The published Shapley share (89% between-class for PH-dimension) uses only the endpoint 2x2 of
the 5x5 (s,t) recombination grid in RESULTS/c1_recombine.csv. That grid already contains every
(s,t) pair over five checkpoints per seed -- indices 0 (initialisation), 1 (pre-grok, halfway to
t_grok in checkpoint index), 2 (t_grok), 3 (post-grok midpoint), 4 (convergence) -- so the same
two-factor Shapley decomposition can be evaluated over any sub-interval [a, b] of the trajectory
without new training:

    phi_between(a,b) = 1/2 [f(a,a) - f(b,a)] + 1/2 [f(a,b) - f(b,b)]

with phi_between + phi_within = f(a,a) - f(b,b) exactly.  This tests whether the endpoint
attribution is an artifact of the chosen endpoints: is between-class dominance stable across
phases, concentrated near the transition, or mainly post-grok?

A share is only meaningful when the interval's own diagonal change is non-negligible, so each
interval reports the median |diagonal change| relative to the full-trajectory change alongside
the share, and shares on near-zero denominators are flagged rather than interpreted.

Outputs RESULTS/v6_ckpt_shapley.csv (per estimator x interval x seed) and prints a summary.
Analysis-only: no training, no new measurement -- pure reuse of c1_recombine.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

OUT = REPO / "RESULTS"
ESTS = ["ph_dim", "H1_total", "eff_rank", "pca_pr", "twonn", "mle_k10", "corr_dim"]
PHASES = [  # (a, b, label)
    (0, 1, "init->pre-grok"),
    (1, 2, "pre-grok->t_grok"),
    (2, 3, "t_grok->post-mid"),
    (3, 4, "post-mid->conv"),
    (0, 2, "init->t_grok"),
    (2, 4, "t_grok->conv"),
    (0, 4, "full (published)"),
]
N_BOOT = 10_000


def shapley_interval(g, est, a, b):
    """Shapley shares of one seed's change over checkpoint interval [a, b]."""
    def f(s_, t_):
        q = g[(g.s == s_) & (g.t == t_)][est]
        return float(q.iloc[0]) if len(q) else np.nan
    faa, fba, fab, fbb = f(a, a), f(b, a), f(a, b), f(b, b)
    if not all(np.isfinite(v) for v in (faa, fba, fab, fbb)):
        return None
    diag = faa - fbb
    phi_b = 0.5 * (faa - fba) + 0.5 * (fab - fbb)
    full = f(0, 0) - f(4, 4)
    return {"diag": diag, "phi_between_abs": phi_b, "full_diag": full,
            "share": phi_b / diag if abs(diag) > 1e-9 else np.nan,
            "rel_change": abs(diag) / abs(full) if abs(full) > 1e-9 else np.nan}


def main():
    rng = np.random.default_rng(0)
    C = pd.read_csv(OUT / "c1_recombine.csv")
    rows = []
    for est in ESTS:
        for a, b, label in PHASES:
            per = []
            for d, g in C.groupby("dir"):
                r = shapley_interval(g, est, a, b)
                if r is not None:
                    per.append({"dir": d, "seed": int(g.seed.iloc[0]), **r})
            P = pd.DataFrame(per)
            v = P.share.dropna().values
            med = float(np.median(v)) if len(v) else np.nan
            if len(v) >= 3:
                boots = [np.median(rng.choice(v, len(v), replace=True))
                         for _ in range(N_BOOT)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
            else:
                lo = hi = np.nan
            rows.append({"estimator": est, "a": a, "b": b, "phase": label,
                         "n_seeds": len(P), "n_shares": len(v),
                         "median_share": med, "pct_lo": lo, "pct_hi": hi,
                         "median_rel_change": float(np.median(P.rel_change.dropna())),
                         "share_min": float(np.min(v)) if len(v) else np.nan,
                         "share_max": float(np.max(v)) if len(v) else np.nan})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "v6_ckpt_shapley.csv", index=False)

    for est in ESTS:
        print(f"\n=== {est} ===")
        print(f"  {'phase':<20}{'share':>8}{'  95% pct':>16}{'rel.change':>12}"
              f"{'range over seeds':>22}")
        for _, r in S[S.estimator == est].iterrows():
            print(f"  {r.phase:<20}{r.median_share:>7.0%}"
                  f"   [{r.pct_lo:>5.0%}, {r.pct_hi:>5.0%}]"
                  f"{r.median_rel_change:>11.0%}"
                  f"     [{r.share_min:>5.0%}, {r.share_max:>6.0%}]")
    print("\nwrote v6_ckpt_shapley.csv")


if __name__ == "__main__":
    main()
