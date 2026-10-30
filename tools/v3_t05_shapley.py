"""
T0.5 -- additive Shapley attribution from the checkpoint-recombination grid.

The v2 recombination reported one-factor shares (advance centroids only: 116% of the diagonal
change; advance residuals only: 38%) with a negative interaction absorbing the remainder --
correct but non-additive, and the "116%" invites misreading. The two-factor Shapley value fixes
this: with f(C, R) the estimator on the recombined cloud (centroid state C, residual state R;
0 = initial checkpoint, 1 = converged),

    phi_between = 1/2 [f(C0,R0) - f(C1,R0)] + 1/2 [f(C0,R1) - f(C1,R1)]
    phi_within  = 1/2 [f(C0,R0) - f(C0,R1)] + 1/2 [f(C1,R0) - f(C1,R1)]

(signed so that a positive phi is a share of the observed training-time FALL), and
phi_between + phi_within = f(C0,R0) - f(C1,R1) exactly -- the shares sum to 100% of the
diagonal change by construction, no interaction term left over.

Computed per seed on c1_recombine.csv (9 seeds, unaligned) and on v2_align_recombine.csv
(aligned and unaligned frames), with seed-level medians, percentile + BCa bootstrap intervals,
and leave-one-seed-out ranges. Outputs RESULTS/v3_shapley.csv and prints the per-seed values.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import norm

OUT = REPO / "RESULTS"
ESTS = ["ph_dim", "H1_total", "eff_rank", "pca_pr", "twonn", "mle_k10", "corr_dim"]
N_BOOT = 10_000


def bca(theta, boots, jacks, level=0.95):
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    jacks = np.asarray([j for j in jacks if np.isfinite(j)])
    if len(boots) < 100 or len(jacks) < 3:
        return np.nan, np.nan
    z0 = norm.ppf(np.clip(np.mean(boots < theta), 1e-6, 1 - 1e-6))
    d = jacks.mean() - jacks
    den = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = np.sum(d ** 3) / den if den > 0 else 0.0
    out = []
    for alph in ((1 - level) / 2, 1 - (1 - level) / 2):
        z = z0 + norm.ppf(alph)
        out.append(float(np.percentile(boots, 100 * np.clip(norm.cdf(z0 + z / (1 - a * z)),
                                                            1e-6, 1 - 1e-6))))
    return out[0], out[1]


def shapley_per_seed(G, est):
    """Per-seed Shapley shares from the endpoint 2x2 of the (s,t) grid."""
    rows = []
    for d, g in G.groupby("dir"):
        lo, hi = g.s.min(), g.s.max()
        def f(s_, t_):
            q = g[(g.s == s_) & (g.t == t_)][est]
            return float(q.iloc[0]) if len(q) else np.nan
        f00, f10, f01, f11 = f(lo, lo), f(hi, lo), f(lo, hi), f(hi, hi)
        diag = f00 - f11
        if not all(np.isfinite(v) for v in (f00, f10, f01, f11)) or abs(diag) < 1e-9:
            continue
        phi_b = 0.5 * (f00 - f10) + 0.5 * (f01 - f11)
        phi_w = 0.5 * (f00 - f01) + 0.5 * (f10 - f11)
        assert abs(phi_b + phi_w - diag) < 1e-6 * max(1, abs(diag))
        rows.append({"dir": d, "seed": int(g.seed.iloc[0]), "diag": diag,
                     "phi_between": phi_b / diag, "phi_within": phi_w / diag})
    return pd.DataFrame(rows)


def summarise(P, est, frame, source, rng):
    v = P.phi_between.values
    med = float(np.median(v))
    boots = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(N_BOOT)]
    jack = [np.median(np.delete(v, i)) for i in range(len(v))]
    p_lo, p_hi = np.percentile(boots, [2.5, 97.5])
    b_lo, b_hi = bca(med, boots, jack)
    return {"source": source, "frame": frame, "estimator": est, "n_seeds": len(P),
            "between_share": med, "within_share": float(np.median(P.phi_within.values)),
            "pct_lo": float(p_lo), "pct_hi": float(p_hi), "bca_lo": b_lo, "bca_hi": b_hi,
            "loso_lo": float(np.min(jack)), "loso_hi": float(np.max(jack))}


def main():
    rng = np.random.default_rng(0)
    out = []
    C = pd.read_csv(OUT / "c1_recombine.csv")
    print("=== T0.5: Shapley attribution (between + within = 100% by construction) ===\n")
    print("--- c1_recombine (9 seeds, unaligned) ---")
    print(f"  {'estimator':<10}{'between':>9}{'within':>9}{'  95% BCa (between)':>22}"
          f"{'LOSO':>18}")
    for est in ESTS:
        P = shapley_per_seed(C, est)
        if len(P) < 3:
            continue
        r = summarise(P, est, "unaligned", "c1", rng)
        out.append(r)
        print(f"  {est:<10}{r['between_share']:>8.0%}{r['within_share']:>9.0%}"
              f"      [{r['bca_lo']:>5.0%}, {r['bca_hi']:>5.0%}]"
              f"   [{r['loso_lo']:>5.0%}, {r['loso_hi']:>5.0%}]")
        if est == "ph_dim":
            print("    per-seed phi_between: "
                  + ", ".join(f"{x:+.2f}" for x in P.phi_between))

    A = pd.read_csv(OUT / "v2_align_recombine.csv")
    for frame, g in A.groupby("frame"):
        print(f"\n--- v2_align_recombine, frame = {frame} ---")
        for est in [e for e in ESTS if e in g.columns]:
            P = shapley_per_seed(g, est)
            if len(P) < 3:
                continue
            r = summarise(P, est, frame, "align", rng)
            out.append(r)
            print(f"  {est:<10}{r['between_share']:>8.0%}{r['within_share']:>9.0%}"
                  f"      [{r['bca_lo']:>5.0%}, {r['bca_hi']:>5.0%}]"
                  f"   [{r['loso_lo']:>5.0%}, {r['loso_hi']:>5.0%}]")

    pd.DataFrame(out).to_csv(OUT / "v3_shapley.csv", index=False)
    print("\nwrote v3_shapley.csv")


if __name__ == "__main__":
    main()
