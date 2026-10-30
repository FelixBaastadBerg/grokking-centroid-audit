"""
T0.1 -- reconcile the matched-range attribution point estimates with their leave-one-seed-out
ranges (the possible Table-21 error).

The v2 numbers came from two pipelines that silently disagreed:

  * b2_attribution.py computes the DETERMINISTIC full-sample estimate: invert each of the 9
    intervention seeds' silhouette-vs-alpha curves at the median real silhouette endpoints
    (medians over the 7 Z_23 trajectory runs), then
        attribution = median_seeds(matched change) / median_runs(observed change)
    -- a ratio of medians with a common denominator, 138-139% for PH-dimension.

  * v2_intervals.py was meant to add honest uncertainty, but its attribution_draws() resamples
    the trajectory runs (the matching targets AND the observed-change denominator) on EVERY
    call -- including the calls used for the "point" and the leave-one-seed-out column. Each
    LOSO replicate re-seeded default_rng(2), so all nine replicates shared ONE arbitrary
    bootstrap resample of the trajectory runs. The published LOSO column therefore measured
    nothing about seeds: it was the full-sample estimate evaluated at a single randomly
    perturbed target, which is why it sat far from the point estimate (111% vs 138 for PH-dim,
    32-38% vs 127 for TwoNN) and why its width was near zero.

This file recomputes everything deterministically from the CSVs and reports:
  (1) the aggregation formula, with both aggregations shown (ratio-of-medians, and the
      median-of-ratios over all seed x run pairs);
  (2) all nine seed-level numerators, all seven run-level denominators, and the nine
      seed-level ratios;
  (3) true leave-one-seed-out and leave-one-trajectory-run-out ranges (targets held at the
      full-sample values);
  (4) percentile AND BCa bootstrap intervals, resampling seeds and trajectory runs jointly,
      with PH-dim's measured re-evaluation noise (CV 0.37%) optionally injected;
  (5) a reproduction of the buggy pipeline, confirming where 111% / 32-38% came from.

Outputs: RESULTS/v3_t01_seedlevel.csv, v3_t01_summary.csv, v3_t01_diagnosis.txt
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import norm
from tools.b2_attribution import alpha_at, value_at, real_trajectory_change

OUT = REPO / "RESULTS"
ESTS = ["ph_dim", "eff_rank", "pca_pr", "twonn"]
N_BOOT = 10_000
PHDIM_CV = 0.0037          # measured re-evaluation noise, v2_phdim_noise.csv


def seed_numerators(D, sil_lo, sil_hi, ests):
    """Matched fractional change per intervention seed, at fixed targets."""
    rows = []
    for d, g in D.groupby("dir"):
        a0, _ = alpha_at(g, sil_lo)
        a1, _ = alpha_at(g, sil_hi)
        rec = {"dir": d, "seed": int(g.seed.iloc[0]), "alpha_lo": a0, "alpha_hi": a1}
        for est in ests:
            v0, v1 = value_at(g, est, a0), value_at(g, est, a1)
            rec[f"num_{est}"] = ((v0 - v1) / abs(v0)
                                 if np.isfinite(v0) and np.isfinite(v1) and abs(v0) > 1e-12
                                 else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def run_denominators(T, ests):
    """Observed fractional change per trajectory run."""
    rows = []
    for _, r in T.iterrows():
        rec = {"dir": r["dir"]}
        for est in ests:
            pre, post = r.get(f"pre_{est}", np.nan), r.get(f"post_{est}", np.nan)
            rec[f"obs_{est}"] = ((pre - post) / abs(pre)
                                 if np.isfinite(pre) and np.isfinite(post) and abs(pre) > 1e-12
                                 else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def rom(num, obs):
    """Ratio of medians -- the paper's aggregation."""
    num = num[np.isfinite(num)]; obs = obs[np.isfinite(obs)]
    if not len(num) or not len(obs) or abs(np.median(obs)) < 1e-9:
        return np.nan
    return float(np.median(num) / np.median(obs))


def bca_interval(theta_hat, boots, jacks, level=0.95):
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    jacks = np.asarray([j for j in jacks if np.isfinite(j)])
    if len(boots) < 100 or len(jacks) < 3:
        return np.nan, np.nan
    prop = np.clip(np.mean(boots < theta_hat), 1e-6, 1 - 1e-6)
    z0 = norm.ppf(prop)
    jm = jacks.mean()
    d = jm - jacks
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = np.sum(d ** 3) / denom if denom > 0 else 0.0
    lo_hi = []
    for alph in ((1 - level) / 2, 1 - (1 - level) / 2):
        z = z0 + norm.ppf(alph)
        adj = norm.cdf(z0 + z / (1 - a * z))
        lo_hi.append(float(np.percentile(boots, 100 * np.clip(adj, 1e-6, 1 - 1e-6))))
    return lo_hi[0], lo_hi[1]


def main():
    D = pd.read_csv(OUT / "b3_seed_alpha.csv")
    T = real_trajectory_change()
    T = T[T.dir.str.contains("task-add__p-23")].reset_index(drop=True)
    sil_lo, sil_hi = float(T.sil_init.median()), float(T.sil_final.median())
    print("=== T0.1: attribution reconciliation ===")
    print(f"  aggregation: attribution = median over {D.dir.nunique()} intervention seeds of "
          f"the matched fractional change,\n               divided by the median over {len(T)} "
          f"Z_23 trajectory runs of the observed fractional change\n"
          f"               (RATIO OF MEDIANS; targets sil {sil_lo:+.3f} -> {sil_hi:+.3f}, "
          f"medians over the same {len(T)} runs)\n")

    NUM = seed_numerators(D, sil_lo, sil_hi, ESTS)
    DEN = run_denominators(T, ESTS)
    NUM.to_csv(OUT / "v3_t01_seedlevel.csv", index=False)

    rng = np.random.default_rng(0)
    summary, diag_lines = [], []
    for est in ESTS:
        num = NUM[f"num_{est}"].values.astype(float)
        obs = DEN[f"obs_{est}"].values.astype(float)
        den = float(np.median(obs[np.isfinite(obs)]))
        point = rom(num, obs)
        # the alternative aggregation, for the record: median of all seed x run ratios
        pairs = [n / o for n in num for o in obs
                 if np.isfinite(n) and np.isfinite(o) and abs(o) > 1e-9]
        mor = float(np.median(pairs)) if pairs else np.nan

        print(f"--- {est} ---")
        print(f"  seed-level numerators (matched fractional change at fixed targets):")
        for _, r in NUM.iterrows():
            n = r[f"num_{est}"]
            print(f"    seed {int(r.seed):>2}: num = {n:+.4f}   ratio = {n / den:+.3f}")
        print(f"  run-level denominators (observed fractional change): "
              + ", ".join(f"{o:+.4f}" for o in obs))
        print(f"  denominator median = {den:+.4f}")
        print(f"  FULL-SAMPLE point:  ratio-of-medians = {point:.1%}   "
              f"(median-of-ratios over pairs = {mor:.1%})")

        # true LOSO / LORO at fixed full-sample targets
        loso = [rom(np.delete(num, i), obs) for i in range(len(num))]
        # leaving a trajectory run out moves the TARGETS as well as the denominator
        loro = []
        for j in range(len(T)):
            Tj = T.drop(T.index[j])
            nj = seed_numerators(D, float(Tj.sil_init.median()), float(Tj.sil_final.median()),
                                 [est])[f"num_{est}"].values.astype(float)
            oj = run_denominators(Tj, [est])[f"obs_{est}"].values.astype(float)
            loro.append(rom(nj, oj))
        print(f"  TRUE leave-one-seed-out:            [{np.nanmin(loso):.1%}, {np.nanmax(loso):.1%}]")
        print(f"  TRUE leave-one-trajectory-run-out:  [{np.nanmin(loro):.1%}, {np.nanmax(loro):.1%}]")

        # bootstrap: seeds and trajectory runs resampled jointly; targets recomputed per draw;
        # estimator noise injected on the numerator endpoints via the measured CV
        boots = []
        ok = np.isfinite(num)
        for _ in range(N_BOOT):
            bi = rng.integers(0, len(num), len(num))
            bj = rng.integers(0, len(T), len(T))
            Tb = T.iloc[bj]
            ob = run_denominators(Tb, [est])[f"obs_{est}"].values.astype(float)
            nb = num[bi]
            if est == "ph_dim" and PHDIM_CV > 0:
                nb = nb * (1 + rng.normal(0, PHDIM_CV * np.sqrt(2), len(nb)))
            boots.append(rom(nb, ob))
        boots = np.asarray(boots)
        p_lo, p_hi = np.nanpercentile(boots, [2.5, 97.5])
        b_lo, b_hi = bca_interval(point, boots, loso + loro)
        print(f"  percentile bootstrap 95%:  [{p_lo:.1%}, {p_hi:.1%}]")
        print(f"  BCa bootstrap 95%:         [{b_lo:.1%}, {b_hi:.1%}]")

        # reproduce the buggy pipeline: one fixed resample of T with default_rng(2), as every
        # LOSO replicate in v2_intervals.py used
        tt = T.sample(len(T), replace=True,
                      random_state=int(np.random.default_rng(2).integers(1 << 31)))
        lo_t, hi_t = float(tt.sil_init.median()), float(tt.sil_final.median())
        nb = seed_numerators(D, lo_t, hi_t, [est])[f"num_{est}"].values.astype(float)
        ob = run_denominators(tt, [est])[f"obs_{est}"].values.astype(float)
        buggy = rom(nb, ob)
        print(f"  buggy v2 'LOSO' reproduced (single rng(2) resample of targets): {buggy:.1%}\n")
        diag_lines.append(f"{est}: point {point:.1%}, buggy-LOSO artifact {buggy:.1%}")

        summary.append({"estimator": est, "point_rom": point, "point_mor": mor,
                        "den_median": den,
                        "loso_lo": float(np.nanmin(loso)), "loso_hi": float(np.nanmax(loso)),
                        "loro_lo": float(np.nanmin(loro)), "loro_hi": float(np.nanmax(loro)),
                        "pct_lo": float(p_lo), "pct_hi": float(p_hi),
                        "bca_lo": b_lo, "bca_hi": b_hi,
                        "n_seeds": int(np.isfinite(num).sum()),
                        "n_runs": int(np.isfinite(obs).sum())})

    S = pd.DataFrame(summary)
    S.to_csv(OUT / "v3_t01_summary.csv", index=False)
    (OUT / "v3_t01_diagnosis.txt").write_text(
        "T0.1 diagnosis: the v2 leave-one-seed-out column was computed with the trajectory\n"
        "targets bootstrap-resampled once (default_rng(2)) and shared across all nine\n"
        "replicates, so it was not a LOSO at all. The full-sample points were deterministic\n"
        "and stand; the LOSO column is replaced by the true ranges below.\n\n"
        + "\n".join(diag_lines) + "\n", encoding="utf-8")
    print("wrote v3_t01_seedlevel.csv, v3_t01_summary.csv, v3_t01_diagnosis.txt")


if __name__ == "__main__":
    main()
