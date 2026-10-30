"""
STEP 2 -- timing decomposition. Does each geometric measure LEAD, LAG or COINCIDE with the
test-accuracy rise?

Three statistics, deliberately not one:

  phase   position at which the measure completes half of its total change, on a timeline
          where 0 = memorization (train acc first >= 0.99) and 1 = t_test50 (test acc first
          reaches half its final value). phase < 1 leads, ~1 coincides, > 1 lags.

  ccf     lag maximising the cross-correlation of FIRST DIFFERENCES against test accuracy --
          Monod et al.'s own timing statistic, so results are comparable to theirs. Negative
          lag = the measure moves first.

  hp_gain the only one that can support a PREDICTION claim: does knowing the measure early
          reduce out-of-sample error on t_test50 beyond what the hyperparameters already give?
          2406.02234 (Tan/Bronstein/Monod) showed exactly this confound sinks topological
          generalization measures, so any "X leads" claim that fails here is reported as
          retrospective detection, not prediction.

Anchors already owned: equiv_R2 leads; ambient homology has no event. Both are carried
through the same pipeline so the comparison is like-for-like.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import spearmanr

OUT = REPO / "RESULTS"

MEASURES = ["ph_dim", "corr_dim", "twonn", "mle_id_k5", "mle_id_k10", "mle_id_k20",
            "pca_pr", "eff_rank", "spec_decay", "lid_sd", "lid_skew", "lid_bimod",
            "gap_ratio", "nn1_ratio", "orc_mean", "orc_frac_neg",
            "mag_t1", "mag_t5", "nc1", "nc2_etf", "nc_wb",
            "amb_H1_max", "equiv_R2"]


def movement(ep, v, rng):
    """
    Does the series move at all? Timing is meaningless otherwise -- and worse than meaningless:
    a pure-noise series crosses "half of its start->end change" on the first wobble, so it is
    scored as the EARLIEST leader. That is exactly how ambient H1 first came out ranked ahead of
    every geometric measure. So phase is only reported for series that pass here.
    """
    v = np.asarray(v, float); ep = np.asarray(ep, float)
    ok = np.isfinite(v)
    if ok.sum() < 6 or np.std(v[ok]) == 0:
        return np.nan, np.nan, np.nan
    v, ep = v[ok], ep[ok]
    trend = abs(spearmanr(v, ep).statistic)
    null = np.median([abs(spearmanr(rng.permutation(v), ep).statistic) for _ in range(12)])
    d = np.abs(np.diff(v)); md = np.median(d)
    snr = abs(v[-1] - v[0]) / (1.4826 * md) if md > 0 else np.nan
    return trend, null, snr


def half_epoch(ep, v, smooth=3):
    """
    Epoch at which v completes half its total change, after a rolling median so a single noisy
    checkpoint cannot define the crossing.
    """
    v = np.asarray(v, float); ep = np.asarray(ep, float)
    ok = np.isfinite(v)
    if ok.sum() < 6:
        return np.nan
    v, ep = v[ok], ep[ok]
    if smooth > 1 and len(v) >= smooth:
        v = pd.Series(v).rolling(smooth, center=True, min_periods=1).median().values
    a, b = v[0], v[-1]
    if abs(b - a) < 1e-12:
        return np.nan
    target = a + 0.5 * (b - a)
    crossed = (v - target) * np.sign(b - a) >= 0
    if not crossed.any():
        return np.nan
    return float(ep[np.argmax(crossed)])


def balanced_split(df):
    """
    Alternate seeds WITHIN each (task,p,tf) stratum. Seed-parity alone put only 2 grokked runs
    in confirmation, because grokking is not independent of seed parity in this corpus.
    """
    lab = {}
    for _, g in df.drop_duplicates("dir").groupby(["task", "p", "tf"]):
        for i, d in enumerate(sorted(g.dir)):
            lab[d] = "discovery" if i % 2 == 0 else "confirmation"
    return df.dir.map(lab)


def ccf_lag(ep, v, acc, max_lag=8):
    """Lag (in checkpoints) maximising corr of first differences. Negative => v moves first."""
    v = np.asarray(v, float); acc = np.asarray(acc, float)
    ok = np.isfinite(v) & np.isfinite(acc)
    if ok.sum() < 8:
        return np.nan
    dv, da = np.diff(v[ok]), np.diff(acc[ok])
    if np.std(dv) < 1e-12 or np.std(da) < 1e-12:
        return np.nan
    best, bl = -2.0, np.nan
    for L in range(-max_lag, max_lag + 1):
        if L < 0:
            x, y = dv[:L], da[-L:]
        elif L > 0:
            x, y = dv[L:], da[:-L]
        else:
            x, y = dv, da
        if len(x) < 4 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        c = abs(np.corrcoef(x, y)[0, 1])
        if np.isfinite(c) and c > best:
            best, bl = c, L
    return bl


if __name__ == "__main__":
    D = pd.read_csv(OUT / "geom_series.csv")
    extra = OUT / "geom_series_tf.csv"
    if extra.exists():
        D = pd.concat([D, pd.read_csv(extra)], ignore_index=True)
    D["split"] = balanced_split(D)
    print(f"loaded {D.shape}, {D.dir.nunique()} runs "
          f"({D[D.arch=='transformer'].dir.nunique()} transformer)\n")
    rng = np.random.default_rng(0)
    rows = []
    for (dirn), g in D.groupby("dir"):
        g = g.sort_values("epoch")
        fin = g.test_acc.values[-3:].mean()
        if fin < 0.5:
            continue                                    # timing is undefined without a rise
        tm = g[g.train_acc >= 0.99].epoch.min()
        t50 = g[g.test_acc >= 0.5 * fin].epoch.min()
        if not np.isfinite(tm) or not np.isfinite(t50) or t50 <= tm:
            continue
        for m in MEASURES:
            if m not in g:
                continue
            trend, null, snr = movement(g.epoch.values, g[m].values, rng)
            moves = np.isfinite(trend) and trend > max(2 * null, 0.5) and (snr > 3)
            he = half_epoch(g.epoch.values, g[m].values) if moves else np.nan
            rows.append({"dir": dirn, "measure": m, "moves": bool(moves),
                         "trend": trend, "trend_null": null, "snr": snr,
                         "phase": (he - tm) / (t50 - tm) if np.isfinite(he) else np.nan,
                         "ccf_lag": ccf_lag(g.epoch.values, g[m].values, g.test_acc.values),
                         "t_mem": tm, "t50": t50,
                         "arch": g.arch.iloc[0], "task": g.task.iloc[0], "p": g.p.iloc[0],
                         "tf": g.tf.iloc[0], "seed": g.seed.iloc[0], "split": g.split.iloc[0]})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "geom_timing.csv", index=False)
    pd.set_option("display.width", 200)

    for sp in ("discovery", "confirmation"):
        S = T[T.split == sp]
        if not len(S):
            continue
        print(f"=== {sp.upper()} (n={S.dir.nunique()} runs) ===")
        agg = S.groupby("measure").agg(n_runs=("moves", "size"),
                                       moves=("moves", "mean"),
                                       trend=("trend", "median"),
                                       snr=("snr", "median"),
                                       n=("phase", "count"),
                                       phase_med=("phase", "median"),
                                       phase_q1=("phase", lambda x: x.quantile(.25)),
                                       phase_q3=("phase", lambda x: x.quantile(.75)),
                                       leads=("phase", lambda x: (x < 1).mean()),
                                       ccf_med=("ccf_lag", "median")).round(3)
        print(agg.sort_values("phase_med").to_string())
        print()
