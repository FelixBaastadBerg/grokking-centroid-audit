"""
P1-3 -- predictive horse-race for grokking onset: clustering vs dimension.

For each grokked run, take an EARLY measurement (a fixed early epoch, well before t_grok) of each
statistic and ask how well it predicts t_grok. Question: does early label-silhouette predict onset
at least as well as early PH-dim? If it does, that is operational support for the clustering
reading -- the one kind of support P0-1 failed to deliver causally.

Sampling: uniform only. Failure #18 (log-spaced sampling invalidates timing statistics) applies to
anything timing-related, so the early epoch is a fixed step index, identical across runs.

Baseline: hyperparameters alone (p, train fraction, task, architecture). A statistic that cannot
beat the hyperparameters is retrospective detection, not prediction -- the same standard applied to
the equivariance order parameter earlier.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import spearmanr

OUT = REPO / "RESULTS"
STATS = ["sil_true", "ph_dim", "twonn", "eff_rank", "spec_entropy", "gap_ratio", "mle_k10"]


def loo_mae(X, y):
    """Leave-one-run-out linear fit, mean absolute error."""
    X = np.asarray(X, float).reshape(len(y), -1)
    y = np.asarray(y, float)
    X = np.c_[X, np.ones(len(y))]
    err = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        try:
            beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        except Exception:
            continue
        err.append(abs(X[i] @ beta - y[i]))
    return float(np.mean(err)) if err else np.nan


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    # comeasure.csv is the largest UNIFORMLY-sampled corpus (fixed stride per run) carrying the
    # geometry battery and label-based clustering together. e9_geomdr.csv has only 5 unregularised
    # runs, too few for a horse race.
    D = pd.read_csv(OUT / "comeasure.csv").sort_values(["dir", "epoch"])
    inv = pd.read_csv(OUT / "_inventory.csv").set_index("dir")

    tg, early_rows = {}, []
    EARLY_FRAC = 0.10                      # a fixed early fraction of each run, well before t_grok
    for d, g in D.groupby("dir"):
        g = g.sort_values("epoch")
        fin = g.test_acc.values[-3:].mean()
        if fin < 0.5:
            continue
        t = g[g.test_acc >= 0.5 * fin].epoch.min()
        e_ep = g.epoch.iloc[int(len(g) * EARLY_FRAC)]
        if not np.isfinite(t) or e_ep >= t:
            continue
        row = g[g.epoch == e_ep].iloc[0].to_dict()
        row["t_grok"] = t; row["dir"] = d; row["early_epoch"] = e_ep
        early_rows.append(row); tg[d] = t
    E = pd.DataFrame(early_rows)
    print(f"=== P1-3: onset prediction on {len(E)} grokked runs (uniform stride) ===")
    print(f"    early measurement at the {EARLY_FRAC:.0%} point of each run, always before t_grok")
    print(f"    t_grok range {E.t_grok.min():.0f}-{E.t_grok.max():.0f}\n")

    y = E.t_grok.values
    base_mae = loo_mae(np.zeros((len(y), 1)), y)
    hp = pd.get_dummies(E[["task"]]).astype(float)
    hp["p"] = E.p.values; hp["tf"] = E.tf.values
    hp_mae = loo_mae(hp.values, y)
    print(f"  predict-the-mean baseline MAE : {base_mae:.0f} steps")
    print(f"  hyperparameters-only MAE      : {hp_mae:.0f} steps\n")

    CAND = [("nc_wb", "clustering (within/between)"), ("gap_ratio", "clustering (NN gap)"),
            ("ph_dim", "dimension (PH-dim)"), ("twonn", "dimension (TwoNN)"),
            ("mle_id_k10", "dimension (MLE-ID)"), ("eff_rank", "spectral (effective rank)"),
            ("pca_pr", "spectral (PCA-PR)")]
    rows = []
    for st, lab in CAND:
        if st not in E.columns:
            continue
        x = E[st].values
        good = np.isfinite(x)
        if good.sum() < 8:
            continue
        r = spearmanr(x[good], y[good])
        mae = loo_mae(x[good], y[good])
        both = loo_mae(np.c_[hp.values[good], x[good]], y[good])
        rows.append({"statistic": st, "family": lab, "n": int(good.sum()),
                     "spearman": r.statistic, "p": r.pvalue, "loo_mae": mae,
                     "mae_with_hparams": both, "beats_hparams": both < hp_mae})
    R = pd.DataFrame(rows).sort_values("spearman", key=abs, ascending=False)
    R.to_csv(OUT / "onset_prediction.csv", index=False)
    print(R.round(3).to_string(index=False))

    c = R[R.statistic == "nc_wb"]; ph = R[R.statistic == "ph_dim"]
    if len(c) and len(ph):
        cr, pr = abs(c.spearman.iloc[0]), abs(ph.spearman.iloc[0])
        print(f"\n  early clustering (nc_wb): |rho| = {cr:.3f}, LOO MAE = {c.loo_mae.iloc[0]:.0f}")
        print(f"  early PH-dim            : |rho| = {pr:.3f}, LOO MAE = {ph.loo_mae.iloc[0]:.0f}")
        print("  P1-3 answer: " + ("early clustering predicts onset at least as well as early "
                                   "PH-dim" if cr >= pr - 0.05 else
                                   "early PH-dim predicts onset better than early clustering"))
    n_beat = int(R.beats_hparams.sum()) if len(R) else 0
    print(f"\n  statistics that improve on the hyperparameter-only baseline: {n_beat}/{len(R)}")
    if n_beat == 0:
        print("  -> none beats hyperparameters: all are retrospective detection, not prediction,")
        print("     the same standard that sank the equivariance order parameter.")
