"""
Onset-predictor evaluation harness.

Implements BOTH the paper's protocol and an honest one, so the gap between them
is measurable rather than argued about:

  leaky  : q* chosen on ALL runs (min std of the t_q - T_grok offset), then only
           the offset is leave-one-out cross-validated.  This is what
           dimensions.ipynb does (CELL 47 ranks q over every run; CELL 67 reads
           table.iloc[0]; CELL 65 CVs the offset alone).
  nested : q* AND the offset are both chosen inside each held-out fold.

Two different R^2 numbers are reported, because they mean different things:

  r2_fit : R^2 of an OLS fit of actual ~ predicted.  This is what the paper
           reports ("fit a linear model ... R^2 is 0.98").  It is invariant to any
           constant or affine miscalibration, so it measures CORRELATION, not
           predictive accuracy.
  r2_oos : 1 - SSE/SST, the proper out-of-sample skill score against the training
           mean.  Can go negative.  This is the number that answers "is this
           better than guessing the mean".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .runlib import Run, fit_logistic, t_of_q, HP_KEYS

Q_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


# ==========================================================================
# skill metrics
# ==========================================================================
def r2_fit(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 of OLS y_true ~ y_pred (the paper's definition). Correlation-like."""
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3 or np.std(y_pred[ok]) < 1e-12:
        return np.nan
    r = np.corrcoef(y_true[ok], y_pred[ok])[0, 1]
    return float(r ** 2)


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray, y_ref_mean: float | None = None) -> float:
    """1 - SSE/SST skill score. Negative means worse than predicting the mean."""
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return np.nan
    yt, yp = y_true[ok], y_pred[ok]
    ref = np.mean(yt) if y_ref_mean is None else y_ref_mean
    sst = np.sum((yt - ref) ** 2)
    if sst <= 0:
        return np.nan
    return float(1.0 - np.sum((yt - yp) ** 2) / sst)


def summarise(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[ok], y_pred[ok]
    err = yp - yt
    if len(err) == 0:
        return {"method": label, "n": 0}
    return {
        "method": label,
        "n": int(len(err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "medae": float(np.median(np.abs(err))),
        "r2_fit": r2_fit(yt, yp),
        "r2_oos": r2_oos(yt, yp),
        "pi_lo": float(np.percentile(err, 2.5)),
        "pi_hi": float(np.percentile(err, 97.5)),
        "norm_mae": float(np.mean(np.abs(err)) / np.median(yt)) if np.median(yt) > 0 else np.nan,
    }


# ==========================================================================
# signal providers -- everything that can play the role of "the metric"
# ==========================================================================
def signal_dim(layer: str, metric: str):
    def f(run: Run) -> pd.Series:
        return run.dim_series(layer, metric)
    f.__name__ = f"{layer}|{metric}"
    return f


def signal_metrics(metric_name: str, input_type: str):
    def f(run: Run) -> pd.Series:
        return run.series(metric_name, input_type)
    f.__name__ = f"{input_type}|{metric_name}"
    return f


def signal_weights_l2(layer: str):
    def f(run: Run) -> pd.Series:
        m = run.metrics
        sel = m[(m.metric_name == "weights_l2") & (m.layer == layer)]
        if sel.empty:
            return pd.Series(dtype=float)
        return sel.groupby("epoch")["value"].mean().sort_index()
    f.__name__ = f"weights_l2|{layer}"
    return f


def control_fixed_sigmoid(midpoint: float = 20000.0, k: float = 3e-4):
    """Pure null: the SAME sigmoid for every run. Knows nothing about the model."""
    def f(run: Run) -> pd.Series:
        idx = run.test_acc.index
        return pd.Series(1.0 / (1.0 + np.exp(-k * (idx.values - midpoint))), index=idx)
    f.__name__ = "CONTROL|fixed_sigmoid"
    return f


def control_phase_shuffled(inner, seed: int = 0):
    """Keeps the marginal distribution, destroys temporal ordering."""
    def f(run: Run) -> pd.Series:
        s = inner(run)
        if len(s) == 0:
            return s
        rng = np.random.default_rng(seed + abs(hash(run.run_id)) % 10000)
        return pd.Series(rng.permutation(s.values), index=s.index)
    f.__name__ = f"CONTROL|phase_shuffled({inner.__name__})"
    return f


def control_time_reversed(inner):
    """Keeps the sigmoid shape, destroys causal alignment with grokking."""
    def f(run: Run) -> pd.Series:
        s = inner(run)
        if len(s) == 0:
            return s
        return pd.Series(s.values[::-1], index=s.index)
    f.__name__ = f"CONTROL|time_reversed({inner.__name__})"
    return f


# ==========================================================================
# per-run timing-marker table
# ==========================================================================
def build_tq_table(runs: dict[str, Run], signal, q_grid=Q_GRID,
                   target: str = "t_grok") -> pd.DataFrame:
    """
    For each run: fit a logistic to the signal, emit t_q for every q, plus the
    target onset time and the hyperparameters needed for granulation.
    """
    rows = []
    for rid, r in runs.items():
        s = signal(r)
        if s is None or len(s.dropna()) < 10:
            continue
        fitm = fit_logistic(s)
        if fitm is None:
            continue
        tgt = getattr(r, target)
        if not np.isfinite(tgt):
            continue
        row = {"run_id": rid, "target": float(tgt),
               "k": fitm["k"], "t0": fitm["t0"]}
        for q in q_grid:
            row[f"tq_{q:.2f}"] = t_of_q(fitm, float(q))
        for k in HP_KEYS:
            row[k] = r.meta[k]
        rows.append(row)
    return pd.DataFrame(rows)


def _pick_q(df: pd.DataFrame, q_grid) -> float:
    """Paper's rule: q* minimises the across-run std of (t_q - T_grok)."""
    best_q, best_std = np.nan, np.inf
    for q in q_grid:
        col = f"tq_{q:.2f}"
        if col not in df:
            continue
        d = (df[col] - df["target"]).to_numpy(float)
        d = d[np.isfinite(d)]
        if len(d) < 3:
            continue
        sd = float(np.std(d, ddof=1))
        if sd < best_std:
            best_std, best_q = sd, float(q)
    return best_q


# ==========================================================================
# evaluation protocols
# ==========================================================================
def eval_leaky(df: pd.DataFrame, q_grid=Q_GRID) -> tuple[dict, np.ndarray, np.ndarray]:
    """q* on all runs, then LOOCV the offset only. Reproduces the notebook."""
    q = _pick_q(df, q_grid)
    if not np.isfinite(q):
        return {"method": "leaky", "n": 0}, np.array([]), np.array([])
    x = df[f"tq_{q:.2f}"].to_numpy(float)
    y = df["target"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    preds = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        preds[i] = x[i] - np.mean(x[tr] - y[tr])
    out = summarise(y, preds, "leaky (q* on all runs)")
    out["q_star"] = q
    return out, y, preds


def eval_nested(df: pd.DataFrame, q_grid=Q_GRID) -> tuple[dict, np.ndarray, np.ndarray]:
    """q* AND offset chosen strictly inside each fold."""
    y = df["target"].to_numpy(float)
    n = len(y)
    preds = np.full(n, np.nan)
    qs = []
    for i in range(n):
        tr_mask = np.arange(n) != i
        q = _pick_q(df.iloc[tr_mask], q_grid)
        if not np.isfinite(q):
            continue
        qs.append(q)
        x = df[f"tq_{q:.2f}"].to_numpy(float)
        if not np.isfinite(x[i]):
            continue
        d = (x[tr_mask] - y[tr_mask])
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        preds[i] = x[i] - np.mean(d)
    out = summarise(y, preds, "nested (q* inside fold)")
    out["q_star_mode"] = float(pd.Series(qs).mode().iloc[0]) if qs else np.nan
    out["q_star_n_distinct"] = int(len(set(qs)))
    return out, y, preds


# ---- baselines ----
def eval_global_mean(df: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    y = df["target"].to_numpy(float)
    n = len(y)
    preds = np.array([np.mean(np.delete(y, i)) for i in range(n)])
    return summarise(y, preds, "BASELINE global mean (LOO)"), y, preds


def eval_granule_mean(df: pd.DataFrame, fixed: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    """Predict a run's T_grok as the mean T_grok of the OTHER runs in its granule."""
    y = df["target"].to_numpy(float)
    preds = np.full(len(y), np.nan)
    keys = df[fixed].astype(str).agg("|".join, axis=1).to_numpy()
    for i in range(len(y)):
        same = (keys == keys[i]) & (np.arange(len(y)) != i)
        if same.sum() >= 1:
            preds[i] = np.mean(y[same])
    return summarise(y, preds, f"BASELINE granule mean (fixed={'+'.join(fixed)})"), y, preds


def eval_hparam_only(df: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Hyperparameters alone, LOOCV. Linear model on log(lr), log(wd+eps),
    train_fraction plus the lr:wd interaction -- deliberately simple so it cannot
    be accused of being a tuned competitor.
    """
    y = df["target"].to_numpy(float)
    lr = np.log10(df["lr"].to_numpy(float))
    wd = np.log10(df["weight_decay"].to_numpy(float) + 1e-5)
    tf = df["train_fraction"].to_numpy(float)
    X = np.column_stack([np.ones_like(y), lr, wd, tf, lr * wd])
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        try:
            beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            preds[i] = X[i] @ beta
        except np.linalg.LinAlgError:
            pass
    return summarise(y, preds, "BASELINE hyperparameters only (LOO)"), y, preds


def eval_all(df: pd.DataFrame, q_grid=Q_GRID) -> pd.DataFrame:
    rows = []
    for fn in (eval_leaky, eval_nested, eval_global_mean, eval_hparam_only):
        res, _, _ = fn(df) if fn not in (eval_leaky, eval_nested) else fn(df, q_grid)
        rows.append(res)
    for fixed in (["train_fraction", "lr"], ["train_fraction", "weight_decay"], ["lr", "weight_decay"]):
        res, _, _ = eval_granule_mean(df, fixed)
        rows.append(res)
    return pd.DataFrame(rows)
