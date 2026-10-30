"""
CAUSAL (prefix) evaluation -- the test that decides whether this is PREDICTION
or merely retrospective DETECTION.

The paper's protocol fits a logistic to the WHOLE metric trajectory, including
epochs after grokking, then reports that t_q lines up with T_grok. That is
hindsight: to compute t_q you already had to watch the transition finish.

Here we truncate every run at an absolute observation cutoff E, fit the logistic
using ONLY epochs <= E, calibrate q* and the offset on the OTHER (also truncated)
runs, and predict the held-out run's T_grok. Runs whose true T_grok > E are
genuine forecasts; those are reported separately.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from .runlib import Run, fit_logistic, t_of_q, HP_KEYS
from .predictor import Q_GRID, _pick_q, summarise

def build_tq_table_truncated(runs: dict[str, Run], signal, cutoff: float,
                             q_grid=Q_GRID, target: str = "t_grok",
                             min_points: int = 10) -> pd.DataFrame:
    rows = []
    for rid, r in runs.items():
        s = signal(r)
        if s is None or len(s) == 0:
            continue
        s = s[s.index <= cutoff].dropna()
        if len(s) < min_points:
            continue
        f = fit_logistic(s)
        if f is None:
            continue
        tgt = getattr(r, target)
        if not np.isfinite(tgt):
            continue
        row = {"run_id": rid, "target": float(tgt), "k": f["k"], "t0": f["t0"],
               "is_forecast": bool(tgt > cutoff)}
        for q in q_grid:
            row[f"tq_{q:.2f}"] = t_of_q(f, float(q))
        for k in HP_KEYS:
            row[k] = r.meta[k]
        rows.append(row)
    return pd.DataFrame(rows)

def eval_causal(df: pd.DataFrame, q_grid=Q_GRID) -> dict:
    """Nested LOOCV on a truncated table; reports all runs and forecast-only runs."""
    if len(df) < 5:
        return {"n": 0}
    y = df["target"].to_numpy(float); n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        q = _pick_q(df.iloc[tr], q_grid)
        if not np.isfinite(q):
            continue
        x = df[f"tq_{q:.2f}"].to_numpy(float)
        if not np.isfinite(x[i]):
            continue
        d = (x[tr] - y[tr]); d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        preds[i] = x[i] - np.mean(d)
    out = summarise(y, preds, "causal nested")
    fc = df["is_forecast"].to_numpy(bool)
    if fc.sum() >= 3:
        out_fc = summarise(y[fc], preds[fc], "forecast-only")
        out.update({f"fc_{k}": v for k, v in out_fc.items() if k != "method"})
    out["n_forecast"] = int(fc.sum())
    # baseline on the same rows: LOO mean of the other runs' T_grok
    base = np.array([np.mean(np.delete(y, i)) for i in range(n)])
    out["base_mae"] = float(np.mean(np.abs(base - y)))
    if fc.sum() >= 3:
        out["fc_base_mae"] = float(np.mean(np.abs(base[fc] - y[fc])))
    return out
