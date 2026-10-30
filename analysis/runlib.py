"""
Core data layer for the audit: run loading, T_grok definition, logistic fits,
and the ONE discovery/confirmation split.

Design rules enforced here:
  * run identity comes from run_meta.json, never from a directory name (D1)
  * T_grok is the logistic midpoint of test accuracy, everywhere (paper fix D1)
  * the discovery/confirmation split is deterministic, computed once, and cached
    to disk so it cannot drift between scripts
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data_runs"
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

# --- run filter, matching dimensions.ipynb CELL 1 so numbers stay comparable ---
ACC_THRESH = 0.90
EARLY_EPOCH_CUTOFF = 5000

MIN_POINTS_FIT = 10
SMOOTH_WINDOW = 9
SMOOTH_POLY = 2

HP_KEYS = ["train_fraction", "lr", "weight_decay"]


# ==========================================================================
# logistic machinery -- ported verbatim from dimensions.ipynb CELL 38 so that
# any difference in results is attributable to protocol, not to reimplementation
# ==========================================================================
def logistic(t, lo, hi, k, t0):
    return lo + (hi - lo) / (1.0 + np.exp(-k * (t - t0)))


def smooth_series(s: pd.Series, window: int = SMOOTH_WINDOW, poly: int = SMOOTH_POLY) -> pd.Series:
    s = s.dropna()
    if len(s) < 7:
        return s
    w = window if window % 2 == 1 else window + 1
    w = min(w, len(s) if len(s) % 2 == 1 else len(s) - 1)
    if w < 5:
        return s
    y = savgol_filter(s.values, window_length=w, polyorder=min(poly, w - 1))
    return pd.Series(y, index=s.index, name=s.name)


def _robust_init(t, y):
    lo0 = np.median(y[: max(3, len(y) // 10)])
    hi0 = np.median(y[-max(3, len(y) // 10):])
    mid = (lo0 + hi0) / 2
    idx = np.where(y >= mid)[0] if hi0 >= lo0 else np.where(y <= mid)[0]
    t00 = float(t[idx[0]]) if len(idx) else float(np.median(t))
    tr = float(t.max() - t.min()) if len(t) else 1.0
    return lo0, hi0, 10.0 / tr, t00


def fit_logistic(series: pd.Series) -> dict | None:
    s = series.dropna()
    if len(s) < MIN_POINTS_FIT:
        return None
    s_sm = smooth_series(s)
    t = s_sm.index.values.astype(float)
    y = s_sm.values.astype(float)
    if np.nanstd(y) < 1e-10:
        return None

    lo0, hi0, k0, t00 = _robust_init(t, y)
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    pad = 2.0 * (y_max - y_min + 1e-8)
    bounds = ([y_min - pad, y_min - pad, 1e-8, float(t.min())],
              [y_max + pad, y_max + pad, 1.0, float(t.max())])
    try:
        popt, pcov = curve_fit(logistic, t, y, p0=[lo0, hi0, k0, t00],
                               bounds=bounds, maxfev=20000)
    except Exception:
        return None

    lo, hi, k, t0 = popt
    se_t0 = np.nan
    if pcov is not None and np.all(np.isfinite(pcov)) and pcov.shape == (4, 4):
        se_t0 = float(np.sqrt(max(pcov[3, 3], 0.0)))
    return {"lo": float(lo), "hi": float(hi), "k": float(k), "t0": float(t0),
            "se_t0": se_t0, "series_sm": s_sm}


def t_of_q(fit: dict, q: float) -> float:
    """Timing marker: t_q = t0 + (1/k) ln(q/(1-q))."""
    k, t0 = fit["k"], fit["t0"]
    if not np.isfinite(k) or k <= 0:
        return np.nan
    return float(t0 + (1.0 / k) * np.log(q / (1.0 - q)))


# ==========================================================================
# runs
# ==========================================================================
@dataclass
class Run:
    run_id: str
    meta: dict
    metrics: pd.DataFrame
    dims: pd.DataFrame
    _cache: dict = field(default_factory=dict)

    # ---- series accessors ----
    def series(self, metric_name: str, input_type: str = "test") -> pd.Series:
        m = self.metrics
        sel = m[(m.metric_name == metric_name) & (m.input_type == input_type)]
        if sel.empty:
            return pd.Series(dtype=float)
        return sel.groupby("epoch")["value"].mean().sort_index()

    def dim_series(self, layer: str, metric: str) -> pd.Series:
        d = self.dims
        sel = d[(d.layer == layer) & (d.metric_name == metric)]
        if sel.empty:
            return pd.Series(dtype=float)
        return sel.groupby("epoch")["value"].mean().sort_index()

    @property
    def test_acc(self) -> pd.Series:
        return self.series("accuracy", "test")

    @property
    def train_acc(self) -> pd.Series:
        return self.series("accuracy", "train")

    @property
    def gen_gap(self) -> pd.Series:
        tr, te = self.train_acc, self.test_acc
        idx = tr.index.intersection(te.index)
        return (tr.loc[idx] - te.loc[idx]).sort_index()

    # ---- targets ----
    @property
    def t_grok(self) -> float:
        """T_grok := logistic midpoint of the TEST ACCURACY curve (standardised)."""
        if "t_grok" not in self._cache:
            f = fit_logistic(self.test_acc)
            self._cache["t_grok"] = float(f["t0"]) if f else np.nan
            self._cache["t_grok_fit"] = f
        return self._cache["t_grok"]

    @property
    def t_grok_fit(self) -> dict | None:
        _ = self.t_grok
        return self._cache["t_grok_fit"]

    @property
    def t_grok_gap(self) -> float:
        """Alternative target: logistic midpoint of the gap-collapse curve."""
        if "t_grok_gap" not in self._cache:
            f = fit_logistic(self.gen_gap)
            self._cache["t_grok_gap"] = float(f["t0"]) if f else np.nan
        return self._cache["t_grok_gap"]

    def t_grok_threshold(self, thresh: float = 0.5) -> float:
        """Sanity-check target only: first epoch smoothed test acc crosses `thresh`."""
        s = smooth_series(self.test_acc)
        hit = s[s >= thresh]
        return float(hit.index[0]) if len(hit) else np.nan

    @property
    def hp(self) -> tuple:
        return tuple(self.meta[k] for k in HP_KEYS)


def _passes_filter(metrics: pd.DataFrame) -> tuple[bool, str]:
    sel = metrics[(metrics.metric_name == "accuracy") & (metrics.input_type == "test")]
    if sel.empty:
        return False, "missing test accuracy"
    acc = sel.groupby("epoch")["value"].mean().sort_index()
    if not np.isfinite(acc.max()):
        return False, "test accuracy all-nan"
    if acc.max() < ACC_THRESH:
        return False, f"never reaches {ACC_THRESH} (max={acc.max():.3f})"
    early = acc[acc.index <= EARLY_EPOCH_CUTOFF]
    if len(early) and (early >= ACC_THRESH).any():
        return False, "too-early grok"
    return True, ""


def load_runs(verbose: bool = False) -> tuple[dict[str, Run], list[dict]]:
    runs, dropped = {}, []
    for d in sorted(DATA.iterdir()):
        if not (d / "run_meta.json").exists():
            continue
        meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(d / "metrics.csv")
        keep, why = _passes_filter(metrics)
        if not keep:
            dropped.append({"run_id": d.name, "reason": why})
            if verbose:
                print(f"[DROP] {d.name[:60]} -> {why}")
            continue
        runs[d.name] = Run(run_id=d.name, meta=meta, metrics=metrics,
                           dims=pd.read_csv(d / "dimensions.csv"))
    return runs, dropped


# ==========================================================================
# THE split -- computed once, cached, never re-drawn
# ==========================================================================
SPLIT_PATH = RESULTS / "split.json"
CONFIRM_FRACTION = 0.30


def make_split(runs: dict[str, Run], force: bool = False) -> dict:
    """
    Split by held-out GRANULE so hyperparameter families cannot leak across.

    A granule is a (train_fraction, lr) cell; whole granules go to one side.
    Assignment is a deterministic hash of the granule key -- it does not depend on
    iteration order, run count, or anything that could be nudged to flatter a result.
    """
    if SPLIT_PATH.exists() and not force:
        return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))

    granules = sorted({(r.meta["train_fraction"], r.meta["lr"]) for r in runs.values()})

    def h(g) -> float:
        return int(hashlib.sha256(repr(g).encode()).hexdigest()[:12], 16) / 16 ** 12

    scored = sorted(granules, key=h)
    n_conf = max(1, round(CONFIRM_FRACTION * len(scored)))
    conf_granules = {repr(g) for g in scored[:n_conf]}

    assign = {}
    for rid, r in runs.items():
        g = (r.meta["train_fraction"], r.meta["lr"])
        assign[rid] = "confirmation" if repr(g) in conf_granules else "discovery"

    split = {
        "created_by": "analysis/runlib.py:make_split",
        "rule": "granule=(train_fraction,lr); deterministic sha256 ordering; whole granules held out",
        "confirm_fraction_target": CONFIRM_FRACTION,
        "n_granules": len(granules),
        "n_confirm_granules": n_conf,
        "confirmation_granules": sorted(conf_granules),
        "assignment": assign,
        "n_discovery": sum(v == "discovery" for v in assign.values()),
        "n_confirmation": sum(v == "confirmation" for v in assign.values()),
    }
    SPLIT_PATH.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def split_runs(runs: dict[str, Run]) -> tuple[dict[str, Run], dict[str, Run]]:
    split = make_split(runs)
    a = split["assignment"]
    disc = {k: v for k, v in runs.items() if a.get(k) == "discovery"}
    conf = {k: v for k, v in runs.items() if a.get(k) == "confirmation"}
    return disc, conf


def run_table(runs: dict[str, Run]) -> pd.DataFrame:
    rows = []
    for rid, r in runs.items():
        rows.append({
            "run_id": rid,
            "arch": r.meta["arch"], "task": r.meta["task"],
            "train_fraction": r.meta["train_fraction"], "lr": r.meta["lr"],
            "weight_decay": r.meta["weight_decay"], "seed": r.meta["seed"],
            "t_grok": r.t_grok, "t_grok_gap": r.t_grok_gap,
            "t_grok_thresh50": r.t_grok_threshold(0.5),
            "final_test_acc": float(r.test_acc.iloc[-1]) if len(r.test_acc) else np.nan,
        })
    return pd.DataFrame(rows)
