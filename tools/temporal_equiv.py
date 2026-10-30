"""
TEMPORAL analysis of the locked order parameter.

The crux. Spearman(test_acc, equiv_R2) ~ 0.87 across runs is worryingly close to
"measuring generalization twice": if equiv_R2 rises exactly when test accuracy rises,
it is a re-description of accuracy, not a mechanism. The temporal lead is what would
make it more.

Computes equiv_R2 (locked: max of left/right, per-run null) at EVERY checkpoint of
every completed run, then asks two separate questions:

  1. TIMING  -- does equiv_R2 reach half its rise before test accuracy does?
  2. CAUSAL  -- restricted to runs that have NOT yet generalized at a cutoff E, does
                equiv_R2 observed up to E carry information about the future onset
                that a censoring-aware baseline does not? This is the strict test,
                the same protocol that killed the original onset predictor.

Efficiency note: TopoRun.traj is a lazy NpzFile, so indexing it per checkpoint
re-decompresses the whole array. Here the weight array is pulled into memory once
per run, which is what makes a full per-checkpoint sweep affordable.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from analysis.topo_runs import TopoRun
from analysis.group_structure import (diff_table_for, right_diff_table, equivariance_r2)

OUT = REPO / "RESULTS"


def run_series(run: TopoRun, Dl, Dr, null_at=(0, 0.5, 1.0), n_null=8, seed=0):
    """equiv_R2 (max side) at every checkpoint, plus a null at a few checkpoints."""
    W0 = np.array(run.traj["W0"])           # pull into memory ONCE
    eps = np.array(run.traj["epochs"])
    p = run.p
    arch = run.meta.get("arch", "mlp")
    rows = []
    rng = np.random.default_rng(seed)
    null_idx = {int(f * (len(eps) - 1)) for f in null_at}
    for i in range(len(eps)):
        W = W0[i]
        E = W[:p] if arch == "transformer" else W[:, :p].T
        l = equivariance_r2(E, Dl)
        r = equivariance_r2(E, Dr) if Dr is not None else np.nan
        val = float(np.nanmax([l, r]))
        rec = {"epoch": int(eps[i]), "equiv_R2": val, "left": l, "right": r}
        if i in null_idx:
            ns = []
            for _ in range(n_null):
                Ep = E[rng.permutation(E.shape[0])]
                ns.append(np.nanmax([equivariance_r2(Ep, Dl),
                                     equivariance_r2(Ep, Dr) if Dr is not None else np.nan]))
            rec["null"] = float(np.mean(ns))
        rows.append(rec)
    df = pd.DataFrame(rows)
    c = run.curves.set_index("epoch")
    df["test_acc"] = [float(c.loc[e, "test_acc"]) if e in c.index else np.nan for e in df.epoch]
    df["train_acc"] = [float(c.loc[e, "train_acc"]) if e in c.index else np.nan for e in df.epoch]
    return df


def rise_epoch(epochs, y, frac=0.5, tail=5):
    """Epoch at which y first reaches `frac` of its total rise (baseline -> plateau)."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 6:
        return np.nan
    e, y = np.asarray(epochs)[ok], y[ok]
    base = np.median(y[:tail]); top = np.median(y[-tail:])
    if not np.isfinite(top - base) or (top - base) <= 1e-9:
        return np.nan
    hit = np.where(y >= base + frac * (top - base))[0]
    return float(e[hit[0]]) if hit.size else np.nan


def main() -> int:
    runs = []
    for d in sorted((REPO / "runs_topo").glob("*")):
        if not (d / "trajectory.npz").exists() or not (d / "curves.csv").exists():
            continue
        r = TopoRun(d)
        if r.meta.get("task") not in ("add", "perm"):
            continue
        runs.append(r)
    print(f"{len(runs)} completed runs", flush=True)

    tables = {}
    allser, timing = [], []
    for r in runs:
        key = (r.meta["task"], r.p)
        if key not in tables:
            tables[key] = (diff_table_for(*key), right_diff_table(*key))
        Dl, Dr = tables[key]
        if Dl is None:
            continue
        df = run_series(r, Dl, Dr)
        df["run"] = r.path.name; df["task"] = r.meta["task"]; df["p"] = r.p
        df["tf"] = r.meta["train_fraction"]; df["seed"] = r.meta["seed"]
        df["arch"] = r.meta.get("arch", "mlp")
        allser.append(df)

        if r.curves.test_acc.max() >= 0.5:          # only runs with a real transition
            t_acc = rise_epoch(df.epoch, df.test_acc)
            t_eq = rise_epoch(df.epoch, df.equiv_R2)
            timing.append({"run": r.path.name, "task": r.meta["task"], "p": r.p,
                           "tf": r.meta["train_fraction"], "seed": r.meta["seed"],
                           "arch": r.meta.get("arch", "mlp"),
                           "final_test": float(r.curves.test_acc.iloc[-1]),
                           "t_test50": t_acc, "t_equiv50": t_eq,
                           "lead": (t_acc - t_eq) if np.isfinite(t_acc) and np.isfinite(t_eq) else np.nan})
        print(f"  {r.path.name[:60]}", flush=True)

    S = pd.concat(allser, ignore_index=True); S.to_csv(OUT / "temporal_equiv_series.csv", index=False)
    T = pd.DataFrame(timing); T.to_csv(OUT / "temporal_equiv_timing.csv", index=False)
    print(f"\nwrote {len(S)} checkpoint rows, {len(T)} runs with a transition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
