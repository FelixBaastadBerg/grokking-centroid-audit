"""
NEGATIVE CONTROL on the temporal lead.

"Internal progress precedes the accuracy jump" is already established (Barak et al.
2022, hidden progress; Nanda et al. 2023, progress measures). So a lead by itself is
not evidence for equiv_R2 specifically. Two things must be shown:

  A. equiv_R2 leads MORE / differently than generic monotone internal signals
     (weight norms, train loss, embedding norm, a fixed random projection of the
     embedding, the embedding's participation ratio).
  B. The lead is not DEFINED in a way that makes it mechanically equal the plateau
     length. If every signal that moves at memorisation time gets
     lead = T_grok - t_memorize by construction, the statistic is vacuous.

For B we report, for each signal, both the raw lead and the *position of its rise
within the plateau*:

     phase = (t_signal50 - t_memorize) / (t_test50 - t_memorize)

phase ~ 0  -> the signal moves the instant memorisation completes (uninformative:
              its lead is just the plateau length)
phase ~ 1  -> the signal moves only when accuracy does (no lead)
0 < phase < 1 -> the signal marks a distinct intermediate event
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

OUT = REPO / "RESULTS"


def rise_epoch(epochs, y, frac=0.5, tail=5):
    """Epoch at which y first reaches `frac` of its total excursion, either direction."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 8:
        return np.nan
    e, y = np.asarray(epochs, float)[ok], y[ok]
    base = np.median(y[:tail]); top = np.median(y[-tail:])
    d = top - base
    if not np.isfinite(d) or abs(d) < 1e-12:
        return np.nan
    thr = base + frac * d
    hit = np.where(y >= thr)[0] if d > 0 else np.where(y <= thr)[0]
    return float(e[hit[0]]) if hit.size else np.nan


def main() -> int:
    S = pd.read_csv(OUT / "temporal_equiv_series.csv")
    rows = []
    rng = np.random.default_rng(0)

    for run, g in S.groupby("run"):
        d = REPO / "runs_topo" / run
        if not (d / "trajectory.npz").exists():
            continue
        r = TopoRun(d)
        c = r.curves.sort_values("epoch")
        if c.test_acc.max() < 0.5:
            continue

        # onset and memorisation times
        t_test = rise_epoch(c.epoch, c.test_acc)
        hit = c[c.train_acc >= 0.99]
        t_mem = float(hit.epoch.iloc[0]) if len(hit) else np.nan
        if not np.isfinite(t_test) or not np.isfinite(t_mem) or t_test <= t_mem:
            continue

        g = g.sort_values("epoch")
        sig = {"equiv_R2": (g.epoch.to_numpy(), g.equiv_R2.to_numpy())}

        # ---- embedding-derived controls, one pass over the weights ----
        W0 = np.array(r.traj["W0"]); eps = np.array(r.traj["epochs"]); p = r.p
        arch = r.meta.get("arch", "mlp")
        # projection width must come from the EMBEDDING's feature dim, which differs
        # by architecture (MLP: hidden; Transformer: d_model). Fixed within a run.
        E0 = W0[0][:p] if arch == "transformer" else W0[0][:, :p].T
        R = rng.normal(size=(E0.shape[1], 8))
        emb_norm, rp_var, part_ratio = [], [], []
        for i in range(len(eps)):
            W = W0[i]
            E = W[:p] if arch == "transformer" else W[:, :p].T
            Ec = E - E.mean(0, keepdims=True)
            emb_norm.append(float(np.linalg.norm(E, axis=1).mean()))
            rp_var.append(float((Ec @ R).var()))
            sv = np.linalg.svd(Ec, compute_uv=False) ** 2
            part_ratio.append(float(sv.sum() ** 2 / (sv ** 2).sum()) if sv.sum() > 0 else np.nan)
        sig["CTRL_emb_norm"] = (eps, np.array(emb_norm))
        sig["CTRL_randproj_var"] = (eps, np.array(rp_var))
        sig["CTRL_participation_ratio"] = (eps, np.array(part_ratio))

        # ---- curve-derived controls ----
        for col, name in (("train_loss", "CTRL_train_loss"), ("w0_norm", "CTRL_w0_norm"),
                          ("w1_norm", "CTRL_w1_norm"), ("w2_norm", "CTRL_w2_norm")):
            if col in c.columns:
                sig[name] = (c.epoch.to_numpy(), c[col].to_numpy())

        plateau = t_test - t_mem
        for name, (e, y) in sig.items():
            t50 = rise_epoch(e, y)
            rows.append({
                "run": run, "task": r.meta["task"], "p": r.p,
                "tf": r.meta["train_fraction"], "seed": r.meta["seed"],
                "arch": arch, "signal": name,
                "t_memorize": t_mem, "t_test50": t_test, "plateau": plateau,
                "t_sig50": t50,
                "lead": (t_test - t50) if np.isfinite(t50) else np.nan,
                "phase": ((t50 - t_mem) / plateau) if np.isfinite(t50) and plateau > 0 else np.nan,
            })
        print(f"  {run[:62]}", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "lead_controls.csv", index=False)
    print(f"\nwrote {len(D)} rows, {D.run.nunique()} runs, {D.signal.nunique()} signals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
