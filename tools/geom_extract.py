"""
STEP 1/2 extraction -- run the validated battery on ACTIVATIONS across training.

Point cloud = pre-logit hidden activations of the TRAIN inputs (Ansuini's "last hidden layer",
the one whose ID is claimed to predict test accuracy). Objects are input pairs, so the count is
p^2 * train_fraction -- 423 at p=23 -- not |G|.

Sampling in time is log-spaced plus a dense window at t_grok, the scheme that memo 2 showed is
necessary: uniform sampling over 301 checkpoints puts only ~2 points before a t_grok at index 22.

Also recorded per checkpoint, so STEP 3 can ask whether geometry moves where homology does not:
  amb_H1_max / amb_b1  -- the homological null, on the SAME cloud at the SAME checkpoint
  equiv_R2             -- the one anchor we own that leads

Runs are split into DISCOVERY and CONFIRMATION halves by seed parity before anything is looked at.
"""
from __future__ import annotations
import os, sys, warnings, json, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from analysis import geom
from analysis.lsrun import LandscapeRun
from analysis.tda import prepare_cloud, persistence_diagrams, diagram_summaries
from analysis.group_structure import equivariance_order_parameter

OUT = REPO / "RESULTS"
N_TIME = 15
MAX_PTS = 600      # every estimator was validated at n = 400-800; p=59 gives 3481 points,
                   # where Ollivier-Ricci (O(n^2 log n)) and magnitude (O(n^3)) dominate runtime
                   # for no gain. Subsample is fixed per run so checkpoints stay comparable.


def homology(X, maxdim=1):
    d = persistence_diagrams(prepare_cloud(X, max_points=400, scale="diameter"), maxdim=maxdim)
    s = diagram_summaries(d[1], "H1")
    return {"amb_H1_max": s["H1_max_pers"], "amb_H1_n015": s.get("H1_n_above_0.15", np.nan)}


def run_one(d: Path, n_time=N_TIME):
    r = LandscapeRun(d)
    n = len(r.epochs)
    tg = r.t_grok()
    lg = np.unique(np.round(np.geomspace(1, n - 1, n_time)).astype(int))
    win = np.arange(max(0, tg - 3), min(n, tg + 4)) if tg >= 0 else np.array([], int)
    ts = np.unique(np.concatenate([[0], lg, win, [n - 1]])).astype(int)
    ytr = r.Ytr.numpy()
    sub = (np.random.default_rng(0).choice(len(ytr), MAX_PTS, replace=False)
           if len(ytr) > MAX_PTS else np.arange(len(ytr)))
    sub = np.sort(sub)
    ytr = ytr[sub]
    rows = []
    for t in ts:
        r._load(r.flat(int(t))); r.model.eval()
        with torch.no_grad():
            r.model(r.Xtr, keep_activations=True)
            H = r.model.activations[-2].numpy()[sub]
        b = geom.battery(H, y=ytr)
        b.update(homology(H))
        # representation-side anchor: embedding block, the object equiv_R2 is defined on
        try:
            E = r.W[0][int(t)][:, :r.meta["p"]].T
            eq = equivariance_order_parameter(E, r.meta["task"], r.meta["p"], n_null=4, seed=0)
            b["equiv_R2"] = eq["equiv_R2"] if isinstance(eq, dict) else float(eq)
        except Exception:
            b["equiv_R2"] = np.nan
        b.update({"t": int(t), "epoch": int(r.epochs[int(t)]), "t_grok": tg,
                  "train_acc": float(r.curves.train_acc.values[int(t)]),
                  "test_acc": float(r.curves.test_acc.values[int(t)]),
                  "test_loss": float(r.curves.test_loss.values[int(t)]),
                  "n_points": int(len(H))})
        rows.append(b)
    df = pd.DataFrame(rows)
    for k, v in (("arch", r.meta["arch"]), ("task", r.meta["task"]), ("p", r.meta["p"]),
                 ("tf", r.meta["train_fraction"]), ("seed", r.meta["seed"]),
                 ("final_test", float(r.curves.test_acc.iloc[-1])), ("dir", d.name)):
        df[k] = v
    return df


if __name__ == "__main__":
    inv = pd.read_csv(OUT / "_inventory.csv")
    mlp = inv[inv.arch == "mlp"].copy()
    # stratified subset: keep every (task,p,tf) cell, cap seeds per cell, keep non-grokked too
    keep = []
    for (task, p, tf), g in mlp.groupby(["task", "p", "tf"]):
        keep.extend(g.sort_values("seed").head(3).to_dict("records"))
    sel = pd.DataFrame(keep)
    sel["split"] = np.where(sel.seed % 2 == 0, "discovery", "confirmation")
    sel.to_csv(OUT / "geom_runs.csv", index=False)
    done = set()
    prev = OUT / "geom_series.csv"
    if prev.exists():
        pv = pd.read_csv(prev)
        done = set(pv.dir.unique())
        alld_prev = [pv]
    else:
        alld_prev = []
    sel = sel[~sel.dir.isin(done)]
    print(f"=== resuming: {len(done)} runs already done ===")
    print(f"=== extracting battery on {len(sel)} runs "
          f"({(sel.split=='discovery').sum()} discovery / {(sel.split=='confirmation').sum()} confirmation) ===",
          flush=True)
    print(f"    grokked {int((sel.test>=0.9).sum())}, not grokked {int((sel.test<0.9).sum())}\n", flush=True)
    alld, t0 = list(alld_prev), time.time()
    for i, rec in enumerate(sel.to_dict("records")):
        try:
            df = run_one(REPO / "runs_topo" / rec["dir"])
        except Exception as e:
            print(f"  [{i+1}/{len(sel)}] SKIP {rec['dir'][:44]}: {type(e).__name__} {e}", flush=True)
            continue
        df["split"] = rec["split"]
        alld.append(df)
        print(f"  [{i+1}/{len(sel)}] {rec['task']}/p{rec['p']}/tf{rec['tf']}/s{rec['seed']} "
              f"({len(df)} ckpts, {time.time()-t0:.0f}s)", flush=True)
        pd.concat(alld, ignore_index=True).to_csv(OUT / "geom_series.csv", index=False)
    D = pd.concat(alld, ignore_index=True)
    D.to_csv(OUT / "geom_series.csv", index=False)
    print(f"\nwrote RESULTS/geom_series.csv  {D.shape}")
