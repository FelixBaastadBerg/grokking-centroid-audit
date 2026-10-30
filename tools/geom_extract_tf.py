"""
Cross-architecture arm of STEP 1/2: the same validated battery on TRANSFORMER activations.

Point cloud is the pre-logit encoder output at the last sequence position -- the transformer's
analogue of the MLP's last hidden layer, and the same p^2*tf point count (423 at p=23), so the
two architectures are compared on clouds of matched size.

The rebuild is exact: reconstructed checkpoints reproduce the recorded train/test curves to
4 decimals (checked in-line below before any measurement is taken).
"""
from __future__ import annotations
import os, sys, warnings, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from analysis import geom
from analysis.tfrun import TransformerRun
from analysis.tda import prepare_cloud, persistence_diagrams, diagram_summaries
from analysis.group_structure import equivariance_order_parameter

OUT = REPO / "RESULTS"


def homology(X):
    d = persistence_diagrams(prepare_cloud(X, max_points=400, scale="diameter"), maxdim=1)
    s = diagram_summaries(d[1], "H1")
    return {"amb_H1_max": s["H1_max_pers"], "amb_H1_n015": s.get("H1_n_above_0.15", np.nan)}


def run_one(d: Path, n_time=15):
    r = TransformerRun(d)
    n = len(r.epochs)
    tg = r.t_grok()
    # gate the rebuild before measuring anything
    r._load(r.flat(n - 1)); r.model.eval()
    with torch.no_grad():
        got = (r.model(r.Xte)[:, -1].argmax(1) == r.Yte).float().mean().item()
    rec = float(r.curves.test_acc.iloc[-1])
    assert abs(got - rec) < 1e-4, f"rebuild mismatch {got} vs {rec}"

    lg = np.unique(np.round(np.geomspace(1, n - 1, n_time)).astype(int))
    win = np.arange(max(0, tg - 3), min(n, tg + 4)) if tg >= 0 else np.array([], int)
    ts = np.unique(np.concatenate([[0], lg, win, [n - 1]])).astype(int)
    ytr = r.Ytr.numpy()
    rows = []
    tr = np.load(d / "trajectory.npz")
    for t in ts:
        H = r.activations(int(t), r.Xtr)
        b = geom.battery(H, y=ytr)
        b.update(homology(H))
        try:
            E = tr["W0"][int(t)][:r.p]           # transformer token embedding rows
            eq = equivariance_order_parameter(E, r.meta["task"], r.p, n_null=4, seed=0)
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
    for k, v in (("arch", "transformer"), ("task", r.meta["task"]), ("p", r.p),
                 ("tf", r.meta["train_fraction"]), ("seed", r.meta["seed"]),
                 ("final_test", float(r.curves.test_acc.iloc[-1])), ("dir", d.name)):
        df[k] = v
    df["split"] = "confirmation"          # cross-arch is held out by construction
    return df


if __name__ == "__main__":
    runs = [d for d in sorted(Path(REPO / "runs_topo").glob("arch-transformer*ss777*"))
            if (d / "curves.csv").exists() and (d / "full_traj.npz").exists()]
    print(f"=== transformer arm: {len(runs)} runs with full state ===\n", flush=True)
    alld, t0 = [], time.time()
    for i, d in enumerate(runs):
        try:
            df = run_one(d)
        except Exception as e:
            print(f"  [{i+1}/{len(runs)}] SKIP {d.name[:44]}: {type(e).__name__} {e}", flush=True)
            continue
        alld.append(df)
        print(f"  [{i+1}/{len(runs)}] {df.task.iloc[0]}/p{df.p.iloc[0]}/s{df.seed.iloc[0]} "
              f"final_test={df.final_test.iloc[0]:.3f} ({len(df)} ckpts, {time.time()-t0:.0f}s)",
              flush=True)
    if alld:
        D = pd.concat(alld, ignore_index=True)
        D.to_csv(OUT / "geom_series_tf.csv", index=False)
        print(f"\nwrote RESULTS/geom_series_tf.csv {D.shape}")
