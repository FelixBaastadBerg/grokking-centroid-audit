"""
T1.1 (compute pass) -- within-class spectral statistics per checkpoint, across the whole MLP
corpus.

v2 measured within-class effective rank at 5 stages of the 9 canonical seeds only (item 1,
v2_conditional_trajectory.csv), where it fell 58%. To promote that from a caveat to a
contribution we need it as a full time series across every run we have: grokked and non-grokked,
all tasks, random-label controls, both hidden layers.

The MLP forward pass is reimplemented in numpy straight from trajectory.npz (bias-free 2-layer
ReLU MLP, so [W0, W1, W2] is the whole model), which is ~10x faster than rebuilding the torch
module per checkpoint. The train/test split is regenerated with the run's recorded split_seed
(run_meta.json), falling back to the init seed exactly as train_topology.py does -- note
analysis/lsrun.py ignores split_seed, which is wrong for the ss777 runs; here it matters only
for which points enter the cloud.

Per checkpoint and per layer (first hidden, last hidden) on a fixed 400-point train subsample:
  within_eff_rank / within_pca_pr / within_spec_entropy   spectra of class-centred residuals
  global_eff_rank / global_pca_pr                          same statistics, uncentred by class
  centroid_eff_rank                                        spectrum of the K class centroids
  sil_true                                                 true-label silhouette
plus train/test accuracy at that epoch.  Output: RESULTS/v3_within_series.csv
"""
from __future__ import annotations
import json, os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
import torch
from sklearn.metrics import silhouette_score
from tools.train_topology import build_task

OUT = REPO / "RESULTS"
RUNS = REPO / "runs_topo"
N_SUB = 400
MAX_CK = 61


def spec_stats(M):
    """(eff_rank, pca_pr, spec_entropy) of the mean-centred rows of M."""
    Z = M - M.mean(0)
    s = np.linalg.svd(Z, compute_uv=False) ** 2
    s = s[s > 1e-30]
    if not len(s):
        return np.nan, np.nan, np.nan
    p = s / s.sum()
    ent = float(-(p * np.log(p)).sum())
    return float(np.exp(ent)), float(s.sum() ** 2 / (s ** 2).sum()), ent


def one_run(d):
    meta = json.loads((d / "run_meta.json").read_text())
    if meta["arch"] != "mlp":
        return []
    tr = np.load(d / "trajectory.npz")
    curves = pd.read_csv(d / "curves.csv").set_index("epoch")
    epochs, W0, W1 = tr["epochs"], tr["W0"], tr["W1"]

    p = meta["p"]
    task = meta["task"]
    pairs, y_all, n_sym, _ = build_task(task, p, sym_n=4 if (task == "perm" and p == 24) else 5)
    if meta.get("random_labels"):
        y_all = np.random.default_rng(10_000 + meta["seed"]).permutation(y_all)
    N = len(y_all)
    X = np.zeros((N, 2 * n_sym), np.float32)
    X[np.arange(N), pairs[:, 0]] = 1.0
    X[np.arange(N), n_sym + pairs[:, 1]] = 1.0
    split_seed = meta.get("split_seed", meta["seed"])
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(split_seed)).numpy()
    tr_idx = perm[:int(meta["train_fraction"] * N)]
    sub = np.sort(np.random.default_rng(0).choice(len(tr_idx), min(N_SUB, len(tr_idx)),
                                                  replace=False))
    Xs, ys = X[tr_idx][sub], y_all[tr_idx][sub]
    onehot = {c: ys == c for c in np.unique(ys)}

    idx = (np.unique(np.round(np.linspace(0, len(epochs) - 1, MAX_CK)).astype(int))
           if len(epochs) > MAX_CK else np.arange(len(epochs)))
    rows = []
    for i in idx:
        h = np.maximum(Xs @ W0[i].T, 0.0)
        for layer, H in (("first_hidden", h),
                         ("last_hidden", np.maximum(h @ W1[i].T, 0.0))):
            cen = np.zeros_like(H)
            for c, m in onehot.items():
                cen[m] = H[m].mean(0)
            res = H - cen
            w_er, w_pr, w_se = spec_stats(res)
            g_er, g_pr, _ = spec_stats(H)
            C = np.stack([H[m].mean(0) for m in onehot.values()])
            c_er, _, _ = spec_stats(C)
            try:
                sil = float(silhouette_score(H, ys)) if len(onehot) > 1 else np.nan
            except Exception:
                sil = np.nan
            ep = int(epochs[i])
            cv = curves.loc[ep] if ep in curves.index else None
            rows.append({"dir": d.name, "task": task, "p": p, "seed": meta["seed"],
                         "tf": meta["train_fraction"], "hidden": meta["hidden"],
                         "random_labels": bool(meta.get("random_labels")),
                         "ck": int(i), "epoch": ep, "layer": layer,
                         "train_acc": float(cv.train_acc) if cv is not None else np.nan,
                         "test_acc": float(cv.test_acc) if cv is not None else np.nan,
                         "within_eff_rank": w_er, "within_pca_pr": w_pr,
                         "within_spec_entropy": w_se,
                         "global_eff_rank": g_er, "global_pca_pr": g_pr,
                         "centroid_eff_rank": c_er, "sil_true": sil,
                         "m_per_class": float(len(ys) / len(onehot))})
    return rows


def main():
    inv = pd.read_csv(OUT / "_inventory.csv")
    dirs = [RUNS / r for r in inv[inv.arch == "mlp"].dir]
    # the random-label controls and the v3 overnight runs are not in the inventory
    known = set(inv.dir)
    extra = [d for d in sorted(RUNS.iterdir())
             if d.is_dir() and d.name not in known and (d / "trajectory.npz").exists()
             and ("randlab" in d.name or "__w" in d.name or "p-59" in d.name)]
    dirs += extra
    t0 = time.time()
    rows = []
    for k, d in enumerate(dirs, 1):
        try:
            rows += one_run(d)
        except Exception as e:
            print(f"  SKIP {d.name}: {type(e).__name__} {e}", flush=True)
        if k % 10 == 0:
            print(f"  [{k}/{len(dirs)}] ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "v3_within_series.csv", index=False)
    print(f"wrote v3_within_series.csv {D.shape} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
