"""
P1-4 -- external validity. The universality claim rests on tiny Z_23/Z_59 and small S_n; this adds
p = 97 and 113 modular addition, and (if there is runway) a NON-modular task, sparse parity.

Two questions, and we report the answer either way:
  1. do activations still enter the clustered regime at larger p, where there are more classes?
  2. does the E1 alpha-sweep still separate the confounded from the reliable estimators there?

A scope refinement we find ourselves is better than one a reviewer finds, so if clustering is
weaker at larger p that is stated plainly.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from sklearn.metrics import silhouette_score

from tools.train_topology import build_task
from models import MLP
from utils import stablemax_cross_entropy
from analysis import geom
from tools.linchpin import h1_stats
from tools.e1_centroid import decompose, recompose, ALPHAS

OUT = REPO / "RESULTS"
MAXPTS = 400


def sparse_parity(n_bits=20, k=3, n_samples=4000, seed=0):
    """k-of-n sparse parity: y = XOR of k fixed bits. Non-modular grokking task."""
    g = np.random.default_rng(seed)
    X = g.integers(0, 2, size=(n_samples, n_bits))
    idx = np.arange(k)
    y = X[:, idx].sum(1) % 2
    return X.astype(np.float32) * 2 - 1, y.astype(np.int64), 2


def train(task, p, seed, tf=0.6, epochs=8000, save_every=250, hidden=200, lr=1e-2, wd=0.03,
          max_points=MAXPTS):
    torch.manual_seed(seed)
    if task == "parity":
        Xn, yn, n_sym = sparse_parity(seed=seed)
        X = torch.as_tensor(Xn); Y = torch.as_tensor(yn)
    else:
        pairs, yn, n_sym, _ = build_task(task, p)
        N = len(yn)
        X = torch.zeros(N, 2 * n_sym)
        X[torch.arange(N), torch.as_tensor(pairs[:, 0])] = 1.0
        X[torch.arange(N), n_sym + torch.as_tensor(pairs[:, 1])] = 1.0
        Y = torch.as_tensor(yn, dtype=torch.long)
    N = len(Y)
    g = torch.Generator().manual_seed(777)
    perm = torch.randperm(N, generator=g)
    ntr = int(tf * N)
    Xtr, Ytr, Xte, Yte = X[perm[:ntr]], Y[perm[:ntr]], X[perm[ntr:]], Y[perm[ntr:]]
    model = MLP(input_size=X.shape[1], hidden_sizes=[hidden, hidden], output_size=n_sym, bias=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-25,
                            weight_decay=wd)
    ys = Ytr.numpy()
    sub = np.sort(np.random.default_rng(0).choice(len(ys), min(max_points, len(ys)),
                                                  replace=False))
    curve, snaps = [], {}
    for ep in range(epochs + 1):
        model.train(); opt.zero_grad()
        lo = model(Xtr, keep_activations=True)
        stablemax_cross_entropy(lo, Ytr, dtype=torch.float64).backward(); opt.step()
        if ep % save_every == 0:
            model.eval()
            with torch.no_grad():
                otr = model(Xtr, keep_activations=True)
                H = model.activations[-2].numpy()[sub]
                ote = model(Xte)
                tr = float((otr.argmax(1) == Ytr).float().mean())
                te = float((ote.argmax(1) == Yte).float().mean())
            sil = float(silhouette_score(H, ys[sub])) if len(np.unique(ys[sub])) > 1 else np.nan
            curve.append({"epoch": ep, "train_acc": tr, "test_acc": te, "silhouette": sil})
            if ep == 0:
                snaps["init"] = (H.copy(), ys[sub].copy())
            snaps["final"] = (H.copy(), ys[sub].copy())
    return pd.DataFrame(curve), snaps


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "modular"
    cfg = ([("add", 97), ("add", 113)] if which == "modular" else [("parity", 0)])
    t0 = time.time()
    op_rows, e1_rows, cur_rows = [], [], []
    for task, p in cfg:
        for seed in range(3):
            try:
                curve, snaps = train(task, p, seed)
            except Exception as e:
                print(f"  FAIL {task} p={p} seed{seed}: {type(e).__name__} {e}", flush=True)
                continue
            fin = curve.test_acc.values[-3:].mean()
            grok = fin >= 0.9
            # halfway from CHANCE to final, not half of final. With p classes chance is 1/p and
            # the distinction is negligible, but sparse parity has two classes, where "half of
            # final" is satisfied at initialisation and t_grok comes out 0.
            n_cls = 2 if task == "parity" else p
            chance = 1.0 / n_cls
            thr = chance + 0.5 * (fin - chance)
            tg = curve[curve.test_acc >= thr].epoch.min() if fin > chance else np.nan
            sil_i = curve.silhouette.iloc[0]
            sil_f = curve.silhouette.iloc[-1]
            sil_g = (curve[curve.epoch >= tg].silhouette.iloc[0]
                     if np.isfinite(tg) and (curve.epoch >= tg).any() else np.nan)
            op_rows.append({"arch": "mlp", "task": task, "p": p, "seed": seed, "tf": 0.6,
                            "sil_init": sil_i, "sil_t_grok": sil_g, "sil_final": sil_f,
                            "grokked": bool(grok), "final_test": fin, "t_grok": tg,
                            "source": "P1-4"})
            print(f"  {task}/p{p}/seed{seed}: final_test={fin:.3f} grokked={grok} "
                  f"sil {sil_i:+.3f} -> {sil_f:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
            # E1 alpha sweep on the converged activations
            H, ys = snaps["final"]
            mu, bet, wit = decompose(H, ys)
            for a in ALPHAS:
                Z = recompose(mu, bet, wit, a, 1.0)
                pdim, _ = geom.ph_dim(Z); cd, _ = geom.correlation_dim(Z)
                h = h1_stats(Z)
                e1_rows.append({"arch": "mlp", "task": task, "p": p, "seed": seed, "norm": "trace",
                                "sweep": "alpha", "alpha": a, "beta": 1.0, "ph_dim": pdim,
                                "corr_dim": cd, "twonn": geom.twonn(Z),
                                "mle_k10": geom.mle_id(Z, 10),
                                "pca_pr": geom.pca_participation_ratio(Z),
                                "eff_rank": geom.effective_rank(Z),
                                "H1_max": h["H1_max"], "H1_total": h["H1_total"],
                                "source": "P1-4"})
            pd.DataFrame(op_rows).to_csv(OUT / f"p1_4_operating_{which}.csv", index=False)
            pd.DataFrame(e1_rows).to_csv(OUT / f"p1_4_centroid_{which}.csv", index=False)
            # keep the trajectory too, so t_grok and the silhouette curve stay auditable
            c = curve.copy(); c["task"], c["p"], c["seed"] = task, p, seed
            cur_rows.append(c)
            pd.concat(cur_rows, ignore_index=True).to_csv(
                OUT / f"p1_4_curves_{which}.csv", index=False)
    print(f"\nwrote p1_4_operating_{which}.csv and p1_4_centroid_{which}.csv "
          f"({time.time()-t0:.0f}s)")
