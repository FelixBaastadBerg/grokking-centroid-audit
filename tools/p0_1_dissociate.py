"""
P0-1 -- the dissociation experiment on real training.

Falsifier, stated before running (also written into RESULTS/dissociation_README.md):
    The clustering reading is FALSIFIED if the dimension-only loss accelerates grokking at least
    as much as the clustering-only loss, at matched effect size in each loss's target factor.

Matched effect size: strengths are calibrated in a pilot so that L_clust and L_dim produce
comparable-magnitude movement in silhouette and in within-class TwoNN respectively, before any
t_grok comparison is made.

Both losses passed the two-directional gate in tools/p0_1_losses.py.
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
from tools.p0_1_losses import loss_clust, loss_dim

OUT = REPO / "RESULTS"


def probe(H, y):
    Hn = H.detach().numpy()
    W = Hn - np.stack([Hn[y == c].mean(0) for c in y])
    rms = np.sqrt((W ** 2).sum(1).mean())
    Wn = W / rms if rms > 1e-12 else W
    out = {"silhouette": float(silhouette_score(Hn, y)) if len(np.unique(y)) > 1 else np.nan,
           "twonn_within": float(geom.twonn(Wn)), "mle_within": float(geom.mle_id(Wn, 10)),
           "twonn_all": float(geom.twonn(Hn)), "ph_dim_all": float(geom.ph_dim(Hn)[0]),
           "eff_rank": float(geom.effective_rank(Hn))}
    return out


def run(loss_type, strength, seed, task="add", p=23, tf=0.7, epochs=30000, save_every=250,
        hidden=200, lr=1e-2, wd=0.03):
    torch.manual_seed(seed)
    sym = 4 if task == "perm" else 5
    pairs, y, n_sym, _ = build_task(task, p, sym_n=sym)
    N = len(y)
    X = torch.zeros(N, 2 * n_sym)
    X[torch.arange(N), torch.as_tensor(pairs[:, 0])] = 1.0
    X[torch.arange(N), n_sym + torch.as_tensor(pairs[:, 1])] = 1.0
    Y = torch.as_tensor(y, dtype=torch.long)
    g = torch.Generator().manual_seed(777)
    perm = torch.randperm(N, generator=g)
    ntr = int(tf * N)
    Xtr, Ytr, Xte, Yte = X[perm[:ntr]], Y[perm[:ntr]], X[perm[ntr:]], Y[perm[ntr:]]
    model = MLP(input_size=2 * n_sym, hidden_sizes=[hidden, hidden], output_size=n_sym, bias=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-25,
                            weight_decay=wd)
    rows = []
    for ep in range(epochs + 1):
        model.train(); opt.zero_grad()
        logits = model(Xtr, keep_activations=True)
        loss = stablemax_cross_entropy(logits, Ytr, dtype=torch.float64)
        if loss_type != "base" and strength > 0:
            H = model.activations[-2]
            if loss_type == "clust":
                # `strength` IS the target between/within ratio; weight fixed at 1. The natural
                # ratio runs 0.002-0.30 (RESULTS/p0_1_natural_ratio.csv), so targets are set as
                # multiples of that trajectory rather than as absolute numbers.
                aux = loss_clust(H, Ytr, n_sym, target=strength)
                loss = loss + aux
            else:
                loss = loss + strength * loss_dim(H, Ytr, n_sym)
        loss.backward(); opt.step()
        if ep % save_every == 0:
            model.eval()
            with torch.no_grad():
                otr = model(Xtr, keep_activations=True)
                H = model.activations[-2]
                ote = model(Xte)
                r = {"epoch": ep, "loss_type": loss_type, "strength": strength, "seed": seed,
                     "task": task, "p": p,
                     "train_acc": float((otr.argmax(1) == Ytr).float().mean()),
                     "test_acc": float((ote.argmax(1) == Yte).float().mean())}
                r.update(probe(H, Ytr.numpy()))
                rows.append(r)
    return pd.DataFrame(rows)


def summarise(d):
    fin = d.test_acc.values[-3:].mean()
    tg = d[d.test_acc >= 0.5 * fin].epoch.min() if fin >= 0.5 else np.nan
    mem = d[d.train_acc >= 1.0].epoch.min() if (d.train_acc >= 1.0).any() else np.nan
    at = d[d.epoch == mem].iloc[0] if np.isfinite(mem) and (d.epoch == mem).any() else d.iloc[0]
    return {"t_grok": tg, "t_mem": mem, "final_test_acc": fin,
            "silhouette_at_mem": at.silhouette, "twonn_at_mem": at.twonn_within,
            "phdim_at_mem": at.ph_dim_all, "eff_rank_at_mem": at.eff_rank}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    t0 = time.time()

    if mode == "pilot":
        print("=== PILOT: calibrate strengths for matched effect size (Z23 add, seed 0) ===\n")
        rows = []
        for lt, strengths in (("base", [0.0]), ("clust", [0.6, 1.2, 2.4]),
                              ("dim", [0.003, 0.01, 0.03])):
            for st in strengths:
                d = run(lt, st, 0, epochs=15000)
                s = summarise(d)
                rows.append({"loss_type": lt, "strength": st, **s})
                print(f"  {lt:6s} s={st:<6} sil@mem={s['silhouette_at_mem']:+.3f} "
                      f"twonn@mem={s['twonn_at_mem']:.2f} t_grok={s['t_grok']} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "p0_1_pilot.csv", index=False)
        print("\nwrote RESULTS/p0_1_pilot.csv")
    else:
        cfg = [("add", 23), ("perm", 24)]
        STR = {"base": [0.0], "clust": [0.6, 1.2, 2.4], "dim": [0.003, 0.01, 0.03]}
        alld = []
        for task, p in cfg:
            for lt in ("base", "clust", "dim"):
                for st in STR[lt]:
                    for seed in range(3):
                        try:
                            d = run(lt, st, seed, task=task, p=p)
                        except Exception as e:
                            print(f"  FAIL {task} {lt} {st} s{seed}: {type(e).__name__} {e}",
                                  flush=True); continue
                        s = summarise(d)
                        alld.append({"task": task, "arch": "mlp", "seed": seed, "loss_type": lt,
                                     "strength": st, **s})
                        print(f"  {task}/{lt}/s={st}/seed{seed}: t_grok={s['t_grok']} "
                              f"sil={s['silhouette_at_mem']:+.3f} twonn={s['twonn_at_mem']:.2f} "
                              f"({time.time()-t0:.0f}s)", flush=True)
                        pd.DataFrame(alld).to_csv(OUT / "dissociation.csv", index=False)
        pd.DataFrame(alld).to_csv(OUT / "dissociation.csv", index=False)
        print(f"\nwrote RESULTS/dissociation.csv ({len(alld)} runs)")
