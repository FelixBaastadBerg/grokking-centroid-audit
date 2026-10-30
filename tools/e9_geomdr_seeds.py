"""
E9 -- GeomDR with more seeds, CIs on the acceleration, and a BROAD mediator set logged at both a
fixed epoch and a matched training state.

The point is not to re-establish that the regulariser accelerates grokking (it does), but to test
whether the intervention is SELECTIVE. If it moves effective rank, spectral entropy, TwoNN,
PH-dim, silhouette, NC1, covariance trace, representation and weight norms, margins and gradient
norm all at once, then no single one of them is identified as the mediator.

Two readout points, because a fixed epoch confounds "what the regulariser changed" with "how far
training got":
    fixed      epoch 2000, before any run groks
    matched    the first checkpoint at which train accuracy first reaches 1.0 (memorisation),
               so runs are compared at the same training STATE rather than the same step
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
from tools.geomdr import spectral_entropy_t

OUT = REPO / "RESULTS"


def mediators(H, y, model, logits, loss_val, gnorm):
    Hn = H.detach().numpy()
    out = {"eff_rank": geom.effective_rank(Hn), "pca_pr": geom.pca_participation_ratio(Hn),
           "twonn": geom.twonn(Hn), "mle_k10": geom.mle_id(Hn, 10)}
    pdim, _ = geom.ph_dim(Hn)
    out["ph_dim"] = pdim
    Xc = Hn - Hn.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    s = s[s > 0]; pr = s / s.sum()
    out["spec_entropy"] = float(-(pr * np.log(pr)).sum())
    out["cov_trace"] = float(s.sum() / max(len(Hn) - 1, 1))
    out["rep_norm"] = float(np.linalg.norm(Hn) / np.sqrt(len(Hn)))
    out.update(geom.neural_collapse(Hn, y))
    out.update(geom.clusteredness(Hn))
    try:
        out["sil_true"] = float(silhouette_score(Hn, y)) if len(np.unique(y)) > 1 else np.nan
    except Exception:
        out["sil_true"] = np.nan
    out["weight_norm"] = float(np.sqrt(sum((p.detach() ** 2).sum().item()
                                           for p in model.parameters())))
    lg = logits.detach()
    top2 = torch.topk(lg, 2, dim=1).values
    out["margin"] = float((top2[:, 0] - top2[:, 1]).mean())
    out["train_loss"] = float(loss_val)
    out["grad_norm"] = float(gnorm)
    return out


def run(lam, seed, p=23, tf=0.7, epochs=30000, save_every=250, hidden=200, lr=1e-2, wd=0.03):
    torch.manual_seed(seed)
    pairs, y, n_sym, _ = build_task("add", p)
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
        if lam > 0:
            loss = loss + lam * spectral_entropy_t(model.activations[-2])
        loss.backward()
        gn = float(torch.sqrt(sum((q.grad ** 2).sum() for q in model.parameters()
                                  if q.grad is not None)))
        opt.step()
        if ep % save_every == 0:
            model.eval()
            with torch.no_grad():
                otr = model(Xtr, keep_activations=True)
                H = model.activations[-2]
                ote = model(Xte)
                r = {"epoch": ep, "lam": lam, "seed": seed,
                     "train_acc": float((otr.argmax(1) == Ytr).float().mean()),
                     "test_acc": float((ote.argmax(1) == Yte).float().mean()),
                     "test_loss": float(stablemax_cross_entropy(ote, Yte,
                                                                dtype=torch.float64).item())}
                r.update(mediators(H, Ytr.numpy(), model, otr, loss.item(), gn))
                rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    lams = [0.0, 0.001, 0.003, 0.01]
    seeds = [0, 1, 2, 3, 4]
    alld, t0 = [], time.time()
    for lam in lams:
        for sd in seeds:
            try:
                d = run(lam, sd)
            except Exception as e:
                print(f"  lam={lam} seed={sd} FAILED: {type(e).__name__} {e}", flush=True)
                continue
            fin = d.test_acc.values[-3:].mean()
            tg = d[d.test_acc >= 0.5 * fin].epoch.min() if fin >= 0.5 else np.nan
            d["t_grok"] = tg
            mem = d[d.train_acc >= 1.0].epoch.min() if (d.train_acc >= 1.0).any() else np.nan
            d["t_mem"] = mem
            alld.append(d)
            print(f"  lam={lam:<6} seed={sd}  final_test={fin:.3f}  t_grok={tg}  t_mem={mem}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            pd.concat(alld, ignore_index=True).to_csv(OUT / "e9_geomdr.csv", index=False)
    D = pd.concat(alld, ignore_index=True)
    D.to_csv(OUT / "e9_geomdr.csv", index=False)
    print(f"\nwrote RESULTS/e9_geomdr.csv {D.shape}")
