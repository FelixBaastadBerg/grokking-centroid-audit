"""
Outstanding run 1 -- GeomDR (2607.11666): effective-dimensionality regularization as a
grokking accelerator, and whether the acceleration tracks CLUSTERING.

Their claim is causal: a spectral regularizer on hidden-representation dimensionality
substantially alters grokking dynamics (reported up to ~52x acceleration). Our question is not
whether it works -- we expect it to -- but what it MOVES. If the regularizer is a clustering
lever, then effective dimensionality and clusteredness fall together and the "dimensional" cause
is not separable from the clustering one, which is the same non-identification we showed
observationally in Analysis 1, but now under an intervention the authors themselves propose.

So we co-log, every checkpoint and for every lambda:
    eff_dim     spectral entropy / effective rank of the pre-logit covariance (their quantity)
    nc1, nc_wb  neural collapse (label-based)
    sil_true    silhouette against TRUE class labels -- the metric that actually sees p classes
    twonn       the estimator our intervention showed is NOT confounded
and report t_grok against each.

Scope: "characterize what the intervention moves", never "their result is wrong".
"""
from __future__ import annotations
import os, sys, json, time, warnings
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

OUT = REPO / "RESULTS"


def spectral_entropy_t(H, eps=1e-9):
    """Differentiable spectral entropy of the activation covariance -- the GeomDR penalty."""
    Hc = H - H.mean(0, keepdim=True)
    C = (Hc.T @ Hc) / max(len(H) - 1, 1)
    C = C + eps * torch.eye(C.shape[0], dtype=C.dtype) * torch.diagonal(C).mean().clamp_min(eps)
    try:
        ev = torch.linalg.eigvalsh(C).clamp_min(eps)
    except Exception:
        ev = torch.linalg.svdvals(Hc).clamp_min(eps) ** 2
    p = ev / ev.sum()
    return -(p * torch.log(p)).sum()


def measure(H, y):
    Hn = H.detach().numpy()
    out = {"eff_rank": geom.effective_rank(Hn), "pca_pr": geom.pca_participation_ratio(Hn),
           "twonn": geom.twonn(Hn), "mle_k10": geom.mle_id(Hn, 10)}
    Xc = Hn - Hn.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    s = s[s > 0]
    p = s / s.sum()
    out["spec_entropy"] = float(-(p * np.log(p)).sum())
    out.update(geom.neural_collapse(Hn, y))
    try:
        out["sil_true"] = float(silhouette_score(Hn, y)) if len(np.unique(y)) > 1 else np.nan
    except Exception:
        out["sil_true"] = np.nan
    return out


def run(lam, seed, p=23, tf=0.7, epochs=30000, save_every=250, hidden=200, lr=1e-2, wd=0.03):
    torch.manual_seed(seed)
    pairs, y, n_sym, _ = build_task("add", p)
    N = len(y)
    X = torch.zeros(N, 2 * n_sym)
    X[torch.arange(N), torch.as_tensor(pairs[:, 0])] = 1.0
    X[torch.arange(N), n_sym + torch.as_tensor(pairs[:, 1])] = 1.0
    Y = torch.as_tensor(y, dtype=torch.long)
    g = torch.Generator().manual_seed(777)          # split fixed across lambda and seed
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
        loss.backward(); opt.step()
        if ep % save_every == 0:
            model.eval()
            with torch.no_grad():
                otr = model(Xtr, keep_activations=True)
                H = model.activations[-2]
                ote = model(Xte)
                r = {"epoch": ep, "lam": lam, "seed": seed,
                     "train_acc": float((otr.argmax(1) == Ytr).float().mean()),
                     "test_acc": float((ote.argmax(1) == Yte).float().mean())}
                r.update(measure(H, Ytr.numpy()))
                rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    lams = [0.0, 0.001, 0.003, 0.01]
    seeds = [0, 1, 2]
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
            alld.append(d)
            print(f"  lam={lam:<6} seed={sd}  final_test={fin:.3f}  t_grok={tg}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            pd.concat(alld, ignore_index=True).to_csv(OUT / "geomdr.csv", index=False)
    D = pd.concat(alld, ignore_index=True)
    D.to_csv(OUT / "geomdr.csv", index=False)
    print(f"\nwrote RESULTS/geomdr.csv {D.shape}")
