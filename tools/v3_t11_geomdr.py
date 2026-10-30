"""
T1.1 (regulariser leg) -- does the GeomDR dimensionality regulariser move WITHIN-CLASS spectral
compression, or only the global spectrum?

e9 established the acceleration replicates (up to 2.75x) but logged only global spectral
statistics. If within-class effective rank is the geometric quantity that tracks grokking
(the T1.1 hypothesis), the regulariser -- which penalises the GLOBAL spectral entropy -- should
be tested for whether it also compresses the within-class spectrum, and whether the acceleration
it produces is accompanied by earlier within-class compression.

Training matches tools/e9_geomdr_seeds.py exactly (p=23 add, tf=0.7, split seed 777, hidden 200,
AdamW, stablemax, penalty lam * spectral_entropy(H)); the mediator set is the light spectral one
plus the class-conditional decomposition. lam in {0, 0.003, 0.01}, seeds 0-2.

Output: RESULTS/v3_geomdr_within.csv
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from sklearn.metrics import silhouette_score
from tools.train_topology import build_task
from tools.geomdr import spectral_entropy_t
from models import MLP
from utils import stablemax_cross_entropy

OUT = REPO / "RESULTS"


def spec_stats(M):
    Z = M - M.mean(0)
    s = np.linalg.svd(Z, compute_uv=False) ** 2
    s = s[s > 1e-30]
    if not len(s):
        return np.nan, np.nan
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum())), float(s.sum() ** 2 / (s ** 2).sum())


def measure(Hn, y):
    cen = np.zeros_like(Hn)
    for c in np.unique(y):
        cen[y == c] = Hn[y == c].mean(0)
    res = Hn - cen
    w_er, w_pr = spec_stats(res)
    g_er, g_pr = spec_stats(Hn)
    c_er, _ = spec_stats(np.stack([Hn[y == c].mean(0) for c in np.unique(y)]))
    try:
        sil = float(silhouette_score(Hn, y))
    except Exception:
        sil = np.nan
    return {"within_eff_rank": w_er, "within_pca_pr": w_pr, "global_eff_rank": g_er,
            "global_pca_pr": g_pr, "centroid_eff_rank": c_er, "sil_true": sil}


def run(lam, seed, p=23, tf=0.7, epochs=30000, save_every=250, hidden=200, lr=1e-2, wd=0.03):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
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
    model = MLP(input_size=2 * n_sym, hidden_sizes=[hidden, hidden], output_size=n_sym,
                bias=False)
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
                H = model.activations[-2].numpy()
                ote = model(Xte)
                r = {"epoch": ep, "lam": lam, "seed": seed,
                     "train_acc": float((otr.argmax(1) == Ytr).float().mean()),
                     "test_acc": float((ote.argmax(1) == Yte).float().mean())}
                r.update(measure(H, Ytr.numpy()))
                rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    alld, t0 = [], time.time()
    for lam in (0.0, 0.003, 0.01):
        for sd in (0, 1, 2):
            d = run(lam, sd)
            fin = d.test_acc.values[-3:].mean()
            tg = d[d.test_acc >= 0.5 * fin].epoch.min() if fin >= 0.5 else np.nan
            d["t_grok"] = tg
            alld.append(d)
            print(f"  lam={lam:<6} seed={sd}  final_test={fin:.3f}  t_grok={tg}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            pd.concat(alld, ignore_index=True).to_csv(OUT / "v3_geomdr_within.csv", index=False)
    print(f"wrote RESULTS/v3_geomdr_within.csv")
