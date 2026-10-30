"""
Diagnostic for P0-1: what is the NATURAL between/within scatter ratio during training?

The first pilot set L_clust's target to 6.0, which turned out to sit below the natural ratio, so
the loss pushed classes together and silhouette FELL (0.113 -> -0.037). The target has to be set
above the trajectory for the lever to increase clustering.
"""
from __future__ import annotations
import os, sys, warnings
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
from tools.p0_1_losses import within_between

OUT = REPO / "RESULTS"

if __name__ == "__main__":
    rows = []
    for task, p, sym in (("add", 23, 5), ("perm", 24, 4)):
        torch.manual_seed(0)
        pairs, y, n_sym, _ = build_task(task, p, sym_n=sym)
        N = len(y)
        X = torch.zeros(N, 2 * n_sym)
        X[torch.arange(N), torch.as_tensor(pairs[:, 0])] = 1.0
        X[torch.arange(N), n_sym + torch.as_tensor(pairs[:, 1])] = 1.0
        Y = torch.as_tensor(y, dtype=torch.long)
        g = torch.Generator().manual_seed(777)
        perm = torch.randperm(N, generator=g)
        ntr = int(0.7 * N)
        Xtr, Ytr = X[perm[:ntr]], Y[perm[:ntr]]
        m = MLP(input_size=2 * n_sym, hidden_sizes=[200, 200], output_size=n_sym, bias=False)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-25,
                                weight_decay=0.03)
        print(f"=== {task} p={p}: natural between/within scatter ratio ===")
        for ep in range(20001):
            m.train(); opt.zero_grad()
            lo = m(Xtr, keep_activations=True)
            stablemax_cross_entropy(lo, Ytr, dtype=torch.float64).backward(); opt.step()
            if ep in (0, 250, 500, 1000, 2000, 5000, 10000, 20000):
                m.eval()
                with torch.no_grad():
                    m(Xtr, keep_activations=True)
                    H = m.activations[-2]
                    W, B, _ = within_between(H, Ytr, n_sym)
                    sep = (B.pow(2).sum() / len(H)).item()
                    sc = (W.pow(2).sum() / len(H)).item()
                    Hn = H.numpy()
                    sil = silhouette_score(Hn, Ytr.numpy())
                    te = 0.0
                print(f"  epoch {ep:6d}  sep/within = {sep/sc:8.3f}   silhouette = {sil:+.3f}")
                rows.append({"task": task, "p": p, "epoch": ep, "ratio": sep / sc,
                             "silhouette": sil})
        print()
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "p0_1_natural_ratio.csv", index=False)
    print("target for L_clust must sit ABOVE the run's own trajectory; suggested targets:")
    for task, g in D.groupby("task"):
        mx = g.ratio.max()
        print(f"  {task}: max natural ratio {mx:.2f} -> use targets "
              f"{', '.join(f'{mx*k:.1f}' for k in (2, 4, 8))}")
