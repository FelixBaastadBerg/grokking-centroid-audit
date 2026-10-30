"""
Requirement 4 -- co-measure a clustering / neural-collapse metric alongside EVERY dimension and
topology signal, at the SAME uniformly-spaced checkpoints. This is the backbone evidence, and it
also settles a threat to our own headline that surfaced en route:

  gap_ratio (and nn1_ratio) are built from nearest-neighbour distances on the same cloud as the
  dimension estimators, and correlate with them at rho 0.87-0.98. Partialling them out may be
  CIRCULAR -- regressing a quantity on a near-copy of itself. NC1 is structurally different: it
  uses class labels and second moments, not neighbour distances. If the dimension/generalization
  link collapses under gap_ratio but survives under NC1, the honest claim is much weaker than
  "dimension is a clustering artifact".

So every checkpoint records, on the SAME model state:
  activations (p^2*tf points, labels available): ph_dim, corr_dim, twonn, MLE k5/k10/k20,
      pca_pr, eff_rank  +  NC1, NC2-ETF, nc_wb (true neural collapse)  +  gap_ratio, silhouette
  token embeddings (p points, Monod's object):  H1_max, H1_total, H1_nbars  +  its own clustering

Uniform stride, because the timing statistic is a cross-correlation of first differences.
"""
from __future__ import annotations
import os, sys, warnings, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from analysis import geom
from analysis.lsrun import LandscapeRun
from tools.linchpin import h1_stats, clustering_metrics

OUT = REPO / "RESULTS"
MAX_PTS = 500
N_CK = 60


def run_one(d: Path):
    r = LandscapeRun(d)
    n = len(r.epochs)
    stride = max(1, n // N_CK)
    ts = np.arange(0, n, stride)
    ytr = r.Ytr.numpy()
    sub = (np.random.default_rng(0).choice(len(ytr), MAX_PTS, replace=False)
           if len(ytr) > MAX_PTS else np.arange(len(ytr)))
    sub = np.sort(sub); ysub = ytr[sub]
    p = r.meta["p"]
    rows = []
    for t in ts:
        r._load(r.flat(int(t))); r.model.eval()
        with torch.no_grad():
            r.model(r.Xtr, keep_activations=True)
            H = r.model.activations[-2].numpy()[sub]
        b = geom.battery(H, y=ysub, cheap=True)          # dimension + NC on ACTIVATIONS
        b.update({f"act_{k}": v for k, v in clustering_metrics(H).items()})
        E = r.W[0][int(t)][:, :p].T                      # token embeddings = Monod's object
        b.update({f"emb_{k}": v for k, v in h1_stats(E).items()})
        b.update({f"emb_{k}": v for k, v in clustering_metrics(E).items()})
        b.update({"t": int(t), "epoch": int(r.epochs[int(t)]),
                  "train_acc": float(r.curves.train_acc.values[int(t)]),
                  "test_acc": float(r.curves.test_acc.values[int(t)])})
        rows.append(b)
    df = pd.DataFrame(rows)
    for k, v in (("arch", r.meta["arch"]), ("task", r.meta["task"]), ("p", p),
                 ("tf", r.meta["train_fraction"]), ("seed", r.meta["seed"]),
                 ("final_test", float(r.curves.test_acc.iloc[-1])), ("dir", d.name)):
        df[k] = v
    return df


if __name__ == "__main__":
    inv = pd.read_csv(OUT / "_inventory.csv")
    sel = inv[(inv.arch == "mlp") & (inv.test >= 0.9)].copy()
    keep = []
    for (task, p, tf), g in sel.groupby(["task", "p", "tf"]):
        keep.extend(g.sort_values("seed").head(3).to_dict("records"))
    sel = pd.DataFrame(keep)
    print(f"=== co-measurement on {len(sel)} grokked MLP runs, uniform stride ===\n", flush=True)
    alld, t0 = [], time.time()
    for i, rec in enumerate(sel.to_dict("records")):
        try:
            alld.append(run_one(REPO / "runs_topo" / rec["dir"]))
        except Exception as e:
            print(f"  [{i+1}/{len(sel)}] SKIP {rec['dir'][:40]}: {type(e).__name__}", flush=True)
            continue
        print(f"  [{i+1}/{len(sel)}] {rec['task']}/p{rec['p']}/tf{rec['tf']}/s{rec['seed']} "
              f"({len(alld[-1])} ckpts, {time.time()-t0:.0f}s)", flush=True)
        pd.concat(alld, ignore_index=True).to_csv(OUT / "comeasure.csv", index=False)
    D = pd.concat(alld, ignore_index=True)
    D.to_csv(OUT / "comeasure.csv", index=False)
    print(f"\nwrote RESULTS/comeasure.csv {D.shape}")
