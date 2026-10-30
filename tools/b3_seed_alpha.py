"""
B3 -- seed-replicate the real centroid intervention on the canonical cell.

E1 is a BREADTH design: one run per task-architecture cell, 7 runs, no within-cell replication and
therefore no real confirmation half. That is the single biggest empirical objection to the paper.
This fixes it without new training, because the checkpoints of the Z_23 addition cell are retained.

SELECTION, stated carefully. The inventory lists 24 grokked runs in that cell at train fraction 0.8,
but they are NOT 24 independent seeds: most are batch-size variants (bs-4 ... bs-512) of a handful
of seeds, and some seeds appear more than once under different checkpoint-logging schedules. Pooling
them would inflate n with pseudo-replicates and mix hyperparameters. We therefore take exactly one
run per DISTINCT SEED at the canonical full-batch configuration, which gives 9 genuinely independent
seeds -- and we report it as 9, not 24.

Same decomposition as tools/e1_centroid.py,

    x_i  =  mu  +  alpha (mu_{y_i} - mu)  +  beta (x_i - mu_{y_i}),

swept in alpha at beta = 1 on a DENSER grid than E1, because tools/b2_attribution.py has to
interpolate this curve to the silhouette real networks actually occupy and a 7-point grid is too
coarse for that.

Analysis-only: no training is run here.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from analysis.lsrun import LandscapeRun
from tools.e1_centroid import decompose, recompose, measure

OUT = REPO / "RESULTS"
# denser than E1's 7 points: B2 interpolates this curve to a target silhouette
ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
CELL = dict(arch="mlp", task="add", p=23, tf=0.8)
N_SUB = 400          # identical point budget to E1, so the numbers are comparable


def select(inv):
    """One run per distinct seed, canonical full-batch config, no batch-size variants."""
    s = inv[(inv.arch == CELL["arch"]) & (inv.task == CELL["task"]) & (inv.p == CELL["p"])
            & (inv.tf == CELL["tf"]) & (inv.test >= 0.9)].copy()
    s = s[s.bs == 0]                                   # full batch only
    s["suffix"] = s.dir.str.replace(r"^.*tf-0\.8", "", regex=True)
    rank = {"": 0, "__ss777": 1, "__hk": 2, "__ht5full": 3}
    s["rank"] = s.suffix.map(lambda x: rank.get(x, 9))
    s = s.sort_values(["seed", "rank"]).drop_duplicates("seed", keep="first")
    return s.sort_values("seed")


def main():
    inv = pd.read_csv(OUT / "_inventory.csv")
    sel = select(inv)
    print(f"=== B3: centroid intervention on {len(sel)} DISTINCT SEEDS of the "
          f"{CELL['task']} p={CELL['p']} {CELL['arch']} tf={CELL['tf']} cell ===")
    print(f"    seeds: {sorted(sel.seed.tolist())}")
    print(f"    alpha grid: {ALPHAS}\n", flush=True)

    rows, t0 = [], time.time()
    for i, (_, rec) in enumerate(sel.iterrows()):
        try:
            r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
            r._load(r.flat(len(r.epochs) - 1)); r.model.eval()
            with torch.no_grad():
                r.model(r.Xtr, keep_activations=True)
                H = r.model.activations[-2].numpy()
            y = r.Ytr.numpy()
        except Exception as e:
            print(f"  skip {rec['dir'][:50]}: {type(e).__name__} {e}", flush=True); continue

        sub = np.sort(np.random.default_rng(0).choice(len(y), min(N_SUB, len(y)), replace=False))
        H, y = H[sub], y[sub]
        mu, bet, wit = decompose(H, y)
        ppc = float(len(y) / len(np.unique(y)))
        for a in ALPHAS:
            Z = recompose(mu, bet, wit, a, 1.0, norm="trace")
            rows.append({"dir": rec["dir"], "seed": int(rec["seed"]), "alpha": a,
                         "points_per_class": ppc, "n_points": len(y),
                         "test": float(rec["test"]), **measure(Z, y)})
        pd.DataFrame(rows).to_csv(OUT / "b3_seed_alpha.csv", index=False)
        print(f"  [{i+1}/{len(sel)}] seed {int(rec['seed']):>2}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "b3_seed_alpha.csv", index=False)
    print(f"\nwrote RESULTS/b3_seed_alpha.csv {D.shape}, {D.dir.nunique()} runs")


if __name__ == "__main__":
    main()
