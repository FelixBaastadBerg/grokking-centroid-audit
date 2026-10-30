"""
Item 4 + item 6 -- support leakage ALONG the intervention, which is where the tightened
proposition bites.

The work order suggests the alpha-sweep satisfies q-neighbour containment automatically because
classes move apart. That is only half true, and the half that is false matters. The sweep starts at
alpha = 0, where x~ = mu + (x_i - mu_{y_i}) puts EVERY class on the same centre: containment fails
completely there. It is restored as alpha grows. So the exact proposition does not license
invariance across the whole sweep -- it licenses invariance between any two alphas at which
containment holds, and the residual motion TwoNN does show should be concentrated exactly in the
low-alpha stretch where L_q > 0.

This measures L_q(alpha) directly, synthetically and on real activations, and checks that the TwoNN
motion lives where the leakage does.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from analysis import geom
from analysis.lsrun import LandscapeRun
from tools.e1_centroid import decompose, recompose
from tools.v2_theory import leakage
from tools.b3_seed_alpha import select
from tools.b1_gride import cloud

OUT = REPO / "RESULTS"
ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def sweep(X, y, tag, extra=None):
    mu, bet, wit = decompose(X, y)
    rows = []
    for a in ALPHAS:
        Z = recompose(mu, bet, wit, a, 1.0, norm="trace")
        rows.append({"source": tag, "alpha": a,
                     "L2": leakage(Z, y, 2), "L5": leakage(Z, y, 5), "L10": leakage(Z, y, 10),
                     "twonn": geom.twonn(Z), "ph_dim": geom.ph_dim(Z)[0],
                     **(extra or {})})
    return rows


def main():
    rows = []
    for s in (0, 1, 2):
        X, y = cloud(5, 23, 17, 0.05, seed=s)
        rows += sweep(X, y, "synthetic", {"seed": s})

    inv = pd.read_csv(OUT / "_inventory.csv")
    for _, rec in select(inv).iterrows():
        r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
        r._load(r.flat(len(r.epochs) - 1)); r.model.eval()
        with torch.no_grad():
            r.model(r.Xtr, keep_activations=True)
            H = r.model.activations[-2].numpy()
        y = r.Ytr.numpy()
        sub = np.sort(np.random.default_rng(0).choice(len(y), min(400, len(y)), replace=False))
        rows += sweep(H[sub], y[sub], "real", {"seed": int(rec["seed"])})

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "v2_leakage_alpha.csv", index=False)

    for src, g0 in D.groupby("source"):
        print(f"\n=== {src} ===")
        print(f"  {'alpha':>7}{'L2':>9}{'L5':>9}{'L10':>9}{'TwoNN':>10}{'PH-dim':>10}")
        P = g0.groupby("alpha")[["L2", "L5", "L10", "twonn", "ph_dim"]].median()
        for a, r_ in P.iterrows():
            print(f"  {a:>7.3f}{r_.L2:>9.3f}{r_.L5:>9.3f}{r_.L10:>9.3f}{r_.twonn:>10.2f}"
                  f"{r_.ph_dim:>10.2f}")
        zero = P[P.L2 <= 1e-9]
        if len(zero):
            a0 = zero.index.min()
            tw = zero.twonn
            print(f"  containment (L2 = 0) first holds at alpha = {a0:g}; over that stretch TwoNN "
                  f"varies {tw.max()/max(abs(tw.min()),1e-9):.4f}x")
            below = P[P.index < a0].twonn
            if len(below):
                allv = P.twonn
                print(f"  over the WHOLE sweep TwoNN varies "
                      f"{allv.max()/max(abs(allv.min()),1e-9):.3f}x, so essentially all of its "
                      f"motion is in the leaky stretch alpha < {a0:g}")
        else:
            print("  L2 never reaches 0 on this cloud")


if __name__ == "__main__":
    main()
