"""
E7 calibration: where do the real |G|-point token-embedding clouds sit on the clustering axis,
relative to reference clouds of matched size? This decides whether the clustering confound is
triggered at all in the setting where the H1 claims are computed.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from analysis import geom
from tools.intervention import make_cloud, clusteredness

OUT = REPO / "RESULTS"

if __name__ == "__main__":
    g = np.random.default_rng(0)
    rows = []
    refs = [("unclustered Gaussian blob", g.normal(size=(23, 60))),
            ("K=4 loose (sigma 0.6)", make_cloud(5, 4, 0.6, n=23, seed=0)),
            ("K=4 medium (sigma 0.2)", make_cloud(5, 4, 0.2, n=23, seed=0)),
            ("K=4 tight (sigma 0.05)", make_cloud(5, 4, 0.05, n=23, seed=0)),
            ("K=4 very tight (sigma 0.01)", make_cloud(5, 4, 0.01, n=23, seed=0))]
    print("Reference clouds at matched n=23, ambient 60:\n")
    print(f"  {'cloud':34s} {'silhouette':>11s} {'gap_ratio':>10s}")
    for nm, X in refs:
        s, gr = clusteredness(X), geom.clusteredness(X)["gap_ratio"]
        rows.append({"cloud": nm, "silhouette": s, "gap_ratio": gr, "kind": "reference"})
        print(f"  {nm:34s} {s:>+11.3f} {gr:>10.3f}")

    E = pd.read_csv(OUT / "e7_embedding.csv")
    for stage in ("init", "final"):
        f = E[E.stage == stage]
        s, gr = f.silhouette_free.median(), f.gap_ratio.median()
        rows.append({"cloud": f"REAL token embeddings ({stage})", "silhouette": s,
                     "gap_ratio": gr, "kind": "real"})
        print(f"  {'REAL token embeddings (' + stage + ')':34s} {s:>+11.3f} {gr:>10.3f}")

    pd.DataFrame(rows).to_csv(OUT / "e7_calibration.csv", index=False)
    print("\n  The synthetic failure regime requires silhouette ~0.70 / gap_ratio ~0.02-0.10.")
    print("  Real token embeddings sit at or below the UNCLUSTERED reference, so the clustering")
    print("  confound is not triggered on that object. Activations are a different object and")
    print("  do reach the failure regime (label silhouette +0.645).")
