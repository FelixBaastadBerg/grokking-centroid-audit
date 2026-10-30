"""
L_2 along the centroid-intervention sweep on the converged Brown-style clouds: the exact
analogue of Appendix E.2 for the Transformer cell. Adds an `L2` column to
RESULTS/brown_transformer_intervention.csv so the proposition's precondition can be read
next to TwoNN's response along the same sweep.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "brown_transformer"))

import numpy as np
import pandas as pd
import torch

import brown_analyze as ba
from brown_train import ModDivisionDataset, P

OUT = REPO / "RESULTS"


def main():
    torch.set_num_threads(6)
    I = pd.read_csv(OUT / "brown_transformer_intervention.csv")
    rows = []
    for d in ba.run_dirs("confirm"):
        meta = ba.load_meta(d)
        ds = ModDivisionDataset(P, meta["frac_train"], seed=meta["seed"])
        Xva, Yva = ds.tensors(ds.val_pairs)
        y = Yva.numpy()
        step = int(I[I.seed == meta["seed"]].ckpt_step.iloc[0])
        H = ba.activations(d, step, ds, Xva)
        mu = H.mean(0)
        cen, res = ba.parts_of(H, y)
        for a in sorted(I.alpha.unique()):
            Z = ba.norm_trace(mu + a * (cen - mu) + res)
            rows.append({"seed": meta["seed"], "alpha": a, "L2": ba.leakage(Z, y, 2)})
        print(f"[sweep-L2] seed {meta['seed']} done", flush=True)
    L = pd.DataFrame(rows)
    M = I.merge(L, on=["seed", "alpha"], how="left")
    M.to_csv(OUT / "brown_transformer_intervention.csv", index=False)
    print(M.groupby("alpha")[["L2", "twonn", "ph_dim"]].median().round(4).to_string())


if __name__ == "__main__":
    main()
