"""
v6.8 -- measure L_2 in the HYBRID clouds of the Transformer plateau-to-convergence leg.

v6.7's Appendix D.5 invokes the containment lemma for the centroids-second leg using the
converged endpoint alone: "that is the invariance of Appendix E.1 applied at the converged
endpoint, where L_2 = 0". The lemma needs containment at BOTH endpoints of the centroid move.
The centroids-second move holds residuals at convergence and advances the centroids, so its two
clouds are

    f(C_plateau, R_conv)   and   f(C_conv, R_conv) ,

and only the second is a real recorded state. RESULTS/brown_transformer_plateau.csv stores L_2
on the two REAL clouds (plateau and converged), not on the hybrid, so the lemma could not be
checked from the stored files. This script measures it.

The clouds are rebuilt exactly as tools/brown_transformer/brown_plateau.py builds them: the same
plateau rule, the same checkpoints, the same centroid/residual split and the same trace
normalisation, so the L_2 values line up with the shares already reported.

Output: RESULTS_v6_8/brown_hybrid_leakage.csv, one row per confirmation seed.

Analysis-only: retained checkpoints, no training.
"""
from __future__ import annotations
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "brown_transformer"))

import numpy as np
import pandas as pd
import torch

import brown_analyze as ba
from brown_plateau import plateau_step
from brown_train import ModDivisionDataset, P

OUT = REPO / "RESULTS_v6_8"
OUT.mkdir(exist_ok=True)


def main():
    torch.set_num_threads(6)
    M = pd.read_csv(REPO / "RESULTS" / "brown_transformer_run_manifest.csv")
    M = M[M.tag == "confirm"]
    rows = []
    for _, r in M.iterrows():
        d = REPO / "runs_brown" / r["dir"]
        tg = int(r["t_grok_step"])
        pl = plateau_step(d, tg)
        if pl < 0:
            print(f"  seed {int(r['seed'])}: no plateau checkpoint, skipped", flush=True)
            continue
        meta = ba.load_meta(d)
        ds = ModDivisionDataset(P, meta["frac_train"], seed=meta["seed"])
        Xva, Yva = ds.tensors(ds.val_pairs)
        y = Yva.numpy()
        final = max(s for s in meta["ckpt_steps"]
                    if (d / "ckpt" / f"step{s:06d}.pt").exists())
        H = {s: ba.activations(d, s, ds, Xva) for s in (pl, final)}
        parts = {s: ba.parts_of(H[s], y) for s in (pl, final)}

        # the four cells, indexed exactly as brown_plateau.py indexes them
        def cloud(si, ti):
            s = pl if si == 0 else final
            t = pl if ti == 0 else final
            return ba.norm_trace(parts[s][0] + parts[t][1])

        L = {(si, ti): ba.leakage(cloud(si, ti), y, 2)
             for si in (0, 1) for ti in (0, 1)}
        rows.append({
            "seed": int(r["seed"]), "grokked": bool(r["grokked"]),
            "step_plateau": pl, "step_final": final,
            # centroids-second leg: residuals held at convergence, centroids advanced
            "L2_Cplateau_Rconv": L[(0, 1)],
            "L2_Cconv_Rconv": L[(1, 1)],
            # centroids-first leg: residuals held at the plateau
            "L2_Cplateau_Rplateau": L[(0, 0)],
            "L2_Cconv_Rplateau": L[(1, 0)],
        })
        print(f"  seed {int(r['seed'])}: L2 f(Cpl,Rconv) = {L[(0, 1)]:.4f}, "
              f"f(Cconv,Rconv) = {L[(1, 1)]:.4f}", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "brown_hybrid_leakage.csv", index=False)

    G = D[D.grokked]
    both_zero = int(((G.L2_Cplateau_Rconv == 0) & (G.L2_Cconv_Rconv == 0)).sum())
    print(f"\n=== centroids-second leg, {len(G)} grokked seeds ===")
    print(f"  L2 at f(C_plateau, R_conv): median {G.L2_Cplateau_Rconv.median():.4f}  "
          f"[{G.L2_Cplateau_Rconv.min():.4f}, {G.L2_Cplateau_Rconv.max():.4f}]")
    print(f"  L2 at f(C_conv,    R_conv): median {G.L2_Cconv_Rconv.median():.4f}  "
          f"[{G.L2_Cconv_Rconv.min():.4f}, {G.L2_Cconv_Rconv.max():.4f}]")
    print(f"  seeds class-contained at BOTH clouds of the move: {both_zero} / {len(G)}")
    print("  -> the containment lemma "
          + ("applies to this leg." if both_zero == len(G)
             else "does NOT license this leg; the small centroid contribution is empirical."))
    print("\nwrote RESULTS_v6_8/brown_hybrid_leakage.csv")


if __name__ == "__main__":
    main()
