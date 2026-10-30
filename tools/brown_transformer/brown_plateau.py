"""
Matched plateau-to-convergence decomposition for the Brown-style Transformer cell.

WHY. Brown et al.'s qualitative phenomenon is a TwoNN decline from the memorisation
plateau to convergence. The v6.2 attribution decomposed initialisation to convergence, an
interval over which TwoNN *increases* in this cell, so it attributed a different quantity
from the one Brown report. This script decomposes the matched interval.

PLATEAU RULE (fixed before any decomposition output was inspected; see
config_selection.md). For each grokked confirmation seed, the plateau checkpoint is the
last saved checkpoint strictly before t_grok at which validation accuracy is at most 0.10
and training accuracy is at least 0.99. The rule uses only training and validation
accuracy: no representation statistic enters it. The convergence checkpoint is the final
saved checkpoint, as elsewhere in the paper.

OUTPUT. RESULTS/brown_transformer_plateau.csv holds, per seed and estimator, the raw 2x2
recombination values over [plateau, convergence], the raw plateau-to-convergence change,
the centroid contribution with centroids advanced first and second, the symmetric Shapley
contribution and share, and L2 at both real endpoints.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "brown_transformer"))

import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes

import brown_analyze as ba
from brown_train import ModDivisionDataset, P

OUT = REPO / "RESULTS"
ESTS = ["ph_dim", "twonn", "eff_rank", "mle_k10"]
VAL_MAX, TRAIN_MIN = 0.10, 0.99


def plateau_step(d, tg):
    """Last saved checkpoint before t_grok with val <= 0.10 and train >= 0.99."""
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    saved = {s for s in meta["ckpt_steps"]
             if (d / "ckpt" / f"step{s:06d}.pt").exists()}
    c = pd.read_csv(d / "curves.csv")
    ok = c[(c.val_acc <= VAL_MAX) & (c.train_acc >= TRAIN_MIN) & (c.step.isin(saved))]
    if tg > 0:
        ok = ok[ok.step < tg]
    return int(ok.step.max()) if len(ok) else -1


def main():
    torch.set_num_threads(6)
    M = pd.read_csv(OUT / "brown_transformer_run_manifest.csv")
    M = M[M.tag == "confirm"]
    rows = []
    for _, r in M.iterrows():
        d = REPO / "runs_brown" / r["dir"]
        tg = int(r["t_grok_step"])
        pl = plateau_step(d, tg)
        if pl < 0:
            print(f"[plateau] seed {int(r['seed'])}: no plateau checkpoint", flush=True)
            continue
        meta = ba.load_meta(d)
        ds = ModDivisionDataset(P, meta["frac_train"], seed=meta["seed"])
        Xva, Yva = ds.tensors(ds.val_pairs)
        y = Yva.numpy()
        final = max(s for s in meta["ckpt_steps"]
                    if (d / "ckpt" / f"step{s:06d}.pt").exists())
        H = {s: ba.activations(d, s, ds, Xva) for s in (pl, final)}
        L2 = {s: ba.leakage(ba.norm_trace(H[s]), y, 2) for s in (pl, final)}

        for frame in ("unaligned", "aligned"):
            use = dict(H)
            if frame == "aligned":
                ref = H[final] - H[final].mean(0)
                for s in (pl, final):
                    Xs = H[s] - H[s].mean(0)
                    R, _ = orthogonal_procrustes(Xs, ref)
                    use[s] = Xs @ R
            parts = {s: ba.parts_of(use[s], y) for s in (pl, final)}
            vals = {}
            for si, s in ((0, pl), (1, final)):
                for ti, t in ((0, pl), (1, final)):
                    Z = ba.norm_trace(parts[s][0] + parts[t][1])
                    vals[(si, ti)] = ba.measure(Z, y)
            for est in ESTS:
                f00 = vals[(0, 0)][est]      # plateau centroids, plateau residuals
                f10 = vals[(1, 0)][est]      # converged centroids, plateau residuals
                f01 = vals[(0, 1)][est]      # plateau centroids, converged residuals
                f11 = vals[(1, 1)][est]      # converged, converged
                diag = f00 - f11             # positive = the statistic falls
                first = f00 - f10            # centroids advanced first (raw)
                second = f01 - f11           # centroids advanced second (raw)
                phi_b = 0.5 * (first + second)
                rows.append({
                    "seed": int(r["seed"]), "grokked": bool(r["grokked"]),
                    "frame": frame, "estimator": est,
                    "step_plateau": pl, "step_final": final, "t_grok_step": tg,
                    "L2_plateau": L2[pl], "L2_final": L2[final],
                    "f00": f00, "f10": f10, "f01": f01, "f11": f11,
                    "raw_change": diag,
                    "rel_change": abs(diag) / max(abs(f00), 1e-12),
                    "centroid_first_raw": first, "centroid_second_raw": second,
                    "phi_between_raw": phi_b, "phi_within_raw": diag - phi_b,
                    "centroid_first_share": first / diag if abs(diag) > 1e-9 else np.nan,
                    "centroid_second_share": second / diag if abs(diag) > 1e-9 else np.nan,
                    "shapley_between_share": phi_b / diag if abs(diag) > 1e-9 else np.nan,
                })
        print(f"[plateau] seed {int(r['seed'])}: plateau {pl}, final {final} done",
              flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "brown_transformer_plateau.csv", index=False)

    print("\n=== plateau -> convergence, grokked seeds, unaligned ===")
    G = D[(D.frame == "unaligned") & D.grokked]
    for est in ESTS:
        E = G[G.estimator == est]
        appr = E[E.rel_change > 0.05]
        print(f"\n{est}: n={len(E)}, appreciable={len(appr)}")
        print(f"  raw change   median {E.raw_change.median():+.3f}  "
              f"range [{E.raw_change.min():+.3f}, {E.raw_change.max():+.3f}]")
        print(f"  centroid-first raw  median {E.centroid_first_raw.median():+.3f}  "
              f"range [{E.centroid_first_raw.min():+.3f}, {E.centroid_first_raw.max():+.3f}]")
        print(f"  centroid-second raw median {E.centroid_second_raw.median():+.3f}  "
              f"range [{E.centroid_second_raw.min():+.3f}, {E.centroid_second_raw.max():+.3f}]")
        if len(appr):
            print(f"  shares: first {appr.centroid_first_share.median():+.0%} "
                  f"[{appr.centroid_first_share.min():+.0%}, {appr.centroid_first_share.max():+.0%}]"
                  f"  second {appr.centroid_second_share.median():+.0%} "
                  f"[{appr.centroid_second_share.min():+.0%}, {appr.centroid_second_share.max():+.0%}]"
                  f"  shapley {appr.shapley_between_share.median():+.0%} "
                  f"[{appr.shapley_between_share.min():+.0%}, {appr.shapley_between_share.max():+.0%}]")
    print(f"\nL2: plateau median {G.L2_plateau.median():.3f}, "
          f"final median {G.L2_final.median():.3f}")
    print("wrote brown_transformer_plateau.csv")


if __name__ == "__main__":
    main()
