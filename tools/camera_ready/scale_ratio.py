"""
Separation scale at the endpoints (Appendix D.2, "Separation scale at the endpoints").

For each primary Z_23 seed, split the pre-logit cloud at initialisation and at convergence into
its centroid part mu_{y_i} - mu and its residual part x_i - mu_{y_i}, and report the ratio a/b of
their root-mean-square norms. Trace normalisation removes one global scale per cloud but not this
ratio. As a reproduction check, the 2x2 endpoint recombination cells are rebuilt from the same
activations and compared with RESULTS_v6_4/fullgrid_recombine.csv; only seeds whose PH-dimension
cells reproduce exactly (max abs deviation < 1e-5) enter the reported medians.

This analysis was added post hoc for the camera-ready version. It reads endpoint activations from
an .npz file with keys

    y_<seed>              class labels of the 529 complete-grid inputs, fixed order
    H_<seed>_init         (529, 200) pre-logit activations at the initial checkpoint
    H_<seed>_converged    (529, 200) pre-logit activations at the final checkpoint

for seeds 0, 1, 2, 3, 4, 10, 11, 12, 13. That activation file is NOT distributed with this
repository; it can be regenerated from the training runs (runs_topo/) with the activation code in
tools/v6_4/fullgrid.py (full_grid_inputs, acts). The per-seed output used in the paper is
RESULTS_camera_ready/scale_ratio.csv.

Run:  python tools/camera_ready/scale_ratio.py --acts path/to/z23_endpoint_acts.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analysis import geom  # noqa: E402

SEEDS = [0, 1, 2, 3, 4, 10, 11, 12, 13]


def parts(H, y):
    M = {c: H[y == c].mean(0) for c in np.unique(y)}
    cen = np.stack([M[c] for c in y])
    return cen, H - cen


def rms(A):
    return float(np.sqrt((A ** 2).sum(1).mean()))


def trace_norm(Z):
    Zc = Z - Z.mean(0)
    return Zc / (np.sqrt((Zc ** 2).sum(1).mean()) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True, help="endpoint activations .npz (see docstring)")
    ap.add_argument("--out", default=str(REPO / "RESULTS_camera_ready" / "scale_ratio.csv"))
    a = ap.parse_args()

    z = np.load(a.acts, allow_pickle=True)
    ref = pd.read_csv(REPO / "RESULTS_v6_4" / "fullgrid_recombine.csv")
    ref = ref[ref.pipeline == "raw"]
    rows = []
    for s in SEEDS:
        y = np.asarray(z[f"y_{s}"])
        H = {0: np.asarray(z[f"H_{s}_init"], float), 1: np.asarray(z[f"H_{s}_converged"], float)}
        C, Rr, ratio = {}, {}, {}
        for k in (0, 1):
            C[k], Rr[k] = parts(H[k], y)
            ratio[k] = rms(C[k] - C[k].mean(0)) / rms(Rr[k])
        r = ref[ref.seed == s]
        smax = r.s.max()
        dph, dtw = 0.0, 0.0
        for c in (0, 1):
            for q in (0, 1):
                Z = trace_norm(C[c] + Rr[q])
                ph = geom.ph_dim(Z)[0]
                tw = geom.twonn(Z)
                rr = r[(r.s == c * smax) & (r.t == q * smax)].iloc[0]
                dph = max(dph, abs(ph - rr.ph_dim))
                dtw = max(dtw, abs(tw - rr.twonn))
        rows.append(dict(seed=s, ratio_init=ratio[0], ratio_conv=ratio[1],
                         max_abs_dev_ph_dim_vs_fullgrid=dph,
                         max_abs_dev_twonn_vs_fullgrid=dtw,
                         exact_reproduction=dph < 1e-5))
        print(f"seed {s}: a/b init {ratio[0]:.3f}  conv {ratio[1]:.3f}  "
              f"max |dPH| {dph:.2e}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(a.out, index=False)
    E = D[D.exact_reproduction]
    print(f"\n{len(E)} exactly reproducing seeds: a/b init {E.ratio_init.median():.3f} "
          f"[{E.ratio_init.min():.3f}, {E.ratio_init.max():.3f}], conv "
          f"{E.ratio_conv.median():.2f} [{E.ratio_conv.min():.2f}, {E.ratio_conv.max():.2f}]")


if __name__ == "__main__":
    main()
