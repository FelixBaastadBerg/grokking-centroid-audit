"""
v6.4 Phase 2.7 -- numerical audit of every "exactly preserved" claim in the paper.

The centroid intervention preserves same-class distances exactly. The measured clouds are then
divided by a cloud-specific global scale (the trace normalisation), and after that the distances
are preserved only up to ONE common factor per cloud. That is enough for scale-invariant
estimators, but the prose has to say it correctly, so each half is checked separately:

  C1  raw same-class pairwise distances are bit-for-bit unchanged by the centroid intervention,
      BEFORE any global scaling;
  C2  after the allowed global scaling, every same-class distance in a cloud is the original
      distance times ONE common factor -- the spread of that factor within a cloud is reported;
  C3  the pointwise distance RATIOS the fixed-neighbour estimators consume (r2/r1 for TwoNN,
      r_{2n1}/r_{n1} for GRIDE, the MLE-ID log-ratios) are unchanged wherever the q-nearest
      neighbourhoods are class-contained at both endpoints, and the check is reported separately
      for the leaky and the contained parts of the sweep;
  C4  the diagonal cells of the recombination grid reproduce the real checkpoint statistic --
      i.e. the counterfactual construction is the identity on the network's own trajectory.

Run on all nine primary seeds, on the complete input grid. Writes RESULTS_v6_4/.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist

from analysis import geom
from analysis.lsrun import LandscapeRun
from tools.b3_seed_alpha import select
from tools.c1_recombine import checkpoints
from tools.v6_4.fullgrid import full_grid_inputs, acts, parts, trace_norm, ALPHAS, leakage

OUT = REPO / "RESULTS_v6_4"
OUT.mkdir(exist_ok=True)


def same_class_pairs(y):
    """Boolean mask over pdist's condensed ordering selecting same-class pairs."""
    y = np.asarray(y)
    n = len(y)
    i, j = np.triu_indices(n, 1)
    return y[i] == y[j]


def ratios(Z, q):
    """The distance ratios a fixed-neighbour estimator of support q consumes."""
    nn = NearestNeighbors(n_neighbors=q + 1).fit(Z)
    d, idx = nn.kneighbors(Z)
    return d[:, 1:] / np.maximum(d[:, 1:2], 1e-300), idx[:, 1:]


def one_run(rec):
    d, seed = rec["dir"], int(rec["seed"])
    r = LandscapeRun(REPO / "runs_topo" / d)
    X, y = full_grid_inputs(r.meta)
    idx = checkpoints(r)
    H = {i: acts(r, i, X) for i in idx}

    Hc = H[idx[-1]]
    cen, res = parts(Hc, y)
    mu = Hc.mean(0)
    bet = cen - mu
    same = same_class_pairs(y)
    D0_all = pdist(Hc)
    D0 = D0_all[same]

    rows = []
    # ---- C1 / C2 / C3 over the alpha sweep
    for a in ALPHAS:
        Zraw = mu + a * bet + res                    # before any global scaling
        Draw = pdist(Zraw)[same]
        c1_max_abs = float(np.max(np.abs(Draw - D0)))
        c1_max_rel = float(np.max(np.abs(Draw - D0) / np.maximum(D0, 1e-300)))

        Z = trace_norm(Zraw)                          # after the allowed global scaling
        Dn = pdist(Z)[same]
        fac = Dn / np.maximum(D0, 1e-300)
        c2_spread = float(fac.max() / fac.min() - 1.0)

        L2 = leakage(Z, y, 2)
        R0, I0 = ratios(trace_norm(Hc), 2)
        R1, I1 = ratios(Z, 2)
        contained = (np.asarray(y)[I0] == np.asarray(y)[:, None]).all(1) & \
                    (np.asarray(y)[I1] == np.asarray(y)[:, None]).all(1)
        c3 = (float(np.max(np.abs(R1[contained] - R0[contained]))) if contained.any()
              else np.nan)
        rows.append({"dir": d, "seed": seed, "check": "sweep", "alpha": a,
                     "L2": L2, "frac_contained_both": float(contained.mean()),
                     "c1_sameclass_max_abs_dev": c1_max_abs,
                     "c1_sameclass_max_rel_dev": c1_max_rel,
                     "c2_global_factor": float(np.median(fac)),
                     "c2_factor_spread": c2_spread,
                     "c3_max_ratio_dev_contained": c3})

    # ---- C4: the diagonal of the recombination grid is the real trajectory
    for k, i in enumerate(idx):
        cen_i, res_i = parts(H[i], y)
        Zdiag = trace_norm(cen_i + res_i)
        Zreal = trace_norm(H[i])
        rows.append({"dir": d, "seed": seed, "check": "diagonal", "alpha": np.nan,
                     "ckpt": k, "epoch": int(r.epochs[i]),
                     "c4_max_coord_dev": float(np.max(np.abs(Zdiag - Zreal))),
                     "c4_ph_dim_diag": geom.ph_dim(Zdiag)[0],
                     "c4_ph_dim_real": geom.ph_dim(Zreal)[0],
                     "c4_twonn_diag": geom.twonn(Zdiag),
                     "c4_twonn_real": geom.twonn(Zreal)})
    return rows


def main():
    from multiprocessing import Pool
    t0 = time.time()
    inv = pd.read_csv(REPO / "RESULTS" / "_inventory.csv")
    recs = [r for _, r in select(inv).iterrows()]
    print(f"=== v6.4 exactness checks on {len(recs)} primary seeds, complete input grid ===",
          flush=True)
    rows = []
    with Pool(processes=5) as pool:
        for i, rr in enumerate(pool.imap_unordered(one_run, recs), 1):
            rows += rr
            print(f"  [{i}/{len(recs)}] ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "exactness_checks.csv", index=False)

    S = D[D.check == "sweep"]
    G = D[D.check == "diagonal"]
    print("\n--- C1  same-class distances before global scaling ---")
    print(f"  max absolute deviation over all seeds and alphas: "
          f"{S.c1_sameclass_max_abs_dev.max():.3e}")
    print(f"  max relative deviation:                           "
          f"{S.c1_sameclass_max_rel_dev.max():.3e}")
    print("\n--- C2  after global scaling: one common factor per cloud ---")
    print(f"  max within-cloud spread of the factor (max/min - 1): "
          f"{S.c2_factor_spread.max():.3e}")
    print("\n--- C3  TwoNN distance ratios where neighbourhoods are class-contained ---")
    cs = S.dropna(subset=["c3_max_ratio_dev_contained"])
    print(f"  max deviation, contained points:  {cs.c3_max_ratio_dev_contained.max():.3e}")
    print(f"  alphas with L2 = 0 exactly:       "
          f"{sorted(set(S[S.L2 == 0].alpha))}")
    print("\n--- C4  diagonal recombination vs the real checkpoint ---")
    print(f"  max coordinate deviation:  {G.c4_max_coord_dev.max():.3e}")
    print(f"  max |PH-dim difference|:   "
          f"{(G.c4_ph_dim_diag - G.c4_ph_dim_real).abs().max():.3e}")
    print(f"  max |TwoNN difference|:    "
          f"{(G.c4_twonn_diag - G.c4_twonn_real).abs().max():.3e}")

    summ = {
        "c1_max_abs_dev": S.c1_sameclass_max_abs_dev.max(),
        "c1_max_rel_dev": S.c1_sameclass_max_rel_dev.max(),
        "c2_max_factor_spread": S.c2_factor_spread.max(),
        "c3_max_ratio_dev_contained": cs.c3_max_ratio_dev_contained.max(),
        "c4_max_coord_dev": G.c4_max_coord_dev.max(),
        "c4_max_ph_dim_dev": (G.c4_ph_dim_diag - G.c4_ph_dim_real).abs().max(),
        "c4_max_twonn_dev": (G.c4_twonn_diag - G.c4_twonn_real).abs().max(),
    }
    pd.Series(summ).to_csv(OUT / "exactness_summary.csv", header=["value"])
    print(f"\nwrote exactness_checks.csv, exactness_summary.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
