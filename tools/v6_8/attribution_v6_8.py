"""
v6.8 -- three attribution computations the paper was missing, all from stored grids.

  interaction     Per seed, from the raw four endpoint cells: the diagonal change, both factor
                  orders, the signed interaction I = f00 - f10 - f01 + f11, I/Delta, and the
                  symmetric allocation. Every quantity is computed PER SEED and then summarised;
                  nothing is formed by dividing medians of different cells.

  path            The endpoint decomposition uses only the corners of the 5x5 grid. Here the same
                  two-factor decomposition is evaluated on each consecutive segment (t_j, t_j+1)
                  and the four segment allocations are summed, giving a checkpoint-path
                  allocation. It is compared with the direct endpoint allocation on the same
                  seeds. Every off-diagonal cell is still counterfactual and four segments are a
                  finite partition, so this is a path SUM, not a continuous or causal attribution.

  z59            The Z_59 replication regenerated from the verified run list of
                  tools/v6_8/z59_audit.py, reported BY TRAIN FRACTION rather than pooled.

Outputs (RESULTS_v6_8/):
  endpoint_interaction.csv         per seed x estimator x cell
  endpoint_interaction_summary.csv medians and raw seed ranges
  checkpoint_path.csv              per seed x segment, raw units
  checkpoint_path_summary.csv      path totals against endpoint totals
  z59_by_train_fraction.csv        per run and per train fraction

Analysis-only: reads stored result files, no training and no re-measurement.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

R = REPO / "RESULTS"
R4 = REPO / "RESULTS_v6_4"
OUT = REPO / "RESULTS_v6_8"
OUT.mkdir(exist_ok=True)

MIN_DENOM = 0.05     # the paper's existing denominator rule, applied per seed
SEGMENTS = [(0, 1), (1, 2), (2, 3), (3, 4)]


def cells(row):
    """The four raw cells of one 2x2, as floats."""
    return float(row.f00), float(row.f10), float(row.f01), float(row.f11)


def decompose(f00, f10, f01, f11):
    """Everything the paper reports about one 2x2, in raw estimator units."""
    diag = f00 - f11                       # the observed change on the real diagonal
    c_first = f00 - f10                    # advance centroids first
    c_second = f01 - f11                   # advance centroids second
    r_first = f00 - f01                    # advance residuals first
    r_second = f10 - f11                   # advance residuals second
    inter = f00 - f10 - f01 + f11          # signed interaction
    phi_c = 0.5 * (c_first + c_second)
    return {
        "f00": f00, "f10": f10, "f01": f01, "f11": f11, "diag": diag,
        "centroid_first_raw": c_first, "centroid_second_raw": c_second,
        "residual_first_raw": r_first, "residual_second_raw": r_second,
        "interaction_raw": inter, "phi_centroid_raw": phi_c,
        "phi_residual_raw": diag - phi_c,
    }


def shares(d, denom_ok):
    """Shares only where the seed's own change clears the denominator rule."""
    out = {}
    if denom_ok:
        out["centroid_first_share"] = d["centroid_first_raw"] / d["diag"]
        out["centroid_second_share"] = d["centroid_second_raw"] / d["diag"]
        out["symmetric_share"] = d["phi_centroid_raw"] / d["diag"]
        out["interaction_over_diag"] = d["interaction_raw"] / d["diag"]
    else:
        for k in ("centroid_first_share", "centroid_second_share", "symmetric_share",
                  "interaction_over_diag"):
            out[k] = np.nan
    return out


def rng(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return (np.nan, np.nan, np.nan) if not len(v) else (np.median(v), v.min(), v.max())


# ------------------------------------------------------------------ 1. endpoint interaction
def interaction():
    S = pd.read_csv(R4 / "fullgrid_shapley.csv")
    S = S[(S.pipeline == "raw") & (S.phase == "init->conv (endpoint)")]
    rows = []
    for _, r in S.iterrows():
        d = decompose(*cells(r))
        ok = abs(d["diag"]) >= MIN_DENOM * max(abs(d["f00"]), 1e-12)
        rows.append({"estimator": r.estimator, "seed": int(r.seed), "dir": r.dir,
                     "denominator_ok": ok, **d, **shares(d, ok)})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "endpoint_interaction.csv", index=False)

    summ = []
    for est, g in D.groupby("estimator"):
        rec = {"estimator": est, "n_seeds": len(g)}
        for col in ("diag", "centroid_first_raw", "centroid_second_raw", "interaction_raw",
                    "phi_centroid_raw", "phi_residual_raw", "centroid_first_share",
                    "centroid_second_share", "symmetric_share", "interaction_over_diag"):
            m, lo, hi = rng(g[col])
            rec[f"{col}_med"], rec[f"{col}_min"], rec[f"{col}_max"] = m, lo, hi
        summ.append(rec)
    S2 = pd.DataFrame(summ)
    S2.to_csv(OUT / "endpoint_interaction_summary.csv", index=False)

    print("=== endpoint interaction, per seed then summarised ===")
    for _, r in S2.iterrows():
        print(f"  {r.estimator}")
        print(f"    diagonal change      {r.diag_med:8.2f}  [{r.diag_min:.2f}, {r.diag_max:.2f}]")
        print(f"    centroids first      {r.centroid_first_raw_med:8.2f}  "
              f"[{r.centroid_first_raw_min:.2f}, {r.centroid_first_raw_max:.2f}]")
        print(f"    centroids second     {r.centroid_second_raw_med:8.2f}  "
              f"[{r.centroid_second_raw_min:.2f}, {r.centroid_second_raw_max:.2f}]")
        print(f"    interaction I        {r.interaction_raw_med:8.2f}  "
              f"[{r.interaction_raw_min:.2f}, {r.interaction_raw_max:.2f}]")
        print(f"    I / diagonal         {r.interaction_over_diag_med:8.3f}  "
              f"[{r.interaction_over_diag_min:.3f}, {r.interaction_over_diag_max:.3f}]")
        print(f"    symmetric share      {100*r.symmetric_share_med:8.1f}%  "
              f"[{100*r.symmetric_share_min:.0f}, {100*r.symmetric_share_max:.0f}]")
        print(f"    order-specific       {100*r.centroid_first_share_med:.0f}% / "
              f"{100*r.centroid_second_share_med:.0f}%")

    # the displayed-median 2x2 is a different object from the median of per-seed calculations
    P = S[S.estimator == "ph_dim"]
    disp = decompose(P.f00.median(), P.f10.median(), P.f01.median(), P.f11.median())
    per = D[D.estimator == "ph_dim"]
    print("\n  displayed-median table against the median of per-seed calculations (PH-dim):")
    for k in ("diag", "interaction_raw", "phi_centroid_raw"):
        print(f"    {k:20s} displayed {disp[k]:7.2f}   median of per-seed "
              f"{per[k].median():7.2f}")
    return D


# ------------------------------------------------------------------ 2. checkpoint-path totals
def path():
    G = pd.read_csv(R4 / "fullgrid_recombine.csv")
    G = G[G.pipeline == "raw"]
    rows = []
    for (d_, est), _ in G.groupby(["dir", "dir"]):
        break
    for est in ("ph_dim", "twonn"):
        for d_, g in G.groupby("dir"):
            seed = int(g.seed.iloc[0])
            cell = {(int(r.s), int(r.t)): float(r[est]) for _, r in g.iterrows()}
            for a, b in SEGMENTS:
                dd = decompose(cell[(a, a)], cell[(b, a)], cell[(a, b)], cell[(b, b)])
                ok = abs(dd["diag"]) >= MIN_DENOM * max(abs(cell[(0, 0)]), 1e-12)
                rows.append({"estimator": est, "seed": seed, "dir": d_,
                             "segment": f"{a}->{b}", "a": a, "b": b,
                             "denominator_ok": ok, **dd})
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "checkpoint_path.csv", index=False)

    # segment allocations summed, against the direct endpoint allocation on the same seed
    E = pd.read_csv(OUT / "endpoint_interaction.csv")
    summ = []
    for est, g in P.groupby("estimator"):
        for seed, gs in g.groupby("seed"):
            e = E[(E.estimator == est) & (E.seed == seed)].iloc[0]
            summ.append({
                "estimator": est, "seed": seed,
                "path_centroid_raw": gs.phi_centroid_raw.sum(),
                "path_residual_raw": gs.phi_residual_raw.sum(),
                "path_diag_raw": gs.diag.sum(),
                "endpoint_centroid_raw": e.phi_centroid_raw,
                "endpoint_residual_raw": e.phi_residual_raw,
                "endpoint_diag_raw": e.diag,
                "path_abs_interaction": gs.interaction_raw.abs().sum(),
                "endpoint_abs_interaction": abs(e.interaction_raw),
            })
    S = pd.DataFrame(summ)
    S["path_share"] = S.path_centroid_raw / S.path_diag_raw
    S["endpoint_share"] = S.endpoint_centroid_raw / S.endpoint_diag_raw
    S["share_gap_pp"] = 100 * (S.path_share - S.endpoint_share)
    S.to_csv(OUT / "checkpoint_path_summary.csv", index=False)

    print("\n=== checkpoint-path allocation (sum of four consecutive segments) ===")
    for est, g in S.groupby("estimator"):
        print(f"  {est}   n = {len(g)} seeds")
        for nm, col in (("diagonal, path sum", "path_diag_raw"),
                        ("diagonal, endpoint", "endpoint_diag_raw"),
                        ("centroid, path sum", "path_centroid_raw"),
                        ("centroid, endpoint", "endpoint_centroid_raw")):
            m, lo, hi = rng(g[col])
            print(f"    {nm:22s} {m:8.2f}  [{lo:.2f}, {hi:.2f}]")
        m, lo, hi = rng(100 * g.path_share)
        print(f"    {'path centroid share':22s} {m:8.1f}%  [{lo:.0f}, {hi:.0f}]")
        m, lo, hi = rng(100 * g.endpoint_share)
        print(f"    {'endpoint share':22s} {m:8.1f}%  [{lo:.0f}, {hi:.0f}]")
        m, lo, hi = rng(g.share_gap_pp)
        print(f"    {'gap (path - endpoint)':22s} {m:8.1f} pp  [{lo:.1f}, {hi:.1f}]")
    return P, S


# ------------------------------------------------------------------ 3. Z_59 by train fraction
def z59():
    audit = pd.read_csv(OUT / "z59_run_audit.csv")
    keep = audit[audit.in_recombination]
    tf = dict(zip(keep.dir, keep.train_fraction))
    seed = dict(zip(keep.dir, keep.seed))

    C = pd.read_csv(R / "v3_crosscell.csv")
    C = C[(C.cell == "Z59_add_mlp") & (C.frame == "unaligned")]
    rows = []
    for est in ("ph_dim", "twonn"):
        for d_, g in C.groupby("dir"):
            if d_ not in tf:
                continue
            cell = {(int(r.C), int(r.R)): float(r[est]) for _, r in g.iterrows()}
            dd = decompose(cell[(0, 0)], cell[(1, 0)], cell[(0, 1)], cell[(1, 1)])
            ok = abs(dd["diag"]) >= MIN_DENOM * max(abs(dd["f00"]), 1e-12)
            rows.append({"estimator": est, "dir": d_, "seed": seed[d_],
                         "train_fraction": tf[d_], "denominator_ok": ok,
                         **dd, **shares(dd, ok)})
    Z = pd.DataFrame(rows).sort_values(["estimator", "train_fraction", "seed"])
    Z.to_csv(OUT / "z59_by_train_fraction.csv", index=False)

    print("\n=== Z_59 replication, by train fraction (never pooled as repeated seeds) ===")
    for est, g in Z.groupby("estimator"):
        print(f"  {est}")
        for t_, gt in g.groupby("train_fraction"):
            m, lo, hi = rng(100 * gt.symmetric_share)
            print(f"    tf {t_:g}  n = {len(gt)} (seeds {','.join(str(int(s)) for s in sorted(gt.seed))})"
                  f"   symmetric {m:6.0f}%  [{lo:.0f}, {hi:.0f}]")
        m, lo, hi = rng(100 * g.symmetric_share)
        print(f"    heterogeneous pooled median {m:.0f}%  [{lo:.0f}, {hi:.0f}]  (descriptive only)")
        mi, _, _ = rng(g.interaction_raw)
        print(f"    interaction median {mi:.2f} estimator units")
    return Z


if __name__ == "__main__":
    interaction()
    path()
    z59()
    print("\nwrote RESULTS_v6_8/{endpoint_interaction,checkpoint_path,z59_by_train_fraction}*.csv")
