"""
STEP 1 gate. Every estimator in analysis/geom.py against a synthetic with a KNOWN answer.
Nothing that fails here is used in STEP 2 or STEP 3.

  dimension estimators   uniform d-ball embedded in 60-dim: all four families must return ~d
                         Swiss roll: 2 (a curved 2-manifold in 3-space -- the case where
                         LINEAR estimators are supposed to fail and local ones are not)
  curvature              sphere > 0, flat grid ~ 0, tree < 0
  magnitude              monotone increasing in t, and larger for a more spread-out cloud
  neural collapse        perfect ETF -> nc1 ~ 0, nc2 ~ 0; random labels -> large
  local ID               bimodal by construction -> bimodality coefficient above 5/9

The Swiss-roll row is the important one for the paper's W2 claim: it is the known case where
fractal/local dimension (2) and linear dimension (3) MUST disagree. If our estimators do not
reproduce that disagreement on a known example, any divergence we later see in activations is
not evidence of anything.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from analysis import geom

OUT = REPO / "RESULTS"


def ball(n, d, amb=60, seed=0):
    g = np.random.default_rng(seed)
    Z = g.normal(size=(n, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    Z *= g.uniform(size=(n, 1)) ** (1 / d)
    return Z @ g.normal(size=(d, amb)) if amb > d else Z


def swiss_roll(n, seed=0):
    g = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * g.uniform(size=n))
    h = 21 * g.uniform(size=n)
    return np.c_[t * np.cos(t), h, t * np.sin(t)]


def sphere(n, d=2, seed=0):
    g = np.random.default_rng(seed)
    Z = g.normal(size=(n, d + 1))
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def grid2d(n, seed=0):
    m = int(np.sqrt(n))
    a = np.linspace(0, 1, m)
    X, Y = np.meshgrid(a, a)
    return np.c_[X.ravel(), Y.ravel()]


def tree_points(depth=8, branch=2, seed=0):
    """Points embedded so their kNN graph is tree-like: hyperbolic-ish, curvature < 0."""
    g = np.random.default_rng(seed)
    pts = [np.zeros(2)]
    frontier = [(np.zeros(2), 0.0, 1.0)]
    for lvl in range(depth):
        nf = []
        for p, ang, r in frontier:
            for b in range(branch):
                a2 = ang + (b - (branch - 1) / 2) * (0.9 / (lvl + 1)) + g.normal() * 0.01
                q = p + r * np.array([np.cos(a2), np.sin(a2)])
                pts.append(q); nf.append((q, a2, r * 0.75))
        frontier = nf
        if len(pts) > 600:
            break
    return np.array(pts)


if __name__ == "__main__":
    rows = []
    print("=== GATE 1: dimension estimators on uniform d-balls in 60-dim ambient ===\n")
    for d in (2, 3, 5, 8):
        X = ball(600, d)
        b = geom.battery(X, cheap=True)
        rows.append({"case": f"{d}-ball", "true": d, **b})
        print(f"  d={d}:  ph_dim={b['ph_dim']:6.2f} (r2 {b['ph_dim_r2']:.2f})  corr_dim={b['corr_dim']:6.2f}"
              f"  twonn={b['twonn']:6.2f}  mle={b['mle_id']:6.2f}  pca_pr={b['pca_pr']:6.2f}"
              f"  eff_rank={b['eff_rank']:6.2f}", flush=True)

    print("\n=== GATE 2 (KEY for W2): Swiss roll -- fractal/local=2, linear=3 MUST disagree ===\n")
    X = swiss_roll(800)
    b = geom.battery(X, cheap=True)
    rows.append({"case": "swiss roll", "true": 2, **b})
    print(f"  ph_dim={b['ph_dim']:.2f}  corr_dim={b['corr_dim']:.2f}  twonn={b['twonn']:.2f}  "
          f"mle={b['mle_id']:.2f}   |   pca_pr={b['pca_pr']:.2f}  eff_rank={b['eff_rank']:.2f}")
    loc = np.nanmean([b["corr_dim"], b["twonn"], b["mle_id"]])
    print(f"  -> local/fractal mean {loc:.2f} (true 2)   linear {b['pca_pr']:.2f} (ambient 3)")
    print(f"  -> DIVERGENCE detected: {b['pca_pr'] - loc:+.2f}  {'PASS' if b['pca_pr'] - loc > 0.4 else 'FAIL'}")

    print("\n=== GATE 3a: Ollivier-Ricci on EXPLICIT graphs with textbook curvature signs ===\n")
    import networkx as nx
    from scipy.sparse import csr_matrix
    for nm, Gx, want in (("complete K_30", nx.complete_graph(30), "strongly > 0"),
                         ("cycle C_60", nx.cycle_graph(60), "~ 0"),
                         ("2D lattice 12x12", nx.grid_2d_graph(12, 12), "~ 0"),
                         ("balanced tree b2 d6", nx.balanced_tree(2, 6), "< 0"),
                         ("random 3-regular", nx.random_regular_graph(3, 120, seed=0), "< 0")):
        A = csr_matrix(nx.to_scipy_sparse_array(Gx, format="csr"), dtype=float)
        o = geom.ollivier_ricci_graph(A, max_edges=900)
        ok = (("> 0" in want and o["orc_mean"] > 0.05) or
              ("< 0" in want and o["orc_mean"] < -0.02) or
              ("~ 0" in want and abs(o["orc_mean"]) < 0.10))
        print(f"  {nm:20s} want {want:12s} orc_mean={o['orc_mean']:+.4f}  "
              f"frac_neg={o['orc_frac_neg']:.2f}   {'PASS' if ok else 'FAIL'}")
        rows.append({"case": f"orc {nm}", "true": np.nan, **o})

    print("\n=== GATE 3b: Ollivier-Ricci from POINT CLOUDS via their kNN graph ===\n")
    for nm, X, want in (("2-sphere", sphere(400, 2), "> 0"),
                        ("flat 2D grid", grid2d(400), "~ 0"),
                        ("tree-embedded pts", tree_points(), "< 0")):
        o = geom.ollivier_ricci(X, k=8, max_edges=900)
        print(f"  {nm:20s} want {want:4s}   orc_mean={o['orc_mean']:+.4f}  "
              f"frac_neg={o['orc_frac_neg']:.2f}")
        rows.append({"case": f"orc pc {nm}", "true": np.nan, **o})

    print("\n=== GATE 4: magnitude (unit-scaled) monotone in t, bigger for spread-out clouds ===\n")
    for nm, X in (("tight 2-ball", ball(300, 2, amb=0)), ("8-ball", ball(300, 8, amb=0))):
        m = geom.magnitude_function(X)
        v = [m[k] for k in sorted(m, key=lambda s: float(s.split("t")[1]))]
        print(f"  {nm:14s} " + "  ".join(f"{k}={m[k]:8.2f}" for k in
                                         sorted(m, key=lambda s: float(s.split('t')[1]))))
        print(f"                 monotone in t: {all(np.diff(v) > 0)}")

    print("\n=== GATE 5: neural collapse ===\n")
    g = np.random.default_rng(0)
    K, per = 10, 40
    lab = np.repeat(np.arange(K), per)
    M = np.eye(K)
    perfect = M[lab] * 3.0 + g.normal(size=(K * per, K)) * 0.01
    loose = M[lab] * 3.0 + g.normal(size=(K * per, K)) * 1.5
    rand = g.normal(size=(K * per, K))
    for nm, Z in (("collapsed (tight)", perfect), ("loose clusters", loose), ("no structure", rand)):
        nc = geom.neural_collapse(Z, lab)
        print(f"  {nm:20s} nc1={nc['nc1']:9.4f}  nc2_etf={nc['nc2_etf']:.4f}  wb={nc['nc_wb']:9.4f}")

    print("\n=== GATE 6: local-ID distribution shape on a deliberately bimodal cloud ===\n")
    A = ball(300, 2, amb=20, seed=1)
    B = ball(300, 9, amb=20, seed=2) + 50.0
    v1 = geom.lid_distribution(A, 20); v2 = geom.lid_distribution(np.vstack([A, B]), 20)
    print(f"  single 2-ball        mean_lid={v1.mean():5.2f}  bimodality={geom.bimodality(v1):.3f}")
    print(f"  2-ball + 9-ball mix  mean_lid={v2.mean():5.2f}  bimodality={geom.bimodality(v2):.3f}"
          f"   (>0.556 = bimodal)")

    pd.DataFrame(rows).to_csv(OUT / "geom_validation.csv", index=False)
    print("\nwrote RESULTS/geom_validation.csv")
