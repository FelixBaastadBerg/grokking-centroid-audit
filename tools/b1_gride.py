"""
B1 -- scale-locality inside a SINGLE estimator family, using GRIDE.

Every other comparison in this paper is between different estimators, so a sceptic can always say
the split tracks some other difference between them (bandwidth choice, fitting method, statistic).
GRIDE (Denti et al., Sci. Rep. 2022) removes that objection: it is one estimator with one knob,
mu = r_{n2}/r_{n1} at n2 = 2*n1, and n1 sets how far the neighbourhood reaches. TwoNN is exactly
GRIDE at n1 = 1.

The scale-locality account makes a quantitative prediction here, not just a directional one. On a
cloud of K clusters holding m points each, the n2-th neighbour of a point leaves that point's own
cluster once n2 >= m, i.e. once

        n1 >= m / 2.

So the estimate should sit at the true within-cluster dimension for n1 below m/2 and fall away
above it, and the crossover should MOVE with m. That is the falsifiable form of the claim.

GATES, run before any sweep and reported in the output:
  G1  GRIDE at n1 = 1 must reproduce TwoNN on the same cloud (the densities coincide).
  G2  GRIDE must recover known d on unclustered clouds, at the small n1 where it is meant to be
      accurate.
  G3  the crossover must not appear on an UNCLUSTERED cloud.

G3 FAILED ON THE RAW CURVE, and that failure is why this file reports a ratio rather than a curve.
On an unclustered cloud of 256 points GRIDE drifts 5.09 -> 3.29 as n1 goes 1 -> 64 (1.55x), because
a neighbourhood of order n1 in a cloud of n points stops being local once n1 approaches n: boundary
and curvature effects bias every ratio estimator downward at large order. Reading the raw decline as
clustering would be precisely the error this paper is about.

The fix is the pairing already used for the (m, gamma) map in the main text: measure the SAME n1 on
a matched unclustered cloud -- same total point count, same true within-patch dimension, same seed,
same ambient space, differing only in whether the patches are separated -- and analyse the ratio

        R(n1)  =  GRIDE_clustered(n1) / GRIDE_unclustered(n1),

which divides the n1-dependent finite-sample bias out. G3' then checks that the pairing does not
itself manufacture a transition: on two independent UNCLUSTERED clouds the ratio must stay flat.

Analysis-only: synthetic clouds plus the retained real activations, no training.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from analysis import geom
from analysis.lsrun import LandscapeRun
from tools.e1_centroid import decompose, recompose

OUT = REPO / "RESULTS"
N1S = [1, 2, 4, 8, 16, 32, 64]
D_TRUE, AMB = 5, 60


def cloud(d, K, m, sigma, seed=0, amb=AMB):
    """K clusters of m points, each an isotropic d-manifold of scale sigma; true within-cluster d."""
    g = np.random.default_rng(seed)
    P = g.normal(size=(d, amb))
    cen = g.normal(size=(K, amb)) * 3
    X = np.vstack([cen[k] + (g.normal(size=(m, d)) @ P) * sigma for k in range(K)])
    y = np.repeat(np.arange(K), m)
    return X, y


def gates():
    print("=== GATES ===")
    ok = True

    print("  G1  GRIDE(n1=1) vs TwoNN on unclustered clouds of known d")
    rows = []
    for d in (2, 5, 8):
        X, _ = cloud(d, 1, 800, 1.0, seed=1)
        g1, tw = geom.gride(X, 1), geom.twonn(X)
        rows.append({"gate": "G1", "true_d": d, "gride_n1_1": g1, "twonn": tw})
        agree = abs(g1 - tw) / max(tw, 1e-9) < 0.15
        ok &= agree
        print(f"      d={d:>2}   GRIDE {g1:6.2f}   TwoNN {tw:6.2f}   "
              f"{'agree' if agree else 'DISAGREE'}")

    print("  G2  GRIDE recovers known d on UNCLUSTERED clouds at the small n1 it is accurate at")
    for d in (2, 5, 8):
        X, _ = cloud(d, 1, 800, 1.0, seed=2)
        vals = [geom.gride(X, n) for n in N1S]
        small = [v for n, v in zip(N1S, vals) if n <= 8]
        err = max(abs(v - d) for v in small)
        ok &= err < 0.35 * d
        rows += [{"gate": "G2", "true_d": d, "n1": n, "gride": v} for n, v in zip(N1S, vals)]
        print(f"      d={d:>2}   " + "  ".join(f"n1={n}:{v:.2f}" for n, v in zip(N1S, vals))
              + f"   err(n1<=8) {err:.2f}  {'ok' if err < 0.35 * d else 'FAIL'}")

    print("  G3  raw curve on an UNCLUSTERED cloud -- quantifying the bias the pairing removes")
    X, _ = cloud(D_TRUE, 1, 8 * 32, 1.0, seed=3)
    vals = np.array([geom.gride(X, n) for n in N1S])
    spread = float(np.nanmax(vals) / max(np.nanmin(vals), 1e-9))
    rows += [{"gate": "G3", "true_d": D_TRUE, "n1": n, "gride": v} for n, v in zip(N1S, vals)]
    print("      " + "  ".join(f"n1={n}:{v:.2f}" for n, v in zip(N1S, vals))
          + f"   spread {spread:.2f}x  -> raw curves are NOT usable; analyse the paired ratio")

    print("  G3' the pairing must not manufacture a transition: ratio of two INDEPENDENT "
          "unclustered clouds")
    rat = []
    for s in (0, 1, 2):
        A, _ = cloud(D_TRUE, 8, 32, 1.5, seed=100 + s)
        B, _ = cloud(D_TRUE, 8, 32, 1.5, seed=200 + s)
        rat.append([geom.gride(A, n) / geom.gride(B, n) for n in N1S])
    rat = np.array(rat)
    med = np.median(rat, axis=0)
    dev = float(np.nanmax(np.abs(med - 1.0)))
    flat = dev < 0.20
    ok &= flat
    rows += [{"gate": "G3prime", "true_d": D_TRUE, "n1": n, "ratio": v}
             for n, v in zip(N1S, med)]
    print("      " + "  ".join(f"n1={n}:{v:.2f}" for n, v in zip(N1S, med))
          + f"   max |ratio-1| {dev:.2f}  {'flat' if flat else 'NOT FLAT -> void'}")

    pd.DataFrame(rows).to_csv(OUT / "b1_gride_gates.csv", index=False)
    print(f"\n  GATES {'PASSED' if ok else 'FAILED'}   "
          f"(raw-curve bias {spread:.2f}x is removed by pairing)\n")
    return ok


def synthetic_sweep(seeds=(0, 1, 2)):
    """GRIDE vs n1 on clustered clouds at several m. Prediction: crossover at n1 ~ m/2."""
    print("=== synthetic: GRIDE vs neighbourhood order, clustered clouds (true within-d = 5) ===")
    rows = []
    for m in (8, 16, 32, 64):
        for sigma, lab in ((0.05, "clustered"), (1.5, "unclustered")):
            for s in seeds:
                X, _ = cloud(D_TRUE, 8, m, sigma, seed=s)
                for n1 in N1S:
                    if 2 * n1 >= len(X):
                        continue
                    rows.append({"m": m, "sigma": sigma, "regime": lab, "seed": s, "n1": n1,
                                 "gride": geom.gride(X, n1)})
    S = pd.DataFrame(rows)
    # paired ratio: same m, same seed, same n1, clustered over unclustered
    w = S.pivot_table(index=["m", "seed", "n1"], columns="regime", values="gride").reset_index()
    w["ratio"] = w["clustered"] / w["unclustered"]
    P = w.pivot_table(index="m", columns="n1", values="ratio")
    print("  paired ratio clustered/unclustered, median over 3 seeds "
          "(1.0 = clustering has no effect):")
    print("   m \\ n1 " + "".join(f"{n:>8}" for n in P.columns))
    for m, r in P.iterrows():
        print(f"   {m:>5}  " + "".join(f"{v:>8.2f}" for v in r.values) + f"    m/2 = {m//2}")
    return S, w


def crossover(w, drop=0.5):
    """First n1 at which the paired ratio has fallen to `drop` of its sub-gap value."""
    out = []
    for (m, s), g in w.groupby(["m", "seed"]):
        g = g.sort_values("n1")
        base = g[g.n1 <= 2].ratio.mean()
        below = g[g.ratio <= drop * base]
        out.append({"m": m, "seed": s, "base_ratio": base,
                    "n1_cross": float(below.n1.min()) if len(below) else np.nan,
                    "predicted": m / 2.0})
    return pd.DataFrame(out)


def real_sweep():
    """Same knob on real activations, at the two ends of the centroid intervention."""
    print("\n=== real activations: GRIDE vs n1 at alpha = 0 (unclustered) and alpha = 8 ===")
    inv = pd.read_csv(OUT / "_inventory.csv")
    sel = inv[(inv.arch == "mlp") & (inv.test >= 0.9)].drop_duplicates(["task", "p"])
    rows = []
    for _, rec in sel.iterrows():
        try:
            r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
            r._load(r.flat(len(r.epochs) - 1)); r.model.eval()
            with torch.no_grad():
                r.model(r.Xtr, keep_activations=True)
                H = r.model.activations[-2].numpy()
            y = r.Ytr.numpy()
        except Exception as e:
            print(f"  skip {rec['dir'][:40]}: {type(e).__name__}"); continue
        sub = np.sort(np.random.default_rng(0).choice(len(y), min(400, len(y)), replace=False))
        H, y = H[sub], y[sub]
        m_real = len(y) / len(np.unique(y))
        mu, bet, wit = decompose(H, y)
        for a in (0.0, 8.0):
            Z = recompose(mu, bet, wit, a, 1.0, norm="trace")
            for n1 in N1S:
                rows.append({"task": rec["task"], "p": int(rec["p"]), "dir": rec["dir"],
                             "points_per_class": m_real, "alpha": a, "n1": n1,
                             "gride": geom.gride(Z, n1)})
        print(f"  {rec['task']}/p{rec['p']}  m={m_real:.0f}", flush=True)
    R = pd.DataFrame(rows)
    if not len(R):
        return R
    # same pairing as the synthetic arm: alpha=8 over alpha=0 at matched n1, run and cloud, so the
    # n1-dependent finite-sample bias cancels
    w = R.pivot_table(index=["dir", "task", "p", "points_per_class", "n1"],
                      columns="alpha", values="gride").reset_index()
    w["ratio"] = w[8.0] / w[0.0]
    R = R.merge(w[["dir", "n1", "ratio"]], on=["dir", "n1"], how="left")
    P = w.pivot_table(index="points_per_class", columns="n1", values="ratio")
    print("  paired ratio alpha=8 / alpha=0, median over runs "
          "(crossover = first n1 at half the n1=1 value, same rule as the synthetic arm):")
    print("   m \\ n1  " + "".join(f"{n:>8}" for n in P.columns))
    for m_, r_ in P.iterrows():
        v = r_.values
        below = [n for n, x in zip(P.columns, v) if np.isfinite(x) and x <= 0.5 * v[0]]
        cross = below[0] if below else np.nan
        print(f"   {m_:>5.0f}  " + "".join(f"{x:>8.2f}" for x in v)
              + f"    crossover n1={cross}, predicted m/2 = {m_/2:.1f}")
    return R


if __name__ == "__main__":
    t0 = time.time()
    if not gates():
        print("GATES FAILED -- B1 is void, do not use."); raise SystemExit(1)
    S, w = synthetic_sweep()
    C = crossover(w)
    print("\n  crossover (first n1 where the paired ratio halves), median over seeds:")
    for m, g in C.groupby("m"):
        obs = g.n1_cross.median()
        print(f"      m = {m:>3}   observed n1 = {obs:>5.1f}   predicted m/2 = {m/2:>5.1f}")
    S.to_csv(OUT / "b1_gride_synth.csv", index=False)
    w.to_csv(OUT / "b1_gride_ratio.csv", index=False)
    C.to_csv(OUT / "b1_gride_crossover.csv", index=False)
    R = real_sweep()
    R.to_csv(OUT / "b1_gride_real.csv", index=False)
    print(f"\nwrote b1_gride_*.csv   ({time.time()-t0:.0f}s)")
