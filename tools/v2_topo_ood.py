"""
Item 12 -- does the max-persistence + dominance rule transfer off the factorial it was tuned on?

B4 measured the rule's error rates inside one 2x2 construction: uniform disk vs annulus, clumped or
not. That is where the dominance threshold of 0.5 came from, so quoting those error rates as if they
were general would be circular. Here the SAME frozen rule meets no-loop geometries it has never
seen, and one loop-bearing family it has, and we report type-I and power per family.

No-loop families: uniform disk, uniform square, spherical cap, Swiss roll, anisotropic Gaussian,
heavy-tailed (Student-t) cloud, phase-scrambled Fourier embedding, and a covariance-preserving
surrogate of a REAL token-embedding cloud whose winding has been destroyed while its second-order
structure is kept. That last one is the important adversary: it is exactly the object the paper
declares loop-free, so if the rule fires on it the paper's negative result is unsafe.

Loop family (power): the annulus, plus a noisy circle at two thicknesses.

The rule is fixed before any of this: largest H1 bar above the 95th percentile of the longest bar
over 500 covariance-matched Gaussian nulls, AND dominance >= 0.5.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from tools.e2_topo_factorial import h1_full
from tools.b4_topo_null import null_max_lifetimes, DOM_MIN, Q, N_NULL

OUT = REPO / "RESULTS"
N, AMB, SEEDS = 300, 40, 12


def embed(P, amb, g, noise=0.03):
    Q_ = g.normal(size=(P.shape[1], amb))
    X = P @ Q_
    return X + g.normal(size=X.shape) * noise * np.abs(X).mean()


def family(name, seed):
    g = np.random.default_rng(seed)
    if name == "disk":
        z = g.normal(size=(N, 2)); z /= np.linalg.norm(z, axis=1, keepdims=True)
        P = z * g.uniform(size=(N, 1)) ** 0.5
    elif name == "square":
        P = g.uniform(-1, 1, size=(N, 2))
    elif name == "sphere_cap":
        t = g.uniform(0, 0.9, N); ph = g.uniform(0, 2 * np.pi, N)
        P = np.c_[np.sqrt(t) * np.cos(ph), np.sqrt(t) * np.sin(ph), 1 - t]
    elif name == "swiss_roll":
        t = 1.5 * np.pi * (1 + 2 * g.uniform(size=N))
        P = np.c_[t * np.cos(t), 12 * g.uniform(size=N), t * np.sin(t)] / 12
    elif name == "aniso_gauss":
        P = g.normal(size=(N, 3)) * np.array([1.0, 0.25, 0.05])
    elif name == "heavy_tail":
        P = g.standard_t(df=2, size=(N, 3)) / 3
    elif name == "phase_scrambled":
        # a Fourier embedding whose phases are randomised: the same power spectrum, no winding
        t = 2 * np.pi * np.arange(N) / N
        P = np.zeros((N, 6))
        for j, f in enumerate([1, 2, 3]):
            ph = g.uniform(0, 2 * np.pi, N)
            P[:, 2 * j], P[:, 2 * j + 1] = np.cos(f * t + ph), np.sin(f * t + ph)
    elif name == "real_cov_surrogate":
        # second-order structure of a real token-embedding cloud, winding destroyed
        E = pd.read_csv(OUT / "e7_embedding.csv")
        cols = [c for c in E.columns if c.startswith(("emb_", "dim_"))]
        C = np.eye(6)
        if len(cols) >= 3:
            V = E[cols].select_dtypes("number").dropna().values
            if V.shape[0] > 3 and V.shape[1] >= 3:
                C = np.cov(V, rowvar=False) + 1e-6 * np.eye(V.shape[1])
        L = np.linalg.cholesky(C)
        P = g.normal(size=(N, C.shape[0])) @ L.T
    elif name == "annulus":
        t = g.uniform(0, 2 * np.pi, N); r = 1.0 + g.uniform(-0.10, 0.10, N)
        P = np.c_[r * np.cos(t), r * np.sin(t)]
    elif name == "circle_thin":
        t = g.uniform(0, 2 * np.pi, N); r = 1.0 + g.uniform(-0.03, 0.03, N)
        P = np.c_[r * np.cos(t), r * np.sin(t)]
    elif name == "circle_thick":
        t = g.uniform(0, 2 * np.pi, N); r = 1.0 + g.uniform(-0.25, 0.25, N)
        P = np.c_[r * np.cos(t), r * np.sin(t)]
    else:
        raise ValueError(name)
    return embed(P, AMB, g)


NOLOOP = ["disk", "square", "sphere_cap", "swiss_roll", "aniso_gauss", "heavy_tail",
          "phase_scrambled", "real_cov_surrogate"]
LOOP = ["annulus", "circle_thin", "circle_thick"]


def one(args):
    name, seed = args
    X = family(name, seed)
    h = h1_full(X)
    nm = null_max_lifetimes(X, n_null=N_NULL, seed=5000 + seed)
    thr = float(np.percentile(nm, Q))
    return {"family": name, "has_loop": int(name in LOOP), "seed": seed,
            "H1_max": h["H1_max"], "dom_ratio": h["dom_ratio"], "n_bars": h["n_bars"],
            "thr": thr, "sig_max": int(h["H1_max"] > thr),
            "sig_joint": int(h["H1_max"] > thr and h["dom_ratio"] >= DOM_MIN)}


def main():
    from multiprocessing import Pool
    t0 = time.time()
    print("=== Item 12: the frozen rule on out-of-distribution geometries ===")
    print(f"    rule fixed in advance: H1_max > q{Q} of {N_NULL} matched nulls AND dominance "
          f">= {DOM_MIN}")
    print(f"    {len(NOLOOP)} no-loop families, {len(LOOP)} loop families, {SEEDS} seeds each\n",
          flush=True)
    jobs = [(f, s) for f in NOLOOP + LOOP for s in range(SEEDS)]
    rows = []
    with Pool(processes=7) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs), 1):
            rows.append(r)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows).sort_values(["has_loop", "family", "seed"])
    D.to_csv(OUT / "v2_topo_ood.csv", index=False)

    print(f"\n  {'family':<22}{'loop?':>7}{'max-only fires':>17}{'joint fires':>14}"
          f"{'dominance':>12}")
    summ = []
    for f, g in D.groupby("family", sort=False):
        summ.append({"family": f, "has_loop": int(g.has_loop.iloc[0]), "n": len(g),
                     "max_rate": float(g.sig_max.mean()), "joint_rate": float(g.sig_joint.mean()),
                     "dominance": float(g.dom_ratio.median())})
    S = pd.DataFrame(summ).sort_values(["has_loop", "family"])
    for _, r in S.iterrows():
        print(f"  {r.family:<22}{'yes' if r.has_loop else 'no':>7}{r.max_rate:>16.0%}"
              f"{r.joint_rate:>14.0%}{r.dominance:>12.3f}")
    S.to_csv(OUT / "v2_topo_ood_summary.csv", index=False)

    nl, lp = S[S.has_loop == 0], S[S.has_loop == 1]
    print(f"\n  pooled over out-of-distribution NO-LOOP families:")
    print(f"    max-only type-I  {nl.max_rate.mean():.1%}   worst family {nl.max_rate.max():.0%} "
          f"({nl.loc[nl.max_rate.idxmax(), 'family']})")
    print(f"    joint    type-I  {nl.joint_rate.mean():.1%}   worst family {nl.joint_rate.max():.0%} "
          f"({nl.loc[nl.joint_rate.idxmax(), 'family']})")
    print(f"  pooled over LOOP families: joint power {lp.joint_rate.mean():.1%} "
          f"(worst {lp.joint_rate.min():.0%})")
    print(f"\n  dominance separation: no-loop median {nl.dominance.median():.3f}, "
          f"loop median {lp.dominance.median():.3f}, threshold {DOM_MIN}")
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
