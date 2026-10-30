"""
T3.4 -- topology-rule hardening: diffuse and multi-frequency loop alternatives, and binomial
confidence intervals on every error rate.

The frozen rule (H1_max > q95 of 500 covariance-matched nulls AND dominance >= 0.5) was
developed on the B4 factorial and evaluated on held-out families in v2 (v2_topo_ood.csv). Two
gaps remain. (1) The loop alternatives tested were single circles; the REAL representation is a
few-frequency Fourier torus, whose loop signal is split across frequencies and radii -- the
diffuse alternative that motivated the conservative reading. (2) Rates were reported without
uncertainty at n = 12 seeds.

Adds: fourier3_torus (3-frequency embedding of Z_n, the Nanda/Chughtai shape), circle_2freq
(circle with strong second-harmonic wobble), and torus_2d (product of two circles), 12 seeds
each under the SAME frozen rule; then Wilson 95% intervals for every family old and new.

Output: RESULTS/v3_topo_extra.csv, v3_topo_cis.csv
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
from tools.v2_topo_ood import embed, AMB, N

OUT = REPO / "RESULTS"
SEEDS = 12


def family(name, seed):
    g = np.random.default_rng(seed)
    t = 2 * np.pi * np.arange(N) / N
    if name == "fourier3_torus":
        # the grokked-representation shape: 3 incommensurate frequencies, equal amplitude
        fs = g.choice(np.arange(1, 12), 3, replace=False)
        P = np.concatenate([np.c_[np.cos(f * t + g.uniform(0, 2 * np.pi)),
                                  np.sin(f * t + g.uniform(0, 2 * np.pi))] for f in fs], 1)
    elif name == "circle_2freq":
        r = 1.0 + 0.35 * np.cos(2 * t + g.uniform(0, 2 * np.pi))
        P = np.c_[r * np.cos(t), r * np.sin(t)]
    elif name == "torus_2d":
        u_, v_ = g.uniform(0, 2 * np.pi, N), g.uniform(0, 2 * np.pi, N)
        P = np.c_[np.cos(u_), np.sin(u_), 0.5 * np.cos(v_), 0.5 * np.sin(v_)]
    else:
        raise ValueError(name)
    return embed(P, AMB, g)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def one(args):
    name, seed = args
    X = family(name, seed)
    h = h1_full(X)
    nm = null_max_lifetimes(X, n_null=N_NULL, seed=7000 + seed)
    thr = float(np.percentile(nm, Q))
    return {"family": name, "has_loop": 1, "seed": seed, "H1_max": h["H1_max"],
            "dom_ratio": h["dom_ratio"], "n_bars": h["n_bars"], "thr": thr,
            "sig_max": int(h["H1_max"] > thr),
            "sig_joint": int(h["H1_max"] > thr and h["dom_ratio"] >= DOM_MIN)}


def main():
    from multiprocessing import Pool
    t0 = time.time()
    fams = ["fourier3_torus", "circle_2freq", "torus_2d"]
    jobs = [(f, s) for f in fams for s in range(SEEDS)]
    rows = []
    with Pool(processes=6) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs), 1):
            rows.append(r)
            print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    E = pd.DataFrame(rows)
    E.to_csv(OUT / "v3_topo_extra.csv", index=False)

    D = pd.read_csv(OUT / "v2_topo_ood.csv")
    ALL = pd.concat([D, E], ignore_index=True)
    out = []
    print(f"\n  {'family':<18}{'loop?':>6}{'joint rate':>12}{'Wilson 95%':>18}"
          f"{'dominance':>11}")
    for f, g in ALL.groupby("family"):
        k, n = int(g.sig_joint.sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({"family": f, "has_loop": int(g.has_loop.iloc[0]), "n": n,
                    "joint_rate": k / n, "wilson_lo": lo, "wilson_hi": hi,
                    "max_rate": float(g.sig_max.mean()),
                    "dominance": float(g.dom_ratio.median())})
        print(f"  {f:<18}{'yes' if g.has_loop.iloc[0] else 'no':>6}{k/n:>11.0%}"
              f"     [{lo:>4.0%}, {hi:>4.0%}]{g.dom_ratio.median():>11.3f}")
    S = pd.DataFrame(out)
    S.to_csv(OUT / "v3_topo_cis.csv", index=False)
    nl = S[S.has_loop == 0]
    k, n = int((nl.joint_rate * nl.n).round().sum()), int(nl.n.sum())
    lo, hi = wilson(k, n)
    print(f"\n  pooled no-loop joint type-I: {k}/{n} = {k/n:.1%}  Wilson [{lo:.1%}, {hi:.1%}]")


if __name__ == "__main__":
    main()
