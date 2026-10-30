"""
P1 -- TORUS, not circle. Is the number of independent circles a discrete topological
order parameter for grokking?

A modular-arithmetic representation is a superposition of k Fourier modes. Each mode is
a circle; together the residues wind a k-torus. So the natural topological quantity is
not "is there a loop" (ambient PH says no) but

        k = the number of INDEPENDENT circular coordinates the representation supports

which is a discrete, basis-free count. We extract it by taking the top-K persistent H^1
cocycles, building a circular coordinate from each, and counting how many carry coherent
winding against a per-run null.

Independence check: two cocycles that recover the same winding number are the same
circle seen twice, so windings are deduplicated (up to sign / Nyquist alias).

Instrument validation first, as always: synthetic embeddings built from a KNOWN number
of modes must return that number.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from analysis.circcoords import circular_coordinate, winding_fit

OUT = REPO / "RESULTS"


def torus_spectrum(E, u, K=6, coeff=17, n_null=10, seed=0, z_thresh=2.0):
    """
    Returns dict with k (number of independent coherent circles), the winding numbers,
    and the per-cocycle z scores against a row-permutation null.
    """
    p = E.shape[0]
    reals = []
    for rank in range(K):
        th, info = circular_coordinate(E, coeff=coeff, rank=rank, eps_frac=0.5)
        if th is None:
            break
        w = winding_fit(th, u)
        reals.append({"rank": rank, "winding": w["winding"], "resid": w["circ_resid"],
                      "lifetime": info["lifetime"]})
    if not reals:
        return {"k": 0, "windings": [], "z": []}

    # null: same procedure on row-permuted embeddings, per rank
    rng = np.random.default_rng(seed)
    null_by_rank = {r: [] for r in range(len(reals))}
    for _ in range(n_null):
        Ep = E[rng.permutation(p)]
        for rank in range(len(reals)):
            th2, _ = circular_coordinate(Ep, coeff=coeff, rank=rank, eps_frac=0.5)
            if th2 is None:
                continue
            null_by_rank[rank].append(winding_fit(th2, u)["circ_resid"])

    kept, zs = [], []
    for rec in reals:
        ns = np.array(null_by_rank.get(rec["rank"], []), float)
        if ns.size < 3 or ns.std() <= 0:
            continue
        z = (ns.mean() - rec["resid"]) / ns.std()
        zs.append(round(float(z), 2))
        if z >= z_thresh:
            kept.append(rec["winding"])

    # deduplicate: w and -w and w+/-p are the same circle
    seen, uniq = set(), []
    for w in kept:
        if w is None:
            continue
        key = min(abs(w) % p, p - (abs(w) % p))
        if key not in seen and key != 0:
            seen.add(key); uniq.append(w)
    return {"k": len(uniq), "windings": uniq, "z": zs, "n_cocycles": len(reals)}


def synth(p, freqs, amb=200, seed=0, noise=0.0):
    n = np.arange(p); r = np.random.default_rng(seed); E = np.zeros((p, amb))
    for k in freqs:
        E += np.c_[np.cos(2 * np.pi * k * n / p), np.sin(2 * np.pi * k * n / p)] @ r.normal(size=(2, amb))
    if noise:
        E += r.normal(size=E.shape) * noise * E.std()
    return E


if __name__ == "__main__":
    print("=== P1 INSTRUMENT VALIDATION: recover a KNOWN number of modes ===\n")
    rows = []
    for p in (23, 59):
        u = np.arange(p) / p
        for ktrue in (1, 2, 3, 5):
            for seed in (0, 1, 2):
                r = np.random.default_rng(100 + seed)
                freqs = sorted(r.choice(np.arange(1, p // 2), size=ktrue, replace=False).tolist())
                res = torus_spectrum(synth(p, freqs, seed=seed), u, K=6, seed=seed)
                rows.append({"p": p, "k_true": ktrue, "seed": seed, "k_est": res["k"],
                             "freqs": str(freqs), "windings": str(res["windings"])})
        # null: no structure at all
        for seed in (0, 1, 2):
            E = np.random.default_rng(seed).normal(size=(p, 200))
            rows.append({"p": p, "k_true": 0, "seed": seed,
                         "k_est": torus_spectrum(E, u, K=6, seed=seed)["k"],
                         "freqs": "-", "windings": "-"})
        print(f"  p={p} done", flush=True)
    V = pd.DataFrame(rows); V.to_csv(OUT / "p1_validation.csv", index=False)
    pd.set_option("display.width", 220)
    print()
    print(V.groupby(["p", "k_true"]).agg(n=("k_est", "size"), k_est_mean=("k_est", "mean"),
                                         k_est_min=("k_est", "min"), k_est_max=("k_est", "max")).round(2).to_string())
    exact = (V.k_est == V.k_true).mean()
    print(f"\n  exact recovery of k: {exact:.0%}")
    print(f"  mean |error|: {(V.k_est - V.k_true).abs().mean():.2f}")
