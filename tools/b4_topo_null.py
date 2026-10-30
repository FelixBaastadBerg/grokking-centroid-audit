"""
B4 -- recalibrate the topology significance rule, and measure what it actually achieves.

The E2 factorial thresholds a bar at the 95th percentile of the longest-bar lifetime over 12
covariance-matched Gaussian nulls. Twelve draws cannot pin a 95th percentile: the order statistic
is the maximum of the sample, so the nominal 5% level is estimated with enormous variance and the
realised type-I error is unknown. Every "no persistent class above the calibrated threshold"
statement in the paper rests on that rule, so it has to be measured rather than assumed.

What this does, all compute and no training:

  1. Redraw the null with N_NULL = 500 matched Gaussians per cloud and record how the threshold
     moves relative to the 12-draw version, plus a bootstrap interval on the threshold itself.
  2. Fix the DECISION RULE before measuring it, in two variants:
       max-only   the largest bar must exceed the 500-draw 95th percentile;
       joint      the same, AND dominance (longest bar / total persistence) >= DOM_MIN.
     Only the largest bar is ever tested, so there is no multiplicity to correct -- that is the
     reason for testing the max rather than every bar, and it is stated as a design choice.
  3. Report the EMPIRICAL type-I error (fraction of no-loop clouds declared significant) and POWER
     (fraction of loop clouds declared significant), separately at low and high clustering, over
     many more construction seeds than the factorial used.

The claim this licenses is deliberately narrow: within this factorial, the joint rule separates the
loop from the no-loop condition at a measured error rate. It is not a general sufficiency theorem.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from tools.e2_topo_factorial import make, h1_full

OUT = REPO / "RESULTS"
N_NULL = 500          # was 12
N_SEEDS = 30          # construction seeds per cell; the factorial used 6
DOM_MIN = 0.50        # dominance floor for the joint rule, fixed before any measurement
Q = 95


def null_max_lifetimes(X, n_null=N_NULL, seed=0):
    """Longest-bar lifetime over covariance-matched Gaussian nulls of the same size."""
    g = np.random.default_rng(seed)
    Xc = np.asarray(X, float) - np.mean(X, 0)
    C = np.cov(Xc, rowvar=False)
    C = C + 1e-9 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    L = np.linalg.cholesky(C)
    return np.array([h1_full(g.normal(size=(len(X), C.shape[0])) @ L.T)["H1_max"]
                     for _ in range(n_null)])


def one_cloud(args):
    """One (topology, clustering, seed) cell: the cloud, its 500-null threshold, both verdicts."""
    topo, clus, seed = args
    X = make(topo, clus, seed=seed)
    h = h1_full(X)
    nm = null_max_lifetimes(X, seed=1000 + seed)
    thr = float(np.percentile(nm, Q))
    thr12 = float(np.percentile(nm[:12], Q))     # what the 12-draw rule would have said
    g = np.random.default_rng(7)
    bs = [np.percentile(g.choice(nm, len(nm), replace=True), Q) for _ in range(500)]
    return {"topology": topo, "clustering": clus, "seed": seed,
            "H1_max": h["H1_max"], "H1_total": h["H1_total"],
            "n_bars": h["n_bars"], "dom_ratio": h["dom_ratio"],
            "thr500": thr, "thr12": thr12,
            "thr_lo": float(np.percentile(bs, 2.5)),
            "thr_hi": float(np.percentile(bs, 97.5)),
            "sig_max": int(h["H1_max"] > thr),
            "sig_joint": int(h["H1_max"] > thr and h["dom_ratio"] >= DOM_MIN)}


def main():
    from multiprocessing import Pool
    t0 = time.time()
    print("=== B4: topology null recalibration ===")
    print(f"    {N_NULL} matched nulls per cloud (was 12), {N_SEEDS} construction seeds per cell")
    print(f"    rules fixed in advance -- max-only: H1_max > q{Q}(null);  "
          f"joint: that AND dominance >= {DOM_MIN}\n", flush=True)

    jobs = [(t, c, s) for t in ("noloop", "loop") for c in ("low", "high")
            for s in range(N_SEEDS)]
    rows = []
    with Pool(processes=10) as pool:
        for i, r in enumerate(pool.imap_unordered(one_cloud, jobs), 1):
            rows.append(r)
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} clouds ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows).sort_values(["topology", "clustering", "seed"])
    D.to_csv(OUT / "b4_topo_null.csv", index=False)

    print("\n--- how much did the threshold move? ---")
    print(f"  median 12-draw threshold {D.thr12.median():.4f}   "
          f"median 500-draw threshold {D.thr500.median():.4f}   "
          f"ratio {D.thr12.median()/D.thr500.median():.2f}x")
    rel = ((D.thr_hi - D.thr_lo) / D.thr500).median()
    print(f"  bootstrap width of the 500-draw threshold: {100*rel:.1f}% of its value")
    dis = int(((D.H1_max > D.thr12) != (D.H1_max > D.thr500)).sum())
    print(f"  clouds whose verdict FLIPS between the 12-draw and 500-draw thresholds: "
          f"{dis} / {len(D)}")

    print("\n--- realised error rates of the pre-specified rules ---")
    print(f"  {'cell':<20}{'n':>4}{'type-I (no-loop)':>20}{'power (loop)':>16}")
    out = []
    for clus in ("low", "high"):
        for rule in ("sig_max", "sig_joint"):
            a = D[(D.topology == "noloop") & (D.clustering == clus)][rule]
            b = D[(D.topology == "loop") & (D.clustering == clus)][rule]
            out.append({"clustering": clus, "rule": rule, "n_per_cell": len(a),
                        "type_I": float(a.mean()), "power": float(b.mean())})
            print(f"  {clus + ' / ' + rule.replace('sig_', ''):<20}{len(a):>4}"
                  f"{a.mean():>19.1%}{b.mean():>16.1%}")
    E = pd.DataFrame(out)
    E.to_csv(OUT / "b4_error_rates.csv", index=False)

    print("\n--- dominance, the second half of the joint rule ---")
    for t in ("loop", "noloop"):
        for c in ("low", "high"):
            g = D[(D.topology == t) & (D.clustering == c)].dom_ratio
            print(f"  {t:7s} {c:5s}  dominance median {g.median():.3f} "
                  f"[{g.quantile(.25):.3f}, {g.quantile(.75):.3f}]")
    print(f"\nwrote b4_topo_null.csv, b4_error_rates.csv   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
