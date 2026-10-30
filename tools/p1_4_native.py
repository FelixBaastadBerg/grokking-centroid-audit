"""
P1-4, Fix 1 -- the alpha sweep at the NATIVE operating point of the large-p networks.

The external-validity numbers so far come from two clouds: a fixed 400-point budget (which at
p = 97 leaves ~4 points per class, an artifact of the budget) and a corpus-matched cloud at
m = 20. A reviewer will reasonably ask why a network whose cloud holds 58 points per class was
measured at 20. This adds the native operating point so all three are reported side by side.

Two native measurements, because "native" is ambiguous and the two answer different questions:

  native-full   every training point, no subsampling at all: n = p^2*tf, m = n/p. This is
                literally the network's own cloud. Persistent homology is not computed here --
                Vietoris-Rips H_1 on 5.6k points is out of reach -- so the topological columns
                are absent by construction, not by omission.

  native-m      classes subsampled so the cloud is ~400 points, but every point of each retained
                class kept, so m is native while n matches every other measurement in the paper.
                This is the one that isolates m from cloud size, which is what the scale-locality
                claim is about, and it carries the full estimator set.

Prediction under scale-locality: the confound scales with 1/m, so PH-dim's fold at native m should
sit BELOW its m = 20 value while remaining far above 1, and TwoNN should stay near 1 throughout.

Usage:
    python tools/p1_4_native.py saved     # measure the stored 4000-point clouds (fast)
    python tools/p1_4_native.py full      # re-extract uncapped activations, then measure
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from analysis import geom
from tools.linchpin import h1_stats
from tools.e1_centroid import decompose, recompose, ALPHAS

OUT = REPO / "RESULTS"
ACTS = OUT / "p1_4_acts"
TARGET_N = 400          # the cloud size every other measurement in the paper uses


def native_m_subset(H, y, target_n=TARGET_N, seed=0):
    """Keep every point of a random subset of classes: m native, n comparable."""
    g = np.random.default_rng(seed)
    cls = g.permutation(np.unique(y))
    keep, n = [], 0
    for c in cls:
        idx = np.flatnonzero(y == c)
        if n and n + len(idx) > target_n * 1.15:
            break
        keep.append(idx); n += len(idx)
        if n >= target_n * 0.85:
            break
    idx = np.concatenate(keep)
    return H[idx], y[idx]


def sweep(H, y, tag, task, p, seed, do_h1=True):
    m = float(np.median(np.unique(y, return_counts=True)[1]))
    mu, bet, wit = decompose(H, y)
    rows = []
    for a in ALPHAS:
        Z = recompose(mu, bet, wit, a, 1.0)
        pdim, _ = geom.ph_dim(Z); cd, _ = geom.correlation_dim(Z)
        r = {"arch": "mlp", "task": task, "p": p, "seed": seed, "norm": "trace", "sweep": "alpha",
             "alpha": a, "beta": 1.0, "ph_dim": pdim, "corr_dim": cd, "twonn": geom.twonn(Z),
             "mle_k10": geom.mle_id(Z, 10), "pca_pr": geom.pca_participation_ratio(Z),
             "eff_rank": geom.effective_rank(Z), "n_points": len(Z),
             "points_per_class": m, "subsample": tag, "source": "P1-4"}
        if do_h1:
            h = h1_stats(Z); r["H1_max"], r["H1_total"] = h["H1_max"], h["H1_total"]
        rows.append(r)
    return rows


def clouds_from_saved():
    for f in sorted(ACTS.glob("*.npz")):
        stem = f.stem                      # add_p97_seed0
        task, ps, ss = stem.split("_")
        d = np.load(f)
        yield task, int(ps[1:]), int(ss[4:]), d["H"], d["y"]


def clouds_from_full():
    """Re-extract uncapped final activations. The seeds and split are fixed, so these are the
    same networks measured in the matched arm -- nothing new is being trained, the activations
    were simply never kept at full resolution."""
    from tools.p1_4_scale import train
    for task, p in [("add", 97), ("add", 113)]:
        for seed in range(3):
            _, snaps = train(task, p, seed, max_points=10 ** 9)
            H, y = snaps["final"]
            np.savez_compressed(ACTS / f"{task}_p{p}_seed{seed}_full.npz", H=H, y=y)
            yield task, p, seed, H, y


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "saved"
    src = clouds_from_saved if mode == "saved" else clouds_from_full
    t0, rows = time.time(), []
    for task, p, seed, H, y in src():
        Hm, ym = native_m_subset(H, y, seed=seed)
        rows += sweep(Hm, ym, "native_m", task, p, seed, do_h1=True)
        # the whole cloud, no subsampling; H_1 is out of reach at this size
        rows += sweep(H, y, "native_full", task, p, seed, do_h1=False)
        mm = np.median(np.unique(ym, return_counts=True)[1])
        print(f"  {task}/p{p}/seed{seed}: native_m n={len(Hm)} m={mm:.0f}   "
              f"native_full n={len(H)} m={len(H)/len(np.unique(y)):.0f}   "
              f"({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"p1_4_centroid_native_{mode}.csv", index=False)
    print(f"\nwrote p1_4_centroid_native_{mode}.csv ({time.time()-t0:.0f}s)")
