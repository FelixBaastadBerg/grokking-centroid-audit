"""
DIR A experiment 1 -- is the generalizing solution in the SAME connected component of the
low-train-loss set as the memorizing solution it came from?

The null matters more than the statistic here. A path from a low-loss point to a RANDOM far
point rises monotonically, so "peak minus the higher endpoint" is identically zero for it --
such a null can never fire and proves nothing. The correct null uses the network's own exact
symmetry: permuting hidden units leaves the loss unchanged to machine precision but lands on
a different point of weight space. So

    signal : barrier along  W_mem -> W_gen
    null   : barrier along  W_mem -> pi(W_gen)      for random hidden-unit permutations pi

have endpoints of IDENTICAL loss, and any difference between them is about connectivity, not
about where the endpoints sit. This is the standard mode-connectivity control, and
`--gate` checks the permutation invariance numerically before any of it is used.

Barriers are reported relative to the higher endpoint (peak - max(ends)); train accuracy
along the path is tracked separately because a loss barrier with no accuracy drop means
something different from one with a drop.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from analysis.lsrun import LandscapeRun
from analysis.landscape import barrier_1d

OUT = REPO / "RESULTS"


def permute_hidden(r: LandscapeRun, i: int, rng) -> np.ndarray:
    """Random hidden-unit permutation of checkpoint i. Exactly loss-preserving."""
    W0, W1, W2 = r.W[0][i].copy(), r.W[1][i].copy(), r.W[2][i].copy()
    q1 = rng.permutation(W0.shape[0])
    q2 = rng.permutation(W1.shape[0])
    W0 = W0[q1]
    W1 = W1[:, q1][q2]
    W2 = W2[:, q2]
    return np.concatenate([W0.ravel(), W1.ravel(), W2.ravel()]).astype(np.float64)


def run_arm(r: LandscapeRun, npts=31, n_null=4, seed=0):
    ig = len(r.epochs) - 1
    im, tg = r.t_memorized(), r.t_grok()
    if im < 0 or tg < 0 or im >= tg:
        return None
    va, vb = r.flat(im), r.flat(ig)
    P = r.path_losses(va, vb, n=npts)
    rng = np.random.default_rng(seed)
    nb, nacc, ndist = [], [], []
    for _ in range(n_null):
        vp = permute_hidden(r, ig, rng)
        Q = r.path_losses(va, vp, n=npts)
        nb.append(barrier_1d(Q["train_loss"])); nacc.append(float(Q["train_acc"].min()))
        ndist.append(float(np.linalg.norm(vp - va)))
    return {"barrier": barrier_1d(P["train_loss"]),
            "null_barrier": float(np.mean(nb)), "null_sd": float(np.std(nb)),
            "min_tracc": float(P["train_acc"].min()), "null_min_tracc": float(np.mean(nacc)),
            "min_teacc": float(P["test_acc"].min()),
            "dist": float(np.linalg.norm(vb - va)), "null_dist": float(np.mean(ndist)),
            "peak_test_loss": float(P["test_loss"].max()),
            "end_test_loss": float(P["test_loss"][-1])}


if __name__ == "__main__":
    inv = pd.read_csv(OUT / "_inventory.csv")
    inv = inv[(inv.arch == "mlp") & (inv.test >= 0.9)]

    print("=== GATE: hidden-unit permutation must preserve loss exactly ===")
    rg = LandscapeRun(REPO / "runs_topo" / inv.iloc[0]["dir"])
    i = len(rg.epochs) - 1
    a = rg.evaluate(rg.flat(i)); b = rg.evaluate(permute_hidden(rg, i, np.random.default_rng(0)))
    print(f"  train_loss {a['train_loss']:.10f} vs {b['train_loss']:.10f}   "
          f"delta {abs(a['train_loss']-b['train_loss']):.2e}")
    print(f"  test_acc   {a['test_acc']:.6f} vs {b['test_acc']:.6f}")
    assert abs(a["train_loss"] - b["train_loss"]) < 1e-9, "permutation is not loss-preserving"
    print("  GATE PASSED\n")

    print(f"=== A1: memorize -> generalize, vs permutation null, {len(inv)} grokked MLP runs ===\n")
    rows = []
    for _, rec in inv.iterrows():
        try:
            a = run_arm(LandscapeRun(REPO / "runs_topo" / rec["dir"]))
        except Exception as e:
            print(f"  skip {rec['dir'][:44]}: {type(e).__name__}"); continue
        if a is None:
            continue
        a.update({"task": rec["task"], "p": rec["p"], "tf": rec["tf"], "seed": rec["seed"],
                  "test": rec["test"], "dir": rec["dir"]})
        a["ratio"] = a["barrier"] / max(a["null_barrier"], 1e-9)
        rows.append(a)
        print(f"  {rec['task']:4s}/p{rec['p']:<3d}/tf{rec['tf']}/s{rec['seed']}  "
              f"barrier={a['barrier']:7.3f}  null={a['null_barrier']:7.3f}+-{a['null_sd']:.3f}  "
              f"ratio={a['ratio']:6.3f}  min_tracc={a['min_tracc']:.3f} (null {a['null_min_tracc']:.3f})",
              flush=True)
    D = pd.DataFrame(rows); D.to_csv(OUT / "dirA_within.csv", index=False)
    print(f"\n=== SUMMARY (n={len(D)}) ===")
    print(D.groupby(["task", "p"]).agg(
        n=("barrier", "size"), barrier=("barrier", "median"), null=("null_barrier", "median"),
        ratio=("ratio", "median"), min_tracc=("min_tracc", "median"),
        null_tracc=("null_min_tracc", "median"), dist=("dist", "median"),
        null_dist=("null_dist", "median")).round(3).to_string())
    print(f"\n  runs where the real barrier is BELOW its own permutation null: "
          f"{(D.barrier < D.null_barrier).sum()}/{len(D)}")
