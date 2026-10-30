"""
DIR A experiment 3 -- WHEN does cross-run connectivity appear?

C2 compared two endpoints (memorizing vs generalizing). This resolves it in time: for each
checkpoint t, take run A and run B at the SAME t, Hungarian-align B's hidden units to A's,
and measure the accuracy drop along the straight line between them. That drop is a property
of the pair of independent runs at time t -- it cannot come from trajectory continuity,
because A and B share no trajectory.

If the drop collapses at t_grok, then the topological event is the merging of the two runs'
solutions into one connected component of the low-loss set, and it happens exactly when
the networks generalize.

Everything is evaluated on the full p^2 dataset, the one set both runs share.
"""
from __future__ import annotations
import os, sys, warnings, itertools
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from analysis.lsrun import LandscapeRun
from tools.dirA_controls import align, full_path
from tools.dirA_barriers import permute_hidden

OUT = REPO / "RESULTS"


def timing_pair(ra: LandscapeRun, rb: LandscapeRun, n_t=26, npts=11, seed=0):
    n = min(len(ra.epochs), len(rb.epochs))
    tga, tgb = ra.t_grok(), rb.t_grok()
    if tga < 0 or tgb < 0:
        return None
    # Uniform sampling over 301 checkpoints puts only ~2 points before a t_grok at index 22.
    # Sample log-spaced (resolves the early phase and the whole plateau) PLUS a dense window
    # around t_grok, so an event at the transition cannot be missed by the grid.
    lg = np.unique(np.round(np.geomspace(1, n - 1, n_t)).astype(int))
    win = np.arange(max(0, tga - 6), min(n, tga + 7))
    ts = np.unique(np.concatenate([[0], lg, win, [n - 1]])).astype(int)
    rng = np.random.default_rng(seed)
    rows = []
    for t in ts:
        va = ra.flat(t)
        vb = align(ra, t, rb, t)
        L, A = full_path(ra, va, vb, n=npts)
        vp = permute_hidden(rb, t, rng)
        Ln, An = full_path(ra, va, vp, n=npts)
        rows.append({"t": int(t), "epoch": int(ra.epochs[t]),
                     "endacc": float(min(A[0], A[-1])), "minacc": float(A.min()),
                     "acc_drop": float(min(A[0], A[-1]) - A.min()),
                     "null_drop": float(min(An[0], An[-1]) - An.min()),
                     "acc_a": float(ra.curves.test_acc.values[t]),
                     "acc_b": float(rb.curves.test_acc.values[t])})
    d = pd.DataFrame(rows)
    d["t_grok_a"], d["t_grok_b"] = tga, tgb
    d["phase"] = (d.t - tga) / max(n - 1 - tga, 1)
    return d


if __name__ == "__main__":
    inv = pd.read_csv(OUT / "_inventory.csv")
    G = inv[(inv.arch == "mlp") & (inv.test >= 0.9)]
    picks = []
    for (task, p, tf), grp in G.groupby(["task", "p", "tf"]):
        grp = grp.drop_duplicates("seed")
        if len(grp) >= 2:
            recs = grp.to_dict("records")
            picks.append((task, p, tf, recs[0], recs[1]))
    print(f"=== A3: connectivity over training, {len(picks)} independent seed pairs ===\n")
    alld = []
    for task, p, tf, a, b in picks:
        try:
            ra = LandscapeRun(REPO / "runs_topo" / a["dir"])
            rb = LandscapeRun(REPO / "runs_topo" / b["dir"])
            d = timing_pair(ra, rb)
        except Exception as e:
            print(f"  skip {task}/p{p}/tf{tf}: {type(e).__name__} {e}"); continue
        if d is None:
            continue
        d["task"], d["p"], d["tf"] = task, p, tf
        alld.append(d)
        pre = d[d.t < d.t_grok_a]; post = d[d.t > d.t_grok_a]
        print(f"  {task:4s}/p{p:<3d}/tf{tf}  t_grok={d.t_grok_a.iloc[0]:3d}  "
              f"drop PRE={pre['acc_drop'].median():.4f}  POST={post['acc_drop'].median():.4f}  "
              f"null={d['null_drop'].median():.4f}", flush=True)
    D = pd.concat(alld, ignore_index=True)
    D.to_csv(OUT / "dirA_timing.csv", index=False)
    print(f"\n=== SUMMARY (n={len(D)} checkpoint-pairs across {len(alld)} run pairs) ===")
    D["side"] = np.where(D.t < D.t_grok_a, "pre-grok", np.where(D.t > D.t_grok_a, "post-grok", "at-grok"))
    print(D.groupby(["task", "p", "side"]).agg(
        n=("acc_drop", "size"), acc_drop=("acc_drop", "median"), null_drop=("null_drop", "median"),
        endacc=("endacc", "median")).round(4).to_string())
