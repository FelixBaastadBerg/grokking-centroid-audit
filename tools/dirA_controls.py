"""
DIR A controls. A1 showed the memorize->generalize path stays at ~100% train accuracy while a
symmetry-equivalent path collapses to ~60%. Two things could still explain that away.

C1  TRAJECTORY SMOOTHNESS. W_mem and W_gen sit on one optimization trajectory. Maybe ANY two
    checkpoints are linearly connected, and grokking has nothing to do with it. So: regress
    barrier on separation over ALL checkpoint pairs within a run, and ask whether the
    memorize->generalize pair is anomalously low FOR ITS DISTANCE. If it sits on the same
    curve as everything else, A1 is about trajectories, not about the loss landscape.

C2  CROSS-SEED CONNECTIVITY. This one cannot be a trajectory artifact at all, because the two
    endpoints come from independent runs. Are two INDEPENDENT generalizing solutions linearly
    connected to each other, and are two independent MEMORIZING solutions not? That is the
    actual "does the generalizing basin become one connected object" question.
    Seeds differ in their data split, so everything is evaluated on the FULL p^2 dataset,
    which is common to both. Hidden units are Hungarian-aligned first; the permutation null
    is retained as the floor.
"""
from __future__ import annotations
import os, sys, warnings, itertools
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from scipy.optimize import linear_sum_assignment
from analysis.lsrun import LandscapeRun
from analysis.landscape import barrier_1d
from tools.dirA_barriers import permute_hidden

OUT = REPO / "RESULTS"


def align(ra, ia, rb, ib) -> np.ndarray:
    A0, A1_, _ = ra.W[0][ia], ra.W[1][ia], ra.W[2][ia]
    B0, B1_, B2 = rb.W[0][ib].copy(), rb.W[1][ib].copy(), rb.W[2][ib].copy()
    r, c = linear_sum_assignment(-(A0 @ B0.T)); P1 = c[np.argsort(r)]
    B0, B1_ = B0[P1], B1_[:, P1]
    r2, c2 = linear_sum_assignment(-(A1_ @ B1_.T)); P2 = c2[np.argsort(r2)]
    B1_, B2 = B1_[P2], B2[:, P2]
    return np.concatenate([B0.ravel(), B1_.ravel(), B2.ravel()]).astype(np.float64)


def full_eval(r: LandscapeRun, v: np.ndarray) -> tuple[float, float]:
    """Loss and accuracy on ALL p^2 examples -- the one dataset two seeds share."""
    r._load(v); r.model.eval()
    X = torch.cat([r.Xtr, r.Xte]); Y = torch.cat([r.Ytr, r.Yte])
    with torch.no_grad():
        o = r.model(X)
        return (float(r.loss_fn(o, Y, dtype=torch.float64).item()),
                float((o.argmax(1) == Y).float().mean()))


def full_path(r, va, vb, n=31):
    L, A = [], []
    for a in np.linspace(0, 1, n):
        l, ac = full_eval(r, (1 - a) * va + a * vb)
        L.append(l); A.append(ac)
    return np.array(L), np.array(A)


# ------------------------------------------------------------------ C1
def c1(r: LandscapeRun, n_pairs=14, npts=21, seed=0):
    im, tg, ig = r.t_memorized(), r.t_grok(), len(r.epochs) - 1
    if im < 0 or tg < 0 or im >= tg:
        return None
    rng = np.random.default_rng(seed)
    rows = []
    lo = list(range(im, tg))                     # both endpoints inside the plateau
    for _ in range(n_pairs):
        if len(lo) < 2:
            break
        i, j = sorted(rng.choice(lo, 2, replace=False))
        P = r.path_losses(r.flat(i), r.flat(j), n=npts)
        rows.append({"kind": "plateau-plateau", "dist": float(np.linalg.norm(r.flat(j) - r.flat(i))),
                     "barrier": barrier_1d(P["train_loss"]), "min_tracc": float(P["train_acc"].min())})
    P = r.path_losses(r.flat(im), r.flat(ig), n=npts)
    rows.append({"kind": "mem-gen", "dist": float(np.linalg.norm(r.flat(ig) - r.flat(im))),
                 "barrier": barrier_1d(P["train_loss"]), "min_tracc": float(P["train_acc"].min())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ C2
def c2(ra: LandscapeRun, rb: LandscapeRun, seed=0):
    ima, tga, iga = ra.t_memorized(), ra.t_grok(), len(ra.epochs) - 1
    imb, tgb, igb = rb.t_memorized(), rb.t_grok(), len(rb.epochs) - 1
    if min(tga, tgb) < 0:
        return None
    rng = np.random.default_rng(seed)
    out = {}
    for tag, (ia, ib) in (("mem", (ima, imb)), ("gen", (iga, igb))):
        va = ra.flat(ia)
        vb = align(ra, ia, rb, ib)
        L, A = full_path(ra, va, vb)
        vp = permute_hidden(rb, ib, rng)
        Ln, An = full_path(ra, va, vp)
        out[f"{tag}_barrier"] = barrier_1d(L)
        out[f"{tag}_null"] = barrier_1d(Ln)
        out[f"{tag}_minacc"] = float(A.min())
        out[f"{tag}_null_minacc"] = float(An.min())
        out[f"{tag}_endacc"] = float(min(A[0], A[-1]))
        out[f"{tag}_dist"] = float(np.linalg.norm(vb - va))
    return out


if __name__ == "__main__":
    inv = pd.read_csv(OUT / "_inventory.csv")
    G = inv[(inv.arch == "mlp") & (inv.test >= 0.9)]

    print("=== C1: is mem->gen anomalous FOR ITS DISTANCE, vs plateau-plateau pairs? ===\n")
    allc = []
    for _, rec in G.head(18).iterrows():
        try:
            t = c1(LandscapeRun(REPO / "runs_topo" / rec["dir"]))
        except Exception:
            continue
        if t is None:
            continue
        t["dir"] = rec["dir"]; t["task"] = rec["task"]; t["p"] = rec["p"]
        allc.append(t)
    C = pd.concat(allc, ignore_index=True); C.to_csv(OUT / "dirA_c1.csv", index=False)
    pp, mg = C[C.kind == "plateau-plateau"], C[C.kind == "mem-gen"]
    print(f"  plateau-plateau pairs n={len(pp)}   mem-gen pairs n={len(mg)}")
    print(f"  median distance  plateau-plateau {pp.dist.median():9.1f}   mem-gen {mg.dist.median():9.1f}")
    print(f"  median barrier   plateau-plateau {pp.barrier.median():9.4f}   mem-gen {mg.barrier.median():9.4f}")
    print(f"  median min_tracc plateau-plateau {pp.min_tracc.median():9.4f}   mem-gen {mg.min_tracc.median():9.4f}")
    # distance-matched comparison: plateau pairs in the top distance quartile
    hi = pp[pp.dist >= pp.dist.quantile(0.75)]
    print(f"\n  distance-matched (top-quartile plateau pairs, n={len(hi)}, median dist {hi.dist.median():.1f}):")
    print(f"    barrier   {hi.barrier.median():.4f}  vs mem-gen {mg.barrier.median():.4f}")
    print(f"    min_tracc {hi.min_tracc.median():.4f}  vs mem-gen {mg.min_tracc.median():.4f}")

    print("\n=== C2: cross-seed connectivity, memorizing vs generalizing ===\n")
    rows = []
    for (task, p, tf), grp in G.groupby(["task", "p", "tf"]):
        if len(grp) < 2:
            continue
        for a, b in itertools.combinations(grp.to_dict("records"), 2):
            if a["seed"] == b["seed"]:
                continue
            try:
                ra = LandscapeRun(REPO / "runs_topo" / a["dir"])
                rb = LandscapeRun(REPO / "runs_topo" / b["dir"])
                o = c2(ra, rb)
            except Exception as e:
                print(f"  skip: {type(e).__name__} {e}"); continue
            if o is None:
                continue
            o.update({"task": task, "p": p, "tf": tf, "sa": a["seed"], "sb": b["seed"]})
            rows.append(o)
            print(f"  {task:4s}/p{p:<3d}/tf{tf}/s{a['seed']}-s{b['seed']}  "
                  f"MEM bar={o['mem_barrier']:6.3f} null={o['mem_null']:6.3f} minacc={o['mem_minacc']:.3f} "
                  f"| GEN bar={o['gen_barrier']:6.3f} null={o['gen_null']:6.3f} minacc={o['gen_minacc']:.3f}",
                  flush=True)
    S = pd.DataFrame(rows); S.to_csv(OUT / "dirA_c2.csv", index=False)
    if len(S):
        print(f"\n=== C2 SUMMARY (n={len(S)} seed pairs) ===")
        print(S.groupby(["task", "p"]).agg(
            n=("mem_barrier", "size"),
            mem_bar=("mem_barrier", "median"), mem_null=("mem_null", "median"),
            mem_minacc=("mem_minacc", "median"), mem_endacc=("mem_endacc", "median"),
            gen_bar=("gen_barrier", "median"), gen_null=("gen_null", "median"),
            gen_minacc=("gen_minacc", "median"), gen_endacc=("gen_endacc", "median")).round(3).to_string())
