"""
Outstanding run 2 (synthetic -> real bridge) + the cross-architecture / cross-group stress test.

BRIDGE: the intervention says ph_dim / corr_dim / spec_entropy / Monod-H1 are confounded by
clustering and TwoNN / MLE-ID are not. If that matters for practice, then on REAL grokked
networks the two families must tell DIFFERENT stories about the transition. Quantified as:
  - the size of each estimator's change through the transition (pre-grok -> post-grok)
  - the correlation between the confounded and reliable families within runs
  - what each family says about "did the dimension collapse?"

CROSS-ARCH / CROSS-GROUP: the same measurements on TRANSFORMER activations (pre-logit encoder
output at the last position) and on non-abelian S_4, so the triage is not an MLP/abelian artifact.
"""
from __future__ import annotations
import os, sys, warnings, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from analysis import geom
from analysis.tfrun import TransformerRun
from tools.linchpin import h1_stats

OUT = REPO / "RESULTS"
CONF = ["ph_dim", "corr_dim", "spec_entropy", "emb_H1_max", "emb_H1_total"]
RELI = ["twonn", "mle_id_k10"]


def spec_entropy(X):
    Xc = np.asarray(X, float) - np.mean(X, 0)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    s = s[s > 0]
    p = s / s.sum()
    return float(-(p * np.log(p)).sum())


def measure_tf(d: Path, n_ck=40):
    r = TransformerRun(d)
    n = len(r.epochs)
    tr = np.load(d / "trajectory.npz")
    stride = max(1, n // n_ck)
    ts = np.arange(0, n, stride)
    y = r.Ytr.numpy()
    sub = np.sort(np.random.default_rng(0).choice(len(y), min(500, len(y)), replace=False))
    ys = y[sub]
    rows = []
    for t in ts:
        H = r.activations(int(t), r.Xtr)[sub]
        b = geom.battery(H, y=ys, cheap=True)
        b["spec_entropy"] = spec_entropy(H)
        b["sil_true"] = float(silhouette_score(H, ys)) if len(np.unique(ys)) > 1 else np.nan
        E = tr["W0"][int(t)][:r.p]
        b.update({f"emb_{k}": v for k, v in h1_stats(E).items()})
        b.update({"t": int(t), "epoch": int(r.epochs[int(t)]), "t_grok": r.t_grok(),
                  "train_acc": float(r.curves.train_acc.values[int(t)]),
                  "test_acc": float(r.curves.test_acc.values[int(t)])})
        rows.append(b)
    df = pd.DataFrame(rows)
    for k, v in (("arch", "transformer"), ("task", r.meta["task"]), ("p", r.p),
                 ("seed", r.meta["seed"]), ("final_test", float(r.curves.test_acc.iloc[-1])),
                 ("dir", d.name)):
        df[k] = v
    return df


if __name__ == "__main__":
    t0 = time.time()
    pd.set_option("display.width", 230)

    runs = [d for d in sorted((REPO / "runs_topo").glob("arch-transformer*ss777*"))
            if (d / "curves.csv").exists() and (d / "full_traj.npz").exists()]
    print(f"=== transformer arm: {len(runs)} runs ===", flush=True)
    alld = []
    for d in runs:
        try:
            alld.append(measure_tf(d))
            x = alld[-1]
            print(f"  {x.task.iloc[0]}/p{x.p.iloc[0]}/s{x.seed.iloc[0]} "
                  f"final={x.final_test.iloc[0]:.3f} sil_final={x.sil_true.iloc[-1]:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  SKIP {d.name[:44]}: {type(e).__name__} {e}", flush=True)
    T = pd.concat(alld, ignore_index=True) if alld else pd.DataFrame()
    if len(T):
        T.to_csv(OUT / "bridge_tf.csv", index=False)

    M = pd.read_csv(OUT / "comeasure.csv")
    if "t_grok" not in M.columns:       # comeasure stored raw curves; derive the anchor here
        tg = {}
        for d, g in M.groupby("dir"):
            g = g.sort_values("epoch")
            fin = g.test_acc.values[-3:].mean()
            tg[d] = int(g[g.test_acc >= 0.5 * fin].t.min()) if fin >= 0.5 else -1
        M["t_grok"] = M.dir.map(tg)

    print("\n\n=== BRIDGE: do confounded and reliable estimators tell the same story? ===")
    print("    change through the transition, as a fraction of the estimator's own start value\n")

    def story(D, label, embp="emb_"):
        rows = []
        for sig in ["ph_dim", "corr_dim", "eff_rank", "pca_pr", "twonn", "mle_id_k10",
                    f"{embp}H1_max", f"{embp}H1_total"]:
            if sig not in D:
                continue
            ch, pre, post = [], [], []
            for d, g in D.groupby("dir"):
                g = g.sort_values("epoch")
                tg = g['t_grok'].iloc[0]
                a = g[g.t < tg][sig].dropna()
                b = g[g.t > tg][sig].dropna()
                if len(a) < 3 or len(b) < 3 or a.median() == 0:
                    continue
                pre.append(a.median()); post.append(b.median())
                ch.append((b.median() - a.median()) / abs(a.median()))
            if ch:
                rows.append({"estimator": sig, "n": len(ch), "pre": np.median(pre),
                             "post": np.median(post), "rel_change": np.median(ch)})
        R = pd.DataFrame(rows).round(3)
        print(f"  --- {label} ---")
        print(R.to_string(index=False))
        return R

    story(M, "MLP (n=%d runs)" % M.dir.nunique())
    if len(T):
        print()
        story(T, "TRANSFORMER (n=%d runs)" % T.dir.nunique())

    print("\n\n=== CROSS-ARCH / CROSS-GROUP: redundancy with clusteredness ===\n")
    for label, D in (("MLP", M), ("TRANSFORMER", T)):
        if not len(D):
            continue
        print(f"  --- {label} ---")
        for a in ["ph_dim", "eff_rank", "twonn", "mle_id_k10"]:
            if a not in D:
                continue
            per = [spearmanr(g[a], g["gap_ratio"]).statistic for _, g in D.groupby("dir")
                   if g[a].std() > 0 and g["gap_ratio"].std() > 0]
            print(f"    rho({a:11s}, gap_ratio) median {np.median(per):+.3f}  n={len(per)}")
        print()

    if len(T):
        print("=== CROSS-GROUP within transformers (abelian Z_23 vs non-abelian S_4) ===\n")
        for task, g in T.groupby("task"):
            print(f"  {task} (n={g.dir.nunique()} runs): sil_final "
                  f"{g.groupby('dir').sil_true.last().median():.3f}, "
                  f"ph_dim {g.groupby('dir').ph_dim.first().median():.1f} -> "
                  f"{g.groupby('dir').ph_dim.last().median():.1f}, "
                  f"twonn {g.groupby('dir').twonn.first().median():.1f} -> "
                  f"{g.groupby('dir').twonn.last().median():.1f}")
    print(f"\ndone in {time.time()-t0:.0f}s")
