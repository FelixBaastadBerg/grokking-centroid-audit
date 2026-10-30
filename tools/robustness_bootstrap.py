"""
Bootstrap CIs on the headline numbers, and verification that real grokked networks actually sit
at the operating point the intervention assumes.

Three things:
  B1  operating point -- label-silhouette of activations across the WHOLE corpus, at t_grok and
      at convergence, with a bootstrap CI over runs. The intervention's relevance depends on it.
  B2  bootstrap CIs on the Arm A fold changes (resampling seeds and points).
  B3  bootstrap CI on rho(ph_dim, gap_ratio) = 0.991, the C2 redundancy claim, over runs.
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
from analysis.lsrun import LandscapeRun
from analysis import geom
from tools.intervention import make_cloud, clusteredness, estimators, EST

OUT = REPO / "RESULTS"


def boot_ci(vals, f=np.median, n=2000, seed=0):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 3:
        return (np.nan, np.nan, np.nan)
    g = np.random.default_rng(seed)
    bs = [f(g.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(f(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if __name__ == "__main__":
    t0 = time.time()
    pd.set_option("display.width", 220)

    # ---------------------------------------------------------------- B1 operating point
    print("=== B1: do real grokked networks reach the operating point? ===")
    print("    label-silhouette of pre-logit activations, TRUE class labels\n")
    inv = pd.read_csv(OUT / "_inventory.csv")
    sel = inv[(inv.arch == "mlp")].copy()
    rows = []
    for i, rec in enumerate(sel.to_dict("records")):
        try:
            r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
        except Exception:
            continue
        n = len(r.epochs)
        y = r.Ytr.numpy()
        sub = np.sort(np.random.default_rng(0).choice(len(y), min(500, len(y)), replace=False))
        ys = y[sub]
        if len(np.unique(ys)) < 2:
            continue
        tg = r.t_grok()
        out = {}
        for tag, t in (("init", 0), ("t_grok", tg if tg >= 0 else n // 2), ("final", n - 1)):
            r._load(r.flat(int(t))); r.model.eval()
            with torch.no_grad():
                r.model(r.Xtr, keep_activations=True)
                H = r.model.activations[-2].numpy()[sub]
            out[f"sil_{tag}"] = float(silhouette_score(H, ys))
        out.update({"grokked": rec["test"] >= 0.9, "task": rec["task"], "p": rec["p"],
                    "tf": rec["tf"], "seed": rec["seed"], "dir": rec["dir"]})
        rows.append(out)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(sel)} runs ({time.time()-t0:.0f}s)", flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "operating_point.csv", index=False)
    print(f"\n  n={len(S)} runs ({S.grokked.sum()} grokked)")
    for grok in (True, False):
        s = S[S.grokked == grok]
        if not len(s):
            continue
        print(f"\n  {'GROKKED' if grok else 'NOT GROKKED'} (n={len(s)}):")
        for tag in ("init", "t_grok", "final"):
            m, lo, hi = boot_ci(s[f"sil_{tag}"].values)
            print(f"    silhouette @ {tag:8s} median {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    g = S[S.grokked]
    print(f"\n  grokked runs with final silhouette >= 0.60: {(g.sil_final>=0.60).sum()}/{len(g)}")
    print(f"  grokked runs with final silhouette >= 0.50: {(g.sil_final>=0.50).sum()}/{len(g)}")
    print("\n  by group:")
    print(S[S.grokked].groupby(["task", "p"]).sil_final.agg(["size", "median", "min", "max"])
          .round(3).to_string())

    # ---------------------------------------------------------------- B2 Arm A CIs
    print("\n\n=== B2: bootstrap CIs on the Arm A fold changes (true d fixed at 5) ===\n")
    recs = []
    for sigma in (1.5, 0.6, 0.25, 0.1, 0.03, 0.01):
        for seed in range(8):
            X = make_cloud(5, 8, sigma, seed=seed)
            recs.append({"sigma": sigma, "seed": seed, **estimators(X)})
    A = pd.DataFrame(recs)
    A.to_csv(OUT / "robust_armA_boot.csv", index=False)
    gsd = np.random.default_rng(0)
    print(f"  {'estimator':16s} {'fold':>7s}  {'95% CI':>18s}")
    for e in EST:
        folds = []
        for _ in range(2000):
            take = gsd.choice(8, 8, replace=True)
            sub = A[A.seed.isin(take)]
            m = sub.groupby("sigma")[e].mean()
            if len(m) > 1 and abs(m.min()) > 1e-9:
                folds.append(m.max() / abs(m.min()))
        m0 = A.groupby("sigma")[e].mean()
        pt = m0.max() / max(abs(m0.min()), 1e-9)
        print(f"  {e:16s} {pt:7.1f}  [{np.percentile(folds,2.5):7.1f}, {np.percentile(folds,97.5):7.1f}]")

    # ---------------------------------------------------------------- B3 redundancy CI
    print("\n\n=== B3: bootstrap CI on the C2 redundancy claim ===\n")
    D = pd.read_csv(OUT / "comeasure.csv")
    pairs = [("ph_dim", "gap_ratio"), ("ph_dim", "nn1_ratio"), ("eff_rank", "gap_ratio"),
             ("twonn", "gap_ratio"), ("ph_dim", "twonn")]
    for a, b in pairs:
        per = []
        for d, gg in D.groupby("dir"):
            s = gg[[a, b]].dropna()
            if len(s) > 8 and s[a].std() > 0 and s[b].std() > 0:
                per.append(spearmanr(s[a], s[b]).statistic)
        m, lo, hi = boot_ci(per)
        print(f"  rho({a:10s}, {b:10s})  median {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={len(per)}")
    print(f"\ndone in {time.time()-t0:.0f}s")
