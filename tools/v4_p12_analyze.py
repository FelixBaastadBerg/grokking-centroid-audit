"""
P1.2 -- does within-class spectral compression scale with weight decay?

Z_23 addition, width 200, weight decay in {0, 0.003, 0.03, 0.1}, 3 seeds each (the 0.03 level is
the existing canonical cell). Per checkpoint on the fixed 400-point train subsample: within-class
and global effective rank, true-label silhouette, accuracies.

The dose-response question is asked two ways, because runs at different wd reach different
training states at different times:
  (a) terminal: relative within-class effective-rank change, per wd level;
  (b) matched training state: the same change measured from init to the first checkpoint with
      train accuracy 1.0 (memorisation), before generalisation confounds the comparison.

Output: RESULTS/v4_p12_series.csv, v4_p12_summary.txt
"""
from __future__ import annotations
import os, sys, json, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from tools.train_topology import build_task

OUT = REPO / "RESULTS"
RUNS = REPO / "runs_topo"
MAX_CK = 51
N_SUB = 400
LINES = []


def say(s=""):
    print(s, flush=True); LINES.append(s)


def spec_er(M):
    s = np.linalg.svd(M - M.mean(0), compute_uv=False) ** 2
    s = s[s > 1e-30]
    if not len(s):
        return np.nan
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def one_run(d, wd):
    meta = json.loads((d / "run_meta.json").read_text())
    tr = np.load(d / "trajectory.npz")
    curves = pd.read_csv(d / "curves.csv").set_index("epoch")
    pairs, y_all, n_sym, _ = build_task(meta["task"], meta["p"])
    N = len(y_all)
    X = np.zeros((N, 2 * n_sym), np.float32)
    X[np.arange(N), pairs[:, 0]] = 1.0
    X[np.arange(N), n_sym + pairs[:, 1]] = 1.0
    split_seed = meta.get("split_seed", meta["seed"])
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(split_seed)).numpy()
    tr_idx = perm[:int(meta["train_fraction"] * N)]
    sub = np.sort(np.random.default_rng(0).choice(len(tr_idx), min(N_SUB, len(tr_idx)),
                                                  replace=False))
    Xs, ys = X[tr_idx][sub], y_all[tr_idx][sub]
    masks = {c: ys == c for c in np.unique(ys)}
    epochs = tr["epochs"]
    idx = (np.unique(np.round(np.linspace(0, len(epochs) - 1, MAX_CK)).astype(int))
           if len(epochs) > MAX_CK else np.arange(len(epochs)))
    rows = []
    for i in idx:
        h = np.maximum(Xs @ tr["W0"][i].T, 0.0)
        H = np.maximum(h @ tr["W1"][i].T, 0.0)
        cen = np.zeros_like(H)
        for c, m in masks.items():
            cen[m] = H[m].mean(0)
        try:
            sil = float(silhouette_score(H, ys))
        except Exception:
            sil = np.nan
        ep = int(epochs[i])
        cv = curves.loc[ep] if ep in curves.index else None
        rows.append({"dir": d.name, "wd": wd, "seed": meta["seed"], "epoch": ep,
                     "train_acc": float(cv.train_acc) if cv is not None else np.nan,
                     "test_acc": float(cv.test_acc) if cv is not None else np.nan,
                     "train_loss": float(cv.train_loss) if cv is not None else np.nan,
                     "within_eff_rank": spec_er(H - cen), "global_eff_rank": spec_er(H),
                     "sil_true": sil})
    return rows


def main():
    cells = []
    for d in sorted(RUNS.iterdir()):
        n = d.name
        if not (d / "trajectory.npz").exists() or "task-add__p-23" not in n:
            continue
        if n.endswith("__wdsweep"):
            wd = float(n.split("wd-")[1].split("__")[0])
            cells.append((d, wd))
    # the wd = 0.03 level: canonical plain-suffix runs, seeds 0-2
    for s in (0, 1, 2):
        d = RUNS / f"arch-mlp__task-add__p-23__seed-{s}__lr-0.01__wd-0.03__tf-0.8"
        if (d / "trajectory.npz").exists():
            cells.append((d, 0.03))

    say(f"=== P1.2: weight-decay sweep at width 200, {len(cells)} runs ===")
    rows, t0 = [], time.time()
    for k, (d, wd) in enumerate(cells, 1):
        rows += one_run(d, wd)
        print(f"  [{k}/{len(cells)}] wd={wd} {d.name[:44]} ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "v4_p12_series.csv", index=False)

    say(f"\n{'wd':>6} {'n':>2} {'final tr/te':>12} {'wER init->fin (rel)':>22} "
        f"{'wER @memorised':>15} {'gER rel':>9} {'sil fin':>8}")
    per = []
    for (wd), g in D.groupby("wd"):
        for d_, gg in g.groupby("dir"):
            gg = gg.sort_values("epoch")
            w0, w1 = gg.within_eff_rank.iloc[0], gg.within_eff_rank.iloc[-1]
            g0, g1 = gg.global_eff_rank.iloc[0], gg.global_eff_rank.iloc[-1]
            mem = gg[gg.train_acc >= 0.999]
            w_mem = float(mem.within_eff_rank.iloc[0]) if len(mem) else np.nan
            per.append({"wd": wd, "dir": d_, "fin_tr": gg.train_acc.values[-3:].mean(),
                        "fin_te": gg.test_acc.values[-3:].mean(),
                        "w0": w0, "w1": w1, "wrel": (w1 - w0) / w0,
                        "grel": (g1 - g0) / g0, "w_mem_rel": (w_mem - w0) / w0,
                        "sil_fin": gg.sil_true.iloc[-1]})
    P = pd.DataFrame(per)
    P.to_csv(OUT / "v4_p12_perrun.csv", index=False)
    for wd, g in P.groupby("wd"):
        say(f"{wd:>6} {len(g):>2} {g.fin_tr.median():>6.2f}/{g.fin_te.median():.2f} "
            f"{g.w0.median():>7.1f}->{g.w1.median():5.1f} ({g.wrel.median():+.0%}) "
            f"{g.w_mem_rel.median():>14.0%} {g.grel.median():>8.0%} {g.sil_fin.median():>8.3f}")

    ok = P.dropna(subset=["wrel"])
    r = spearmanr(ok.wd, ok.wrel)
    say(f"\nDOSE-RESPONSE: Spearman(wd, terminal within-eff-rank rel change) = "
        f"{r.statistic:+.2f} (p = {r.pvalue:.3g}, n = {len(ok)})")
    ok2 = P.dropna(subset=["w_mem_rel"])
    if len(ok2) >= 6:
        r2 = spearmanr(ok2.wd, ok2.w_mem_rel)
        say(f"               at matched training state (first train-acc-1.0): "
            f"{r2.statistic:+.2f} (p = {r2.pvalue:.3g}, n = {len(ok2)})")

    (OUT / "v4_p12_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    say("\nwrote v4_p12_series.csv, v4_p12_perrun.csv, v4_p12_summary.txt")


if __name__ == "__main__":
    main()
