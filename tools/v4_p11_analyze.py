"""
P1.1 -- the narrow-width claim across 20 seeds of ONE fixed configuration.

Z_59 addition, width 16, lr 0.01, wd 0.03, tf 0.8, 60k epochs, seeds 0-19. EVERY seed enters the
analysis; non-grokked seeds are censored, never dropped. Per checkpoint (~31 per run) on
full-table activations (m = 59 points per class, the calibrated regime):
conditional TwoNN, conditional MLE-ID k10, within-class effective rank, silhouette, accuracies.

Reported, in the work order's order:
  1. fraction that grok (final test >= 0.9; also the 0.5 threshold for sensitivity);
  2. per-seed trajectories (CSV + printed endpoints);
  3. eventual grokkers vs non-generalising memorisers, terminal conditional values;
  4. the PRE-grok within-class change (init -> the checkpoint just before t_grok);
  5. does EARLY within-class simplification predict eventual grokking? (value and slope at a
     fixed early epoch, grokkers vs not, rank-sum + AUC);
  6. time-to-grok with censoring: Kaplan-Meier-style survival at 60k, median if defined.

Output: RESULTS/v4_p11_series.csv, v4_p11_summary.txt
"""
from __future__ import annotations
import os, sys, json, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from tools.v2_conditional import conditional_battery
from tools.train_topology import build_task

OUT = REPO / "RESULTS"
RUNS = REPO / "runs_topo"
MAX_CK = 31
EARLY_EP = 6000          # 10% of the run, well before every observed t_grok
LINES = []


def say(s=""):
    print(s, flush=True); LINES.append(s)


def one_run(d):
    meta = json.loads((d / "run_meta.json").read_text())
    tr = np.load(d / "trajectory.npz")
    curves = pd.read_csv(d / "curves.csv").set_index("epoch")
    pairs, y, n_sym, _ = build_task(meta["task"], meta["p"])
    N = len(y)
    X = np.zeros((N, 2 * n_sym), np.float32)
    X[np.arange(N), pairs[:, 0]] = 1.0
    X[np.arange(N), n_sym + pairs[:, 1]] = 1.0
    epochs = tr["epochs"]
    idx = (np.unique(np.round(np.linspace(0, len(epochs) - 1, MAX_CK)).astype(int))
           if len(epochs) > MAX_CK else np.arange(len(epochs)))
    rows = []
    for i in idx:
        h = np.maximum(X @ tr["W0"][i].T, 0.0)
        H = np.maximum(h @ tr["W1"][i].T, 0.0)
        ep = int(epochs[i])
        cv = curves.loc[ep] if ep in curves.index else None
        rows.append({"dir": d.name, "seed": meta["seed"], "epoch": ep,
                     "train_acc": float(cv.train_acc) if cv is not None else np.nan,
                     "test_acc": float(cv.test_acc) if cv is not None else np.nan,
                     **conditional_battery(H, y)})
    return rows


def main():
    dirs = [d for d in sorted(RUNS.iterdir())
            if d.is_dir() and "task-add__p-59" in d.name and d.name.endswith("__w16")
            and "wd-0.03" in d.name and (d / "trajectory.npz").exists()]
    say(f"=== P1.1: Z59 w16 fixed configuration, {len(dirs)} seeds ===")
    rows, t0 = [], time.time()
    for k, d in enumerate(dirs, 1):
        rows += one_run(d)
        print(f"  [{k}/{len(dirs)}] {d.name[:50]} ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "v4_p11_series.csv", index=False)

    # per-seed summary
    per = []
    for d_, g in D.groupby("dir"):
        g = g.sort_values("epoch")
        fin = g.test_acc.values[-3:].mean()
        grokked = fin >= 0.9
        tg = np.nan
        if fin >= 0.5:
            hit = g[g.test_acc >= 0.5 * fin]
            tg = float(hit.epoch.min())
        pre = g[g.epoch < tg] if np.isfinite(tg) else g
        e_row = g.iloc[(g.epoch - EARLY_EP).abs().argmin()]
        e0 = g.iloc[0]
        # early slope: change in cond_twonn from init to EARLY_EP
        per.append({
            "dir": d_, "seed": int(g.seed.iloc[0]), "fin_test": fin, "grokked": grokked,
            "t_grok": tg, "fin_train": float(g.train_acc.values[-3:].mean()),
            "ct_init": float(e0.cond_twonn), "ct_final": float(g.cond_twonn.iloc[-1]),
            "cm_init": float(e0.cond_mle_k10), "cm_final": float(g.cond_mle_k10.iloc[-1]),
            "wer_init": float(e0.within_eff_rank),
            "wer_final": float(g.within_eff_rank.iloc[-1]),
            "ct_pregrok": float(pre.cond_twonn.iloc[-1]) if len(pre) else np.nan,
            "wer_pregrok": float(pre.within_eff_rank.iloc[-1]) if len(pre) else np.nan,
            "ct_early": float(e_row.cond_twonn), "wer_early": float(e_row.within_eff_rank),
            "ct_early_slope": float(e_row.cond_twonn - e0.cond_twonn),
            "sil_final": float(g.global_eff_rank.iloc[-1]),   # placeholder col kept minimal
        })
    P = pd.DataFrame(per).sort_values("seed")
    P.to_csv(OUT / "v4_p11_perseed.csv", index=False)

    n, ng = len(P), int(P.grokked.sum())
    say(f"\n1. GROK FRACTION: {ng}/{n} at final test >= 0.9 "
        f"({int((P.fin_test >= 0.5).sum())}/{n} at >= 0.5)")

    say("\n2. per-seed endpoints (cond TwoNN, cond MLE, within-eff-rank; * = grokked):")
    for _, r in P.iterrows():
        say(f"   seed {int(r.seed):>2}{'*' if r.grokked else ' '} test {r.fin_test:.2f}  "
            f"ctwonn {r.ct_init:5.1f}->{r.ct_final:5.1f}  cmle {r.cm_init:5.1f}->{r.cm_final:5.1f}  "
            f"wER {r.wer_init:5.1f}->{r.wer_final:5.1f}"
            + (f"  t_grok {r.t_grok:6.0f}" if np.isfinite(r.t_grok) else "  censored"))

    G, M = P[P.grokked], P[(~P.grokked) & (P.fin_train >= 0.95)]
    say(f"\n3. GROKKERS (n={len(G)}) vs NON-GENERALISING MEMORISERS (n={len(M)}):")
    for c, lab in [("ct_final", "cond TwoNN final"), ("cm_final", "cond MLE final"),
                   ("wer_final", "within-eff-rank final")]:
        u = (mannwhitneyu(G[c], M[c], alternative="less")
             if len(G) >= 3 and len(M) >= 3 else None)
        say(f"   {lab:<24} grokked {G[c].median():5.1f}  memoriser {M[c].median():5.1f}"
            + (f"   Mann-Whitney(one-sided) p = {u.pvalue:.3g}" if u else ""))
    say(f"   relative change, cond TwoNN: grokked "
        f"{((G.ct_final-G.ct_init)/G.ct_init).median():+.0%}  memoriser "
        f"{((M.ct_final-M.ct_init)/M.ct_init).median():+.0%}" if len(M) else "")

    say(f"\n4. PRE-GROK within-class change (init -> last checkpoint before t_grok, grokkers):")
    if len(G):
        say(f"   cond TwoNN {G.ct_init.median():.1f} -> {G.ct_pregrok.median():.1f} "
            f"({((G.ct_pregrok-G.ct_init)/G.ct_init).median():+.0%}); "
            f"within-eff-rank {G.wer_init.median():.1f} -> {G.wer_pregrok.median():.1f} "
            f"({((G.wer_pregrok-G.wer_init)/G.wer_init).median():+.0%})")

    say(f"\n5. EARLY PREDICTION (epoch ~{EARLY_EP}):")
    for c, lab in [("ct_early", "cond TwoNN value"), ("wer_early", "within-eff-rank value"),
                   ("ct_early_slope", "cond TwoNN early change")]:
        if len(G) >= 3 and len(M) >= 3:
            u = mannwhitneyu(G[c], M[c])
            auc = u.statistic / (len(G) * len(M))
            say(f"   {lab:<26} grokked {G[c].median():6.2f}  memoriser {M[c].median():6.2f}  "
                f"AUC = {auc:.2f}  p = {u.pvalue:.3g}")
        gg = P[np.isfinite(P.t_grok)]
        if len(gg) >= 5:
            r = spearmanr(gg[c], gg.t_grok)
            say(f"     among grokkers: rho({lab}, t_grok) = {r.statistic:+.2f} (p={r.pvalue:.2g})")

    say(f"\n6. TIME-TO-GROK with censoring at 60000:")
    times = np.where(np.isfinite(P.t_grok), P.t_grok, 60000.0)
    event = np.isfinite(P.t_grok.values)
    order = np.argsort(times)
    surv, at_risk = 1.0, n
    km_median = np.nan
    for i in order:
        if event[i]:
            surv *= (at_risk - 1) / at_risk
            if surv <= 0.5 and not np.isfinite(km_median):
                km_median = times[i]
        at_risk -= 1
    say(f"   events {int(event.sum())}/{n}, censored {n - int(event.sum())}; "
        f"KM survival at 60k = {surv:.2f}; KM median t_grok = "
        + (f"{km_median:.0f}" if np.isfinite(km_median) else "undefined (>60k)"))
    if event.sum():
        say(f"   among grokkers: t_grok median {np.median(times[event]):.0f}, "
            f"range {times[event].min():.0f}-{times[event].max():.0f}")

    (OUT / "v4_p11_summary.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    say("\nwrote v4_p11_series.csv, v4_p11_perseed.csv, v4_p11_summary.txt")


if __name__ == "__main__":
    main()
