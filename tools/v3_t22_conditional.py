"""
T2.2 -- does class-conditional intrinsic dimension actually change through grokking, measured in
a regime BUILT to be measurable?

v2 could not answer this: at the canonical operating point (m ~ 17 points per class, width 200)
conditional TwoNN is calibrated only to d ~ 8 and its readings sat off scale. Two levers move
the problem into calibration, both taken here:
  * larger m: evaluate on the FULL group table (m = p per class: 23 or 59), and use the Z_59
    cell (m = 59);
  * narrower networks: hidden width w in {16, 32} bounds the class-conditional dimension by w,
    inside the validated range at these m.

GATE first (per the work order): conditional TwoNN / MLE-ID / within-spectra on synthetic clouds
with known within-class d at exactly the (K, m) of each measured cell; an estimator is usable at
(K, m, d) only if its median estimate is within 25% of the truth. Then the measurement: the
conditional battery at five checkpoints (init, t_grok/2, t_grok, mid, converged) for every
grokked run of the narrow-width cells and the Z_59 cell, on full-table activations.

Output: RESULTS/v3_cond_gate.csv, v3_cond_trajectory.csv, printed answer.
"""
from __future__ import annotations
import os, sys, json, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from tools.v2_conditional import conditional_battery, cloud
from tools.train_topology import build_task

OUT = REPO / "RESULTS"
RUNS = REPO / "runs_topo"


def gate():
    print("=== T2.2 GATE: conditional estimators at the measured (K, m) ===", flush=True)
    rows = []
    for K, m in ((23, 23), (59, 59)):
        for d in (2, 4, 8, 12, 16, 24, 32):
            for s in (0, 1, 2):
                X, y = cloud(d, K, m, 0.05, seed=s)
                rows.append({"K": K, "m": m, "true_d": d, "seed": s,
                             **conditional_battery(X, y)})
        print(f"  K={K} m={m} done", flush=True)
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "v3_cond_gate.csv", index=False)
    print(f"\n  {'K':>4}{'m':>5}{'true d':>7}{'cond TwoNN':>12}{'cond MLE10':>12}"
          f"{'within eff rank':>17}")
    usable = {}
    for (K, m, d), g in G.groupby(["K", "m", "true_d"]):
        ct, cm, wr = g.cond_twonn.median(), g.cond_mle_k10.median(), g.within_eff_rank.median()
        print(f"  {K:>4}{m:>5}{d:>7}{ct:>12.1f}{cm:>12.1f}{wr:>17.1f}")
        for name, v in (("cond_twonn", ct), ("cond_mle_k10", cm), ("within_eff_rank", wr)):
            if np.isfinite(v) and abs(v - d) / d <= 0.25:
                usable[(K, name)] = max(usable.get((K, name), 0), d)
    print("\n  usable-to (largest true d within 25%):")
    for (K, name), d in sorted(usable.items()):
        print(f"    K={K:<3} {name:<16} d <= {d}")
    return usable


def full_table_acts(d):
    """Activations of the whole p^2 table (m = p per class) from an MLP trajectory."""
    meta = json.loads((d / "run_meta.json").read_text())
    tr = np.load(d / "trajectory.npz")
    curves = pd.read_csv(d / "curves.csv")
    pairs, y, n_sym, _ = build_task(meta["task"], meta["p"])
    N = len(y)
    X = np.zeros((N, 2 * n_sym), np.float32)
    X[np.arange(N), pairs[:, 0]] = 1.0
    X[np.arange(N), n_sym + pairs[:, 1]] = 1.0
    te = curves.test_acc.values
    fin = te[-5:].mean()
    tg = int(np.argmax(te >= 0.5 * fin)) if fin >= 0.5 else -1
    n = len(tr["epochs"])
    # curves rows correspond to checkpoints 1:1 (same save_every)
    idx = sorted(set([0, max(1, tg // 2), tg if tg > 0 else n // 2,
                      (tg + n) // 2 if tg > 0 else (3 * n) // 4, n - 1]))
    stages = ["init", "pre", "t_grok", "mid", "final"][:len(idx)]
    out = []
    for st, i in zip(stages, idx):
        h = np.maximum(X @ tr["W0"][i].T, 0.0)
        H = np.maximum(h @ tr["W1"][i].T, 0.0)
        out.append((st, int(tr["epochs"][i]), H))
    return out, y, float(fin), meta


def main():
    usable = gate()

    print("\n=== T2.2 MEASUREMENT: conditional battery on full-table activations ===",
          flush=True)
    cells = []
    for d in sorted(RUNS.iterdir()):
        n = d.name
        if not (d / "trajectory.npz").exists():
            continue
        if ("__w16" in n or "__w32" in n) and "wd-0.03" in n:
            cells.append(("narrow_" + ("p59" if "p-59" in n else "p23"), d))
        elif "task-add__p-59" in n and n.endswith("__hb"):
            cells.append(("p59_w200", d))
    rows, t0 = [], time.time()
    for cell, d in cells:
        try:
            stages, y, fin, meta = full_table_acts(d)
        except Exception as e:
            print(f"  SKIP {d.name}: {type(e).__name__} {e}", flush=True)
            continue
        for st, ep, H in stages:
            rows.append({"cell": cell, "dir": d.name, "hidden": meta["hidden"],
                         "p": meta["p"], "seed": meta["seed"], "final_test": fin,
                         "stage": st, "epoch": ep, **conditional_battery(H, y)})
        print(f"  {d.name[:58]}  final_test={fin:.2f} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "v3_cond_trajectory.csv", index=False)
    D = pd.DataFrame(rows)
    if not len(D):
        print("no runs available yet"); return
    D.to_csv(OUT / "v3_cond_trajectory.csv", index=False)

    print("\n=== the answer, by cell (grokked runs only) ===")
    G = D[D.final_test >= 0.9]
    for cell, g in G.groupby("cell"):
        i, f = g[g.stage == "init"], g[g.stage == "final"]
        if not len(i) or not len(f):
            continue
        print(f"--- {cell} (n = {g.dir.nunique()} runs, width {g.hidden.iloc[0]}, "
              f"m = {g.p.iloc[0]}) ---")
        for c in ("cond_twonn", "cond_mle_k10", "within_eff_rank"):
            print(f"  {c:<18} init {i[c].median():6.1f} -> final {f[c].median():6.1f}   "
                  f"({(f[c].median()-i[c].median())/i[c].median():+.0%})")


if __name__ == "__main__":
    main()
