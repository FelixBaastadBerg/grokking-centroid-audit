"""
T2.1 -- checkpoint-recombination Shapley attribution beyond the canonical Z_23 cell.

The single most-requested empirical addition: the recombination attribution (and its Shapley
form, T0.5) was measured on one cell. Here the same protocol runs on
  * the Z_59 addition MLP cell (tag hb; seeds 0-1 pre-existing, 2-4 trained overnight),
  * the S_4 composition MLP cell (task-perm p-24, tf 0.8, plain suffix, 5 seeds),
  * the Transformer Z_23 addition cell (ss777, 3 seeds, full parameter trajectories).

Protocol per run: pre-logit activations at the initial and converged checkpoints on a fixed
400-point train subsample; endpoint 2x2 recombination f(C_s, R_t) with trace normalisation;
Shapley shares phi_between/phi_within of the diagonal change; both unaligned and with every
checkpoint Procrustes-aligned to the converged frame first. Per-cell medians with
leave-one-seed-out ranges.

Output: RESULTS/v3_crosscell.csv, v3_crosscell_summary.csv
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from analysis.lsrun import LandscapeRun
from tools.e1_centroid import measure

OUT = REPO / "RESULTS"
RUNS = REPO / "runs_topo"
N_SUB = 400
ESTS = ["ph_dim", "H1_total", "eff_rank", "pca_pr", "twonn", "mle_k10"]

CELLS = {
    "Z59_add_mlp": lambda n: "task-add__p-59" in n and n.endswith("__hb"),
    "S4_perm_mlp": lambda n: ("arch-mlp__task-perm__p-24" in n and "tf-0.8" in n
                              and n.endswith("tf-0.8")),
    "Z23_add_transformer": lambda n: ("arch-transformer__task-add__p-23" in n
                                      and n.endswith("__ss777")),
}


def acts_pair(d):
    """(H_init, H_final, y) on the fixed subsample, for MLP or transformer runs."""
    import json
    meta = json.loads((d / "run_meta.json").read_text())
    if meta["arch"] == "mlp":
        r = LandscapeRun(d)
        n = len(r.epochs)
        def acts(i):
            r._load(r.flat(i)); r.model.eval()
            with torch.no_grad():
                r.model(r.Xtr, keep_activations=True)
                return r.model.activations[-2].numpy()
        y_all = r.Ytr.numpy()
        H0, H1 = acts(0), acts(n - 1)
        fin = float(r.curves.test_acc.values[-5:].mean())
    else:
        from analysis.tfrun import TransformerRun
        r = TransformerRun(d)
        n = len(r.epochs)
        y_all = r.Ytr.numpy()
        H0, H1 = r.activations(0, r.Xtr), r.activations(n - 1, r.Xtr)
        fin = float(r.curves.test_acc.values[-5:].mean())
    sub = np.sort(np.random.default_rng(0).choice(len(y_all), min(N_SUB, len(y_all)),
                                                  replace=False))
    return H0[sub], H1[sub], y_all[sub], fin, meta["seed"]


def parts(H, y):
    cen = np.zeros_like(H)
    for c in np.unique(y):
        cen[y == c] = H[y == c].mean(0)
    return cen, H - cen


def procrustes_to(A, B):
    """Orthogonal Q minimising ||A Q - B||_F (A, B centred)."""
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(Ac.T @ Bc)
    return U @ Vt


def norm_measure(Z, y):
    Zc = Z - Z.mean(0)
    Z = Zc / (np.sqrt((Zc ** 2).sum(1).mean()) + 1e-12)
    return measure(Z, y)


def one_run(d):
    H0, H1, y, fin, seed = acts_pair(d)
    rows = []
    for frame in ("unaligned", "aligned"):
        A0, A1 = H0, H1
        if frame == "aligned":
            A0 = (H0 - H0.mean(0)) @ procrustes_to(H0, H1) + H1.mean(0)
        c0, r0 = parts(A0, y)
        c1, r1 = parts(A1, y)
        for cs, rs, tag in ((c0, r0, (0, 0)), ((c1), r0, (1, 0)),
                            (c0, r1, (0, 1)), (c1, r1, (1, 1))):
            m = norm_measure(cs + rs, y)
            rows.append({"dir": d.name, "seed": seed, "final_test": fin, "frame": frame,
                         "C": tag[0], "R": tag[1], **m})
    return rows


def shapley(g, est):
    """(between_share, diag). Share is NaN when the run's own training-time change is too small
    to decompose (|diag| < 5% of the initial value): a share of nothing is not a share."""
    def f(C, R):
        q = g[(g.C == C) & (g.R == R)][est]
        return float(q.iloc[0]) if len(q) else np.nan
    diag = f(0, 0) - f(1, 1)
    if not np.isfinite(diag) or abs(diag) < 0.05 * max(abs(f(0, 0)), 1e-9):
        return np.nan, diag
    phi_b = 0.5 * (f(0, 0) - f(1, 0)) + 0.5 * (f(0, 1) - f(1, 1))
    return phi_b / diag, diag


def main():
    t0 = time.time()
    rows = []
    for cell, match in CELLS.items():
        dirs = [d for d in sorted(RUNS.iterdir())
                if d.is_dir() and match(d.name) and (d / "trajectory.npz").exists()]
        print(f"--- {cell}: {len(dirs)} runs ---", flush=True)
        for d in dirs:
            try:
                rr = one_run(d)
            except Exception as e:
                print(f"  SKIP {d.name}: {type(e).__name__} {e}", flush=True)
                continue
            for r in rr:
                r["cell"] = cell
            rows += rr
            print(f"  {d.name[:55]} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "v3_crosscell.csv", index=False)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "v3_crosscell.csv", index=False)

    summ = []
    for (cell, frame), g in D.groupby(["cell", "frame"]):
        for est in ESTS:
            per, diags, n_run = {}, [], 0
            for d_, gg in g.groupby("dir"):
                if gg.final_test.iloc[0] < 0.9:
                    continue
                n_run += 1
                b, diag = shapley(gg, est)
                diags.append(diag)
                if np.isfinite(b):
                    per[d_] = b
            if n_run < 2:
                continue
            v = np.array(list(per.values()))
            rec = {"cell": cell, "frame": frame, "estimator": est, "n_runs": n_run,
                   "n_decomposable": len(v),
                   "diag_median": float(np.nanmedian(diags))}
            if len(v) >= 2:
                jack = ([np.median(np.delete(v, i)) for i in range(len(v))]
                        if len(v) >= 3 else [v.min(), v.max()])
                rec.update({"between_share": float(np.median(v)),
                            "loso_lo": float(np.min(jack)), "loso_hi": float(np.max(jack)),
                            "per_seed": ";".join(f"{x:.3f}" for x in v)})
            summ.append(rec)
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "v3_crosscell_summary.csv", index=False)
    print("\n=== Shapley between-class share by cell (grokked runs, unaligned) ===")
    P = S[S.frame == "unaligned"].pivot(index="estimator", columns="cell",
                                        values="between_share")
    print((100 * P).round(0).to_string())
    print(f"\nwrote v3_crosscell.csv, v3_crosscell_summary.csv ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
