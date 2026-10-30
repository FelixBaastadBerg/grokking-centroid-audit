"""
v6.1 -- attribution-rule sensitivity of the endpoint Shapley share, raw recombination
values, and the cell-level within-class contrast.

(1) Factor-order sensitivity (P0.5): for each canonical seed, the raw endpoint 2x2
    recombination values f(C,R) for PH-dimension, the two order-specific centroid
    attributions (centroids advanced first vs second), their interaction, and the
    symmetric Shapley average. Written per seed to RESULTS/v6_recomb_raw.csv
    (machine-readable supplement for every estimator, not only PH-dimension).

(2) Raw per-interval Shapley contributions in estimator units (for the redesigned
    Figure 2b): phi_between and phi_within per checkpoint interval, per seed, written to
    RESULTS/v6_ckpt_contrib.csv.

(3) Cell-level within-class contrast (P2.4): grokked vs non-grokked terminal within-class
    effective-rank change aggregated within (task, p) cells first, so the pooled 53-vs-44
    comparison can be stated without configuration confounding.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd

R = REPO / "RESULTS"
ESTS = ["ph_dim", "H1_total", "eff_rank", "pca_pr", "twonn", "mle_k10", "corr_dim"]


def main():
    C = pd.read_csv(R / "c1_recombine.csv")

    # ---- (1) endpoint 2x2, both factor orders, interaction, Shapley
    rows = []
    for d, g in C.groupby("dir"):
        lo, hi = g.s.min(), g.s.max()
        for est in ESTS:
            def f(s_, t_):
                q = g[(g.s == s_) & (g.t == t_)][est]
                return float(q.iloc[0]) if len(q) else np.nan
            f00, f10, f01, f11 = f(lo, lo), f(hi, lo), f(lo, hi), f(hi, hi)
            diag = f00 - f11
            if not all(np.isfinite(v) for v in (f00, f10, f01, f11)) or abs(diag) < 1e-9:
                continue
            first = (f00 - f10) / diag        # centroids advanced first (residuals initial)
            second = (f01 - f11) / diag       # centroids advanced second (residuals converged)
            rows.append({"dir": d, "seed": int(g.seed.iloc[0]), "estimator": est,
                         "f00": f00, "f10": f10, "f01": f01, "f11": f11, "diag": diag,
                         "centroid_first_share": first, "centroid_second_share": second,
                         "interaction_share": first - second,
                         "shapley_between_share": 0.5 * (first + second)})
    RAW = pd.DataFrame(rows)
    RAW.to_csv(R / "v6_recomb_raw.csv", index=False)

    P = RAW[RAW.estimator == "ph_dim"]
    print("=== PH-dimension endpoint 2x2, per seed ===")
    print(P[["seed", "f00", "f10", "f01", "f11", "diag",
             "centroid_first_share", "centroid_second_share",
             "shapley_between_share"]].round(3).to_string(index=False))
    print("\nmedians: centroid-first {:.0%}  centroid-second {:.0%}  Shapley {:.0%}".format(
        P.centroid_first_share.median(), P.centroid_second_share.median(),
        P.shapley_between_share.median()))
    print("ranges:  first [{:.0%}, {:.0%}]  second [{:.0%}, {:.0%}]".format(
        P.centroid_first_share.min(), P.centroid_first_share.max(),
        P.centroid_second_share.min(), P.centroid_second_share.max()))
    print("median raw values: f00 {:.2f}  f10 {:.2f}  f01 {:.2f}  f11 {:.2f}".format(
        P.f00.median(), P.f10.median(), P.f01.median(), P.f11.median()))
    T = RAW[RAW.estimator == "twonn"]
    print("TwoNN: first {:.0%}  second {:.0%}  Shapley {:.0%}".format(
        T.centroid_first_share.median(), T.centroid_second_share.median(),
        T.shapley_between_share.median()))

    # ---- (2) raw per-interval contributions in estimator units
    PH = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    rows = []
    for d, g in C.groupby("dir"):
        for est in ["ph_dim", "twonn"]:
            def f(s_, t_):
                q = g[(g.s == s_) & (g.t == t_)][est]
                return float(q.iloc[0]) if len(q) else np.nan
            for a, b in PH:
                faa, fba, fab, fbb = f(a, a), f(b, a), f(a, b), f(b, b)
                if not all(np.isfinite(v) for v in (faa, fba, fab, fbb)):
                    continue
                phi_b = 0.5 * (faa - fba) + 0.5 * (fab - fbb)
                phi_w = 0.5 * (faa - fab) + 0.5 * (fba - fbb)
                rows.append({"dir": d, "seed": int(g.seed.iloc[0]), "estimator": est,
                             "a": a, "b": b, "phi_between": phi_b, "phi_within": phi_w,
                             "diag": faa - fbb})
    K = pd.DataFrame(rows)
    K.to_csv(R / "v6_ckpt_contrib.csv", index=False)
    print("\n=== raw per-interval Shapley contributions (PH-dim, estimator units) ===")
    ph = K[K.estimator == "ph_dim"]
    print(ph.groupby(["a", "b"])[["phi_between", "phi_within", "diag"]]
            .median().round(2).to_string())

    # ---- (3) cell-level within-class contrast
    W = pd.read_csv(R / "v3_within_series.csv")
    W = W[W.layer == "last_hidden"]
    recs = []
    for d, gg in W.groupby("dir"):
        gg = gg.sort_values("epoch")
        te = gg.test_acc.values[-3:].mean(); tr = gg.train_acc.values[-3:].mean()
        if gg.random_labels.iloc[0]:
            continue
        v = gg.within_eff_rank.values
        change = v[-1] / max(v[0], 1e-9) - 1
        status = "grok" if te >= 0.9 else ("mem" if tr >= 0.95 else None)
        if status is None:
            continue
        recs.append({"dir": d, "task": gg.task.iloc[0] if "task" in gg else "?",
                     "p": gg.p.iloc[0] if "p" in gg else "?", "status": status,
                     "change": change})
    D = pd.DataFrame(recs)
    if "task" in D and D.task.iloc[0] != "?":
        cell = D.groupby(["task", "p", "status"]).change.agg(["median", "count"])
        print("\n=== within-class eff-rank change by (task, p) cell ===")
        print((100 * cell["median"]).round(1).to_string())
        print(cell["count"].to_string())
    else:
        # derive task/p from the run directory name
        import re
        D["task"] = D.dir.str.extract(r"task-(\w+?)__")[0]
        D["p"] = D.dir.str.extract(r"p-(\d+)")[0].astype(int)
        cell = D.groupby(["task", "p", "status"]).change.agg(["median", "count"])
        print("\n=== within-class eff-rank change by (task, p) cell ===")
        piv = D.pivot_table(index=["task", "p"], columns="status", values="change",
                            aggfunc="median") * 100
        cnt = D.pivot_table(index=["task", "p"], columns="status", values="change",
                            aggfunc="count")
        piv.columns = [f"{c}_median_%" for c in piv.columns]
        cnt.columns = [f"{c}_n" for c in cnt.columns]
        print(piv.round(1).join(cnt).to_string())


if __name__ == "__main__":
    main()
