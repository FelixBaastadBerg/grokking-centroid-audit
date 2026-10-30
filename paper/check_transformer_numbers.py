"""
Check for every modular-division Transformer number quoted in the paper (Sec. 4, Figure 3,
Appendix D.6). Recomputes from RESULTS/brown_transformer_*.csv and RESULTS/brown_parts/ and exits
nonzero on any mismatch. Also checks structural invariants: every result file is present, and no
pilot run is pooled with the ten confirmation seeds.

Run from the repository root:  python paper/check_transformer_numbers.py
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd

R = REPO / "RESULTS"
bad = []


def chk(name, claimed, actual, tol=0.02, rel=False):
    if actual is None or (isinstance(actual, float) and not np.isfinite(actual)):
        print(f"  ?? {name}: could not recompute"); bad.append(name); return
    d = abs(claimed - actual) / max(abs(actual), 1e-9) if rel else abs(claimed - actual)
    ok = d <= tol
    print(f"  {'OK  ' if ok else 'MISMATCH'} {name}: paper {claimed}  recomputed {actual:.4g}")
    if not ok:
        bad.append(name)


FILES = ["brown_transformer_run_manifest.csv", "brown_transformer_trajectories.csv",
         "brown_transformer_leakage.csv", "brown_transformer_intervention.csv",
         "brown_transformer_recombination_raw.csv",
         "brown_transformer_attribution_summary.csv",
         "brown_transformer_plateau.csv"]

print("=== structural checks ===")
for f in FILES:
    ok = (R / f).exists()
    print(f"  {'OK  ' if ok else 'MISSING'} {f}")
    if not ok:
        bad.append(f)
if not bad:
    M = pd.read_csv(R / FILES[0])
    chk("pilot runs", 4, len(M[M.tag == "pilot"]), tol=0)
    chk("confirmation runs", 10, len(M[M.tag == "confirm"]), tol=0)
    conf_dirs = set(M[M.tag == "confirm"].dir)
    for f in FILES[1:5]:
        d = pd.read_csv(R / f)
        pooled = set(d.dir) - conf_dirs
        print(f"  {'OK  ' if not pooled else 'POOLED PILOTS'} {f}: confirmation-only")
        if pooled:
            bad.append(f)

    # ---- numbers quoted in the paper (Sec. 4 and Appendix D.6) ----
    C = M[M.tag == "confirm"]
    print("\n=== run outcomes ===")
    chk("grokked", 9, int(C.grokked.sum()), tol=0)
    chk("not grokked", 1, int((~C.grokked).sum()), tol=0)
    chk("t_grok min", 10800, int(C[C.grokked].t_grok_step.min()), tol=0)
    chk("t_grok max", 25500, int(C[C.grokked].t_grok_step.max()), tol=0)
    chk("non-grokked final val", 0.013, float(C[~C.grokked].final_val_acc.iloc[0]),
        tol=0.001)
    chk("pilot grokked (round 2)", 4, int(M[(M.tag == "pilot2")].grokked.sum()), tol=0)
    chk("pilot grokked (round 1)", 0, int(M[(M.tag == "pilot")].grokked.sum()), tol=0)

    print("\n=== TwoNN plateau-to-final fall (Brown's qualitative result) ===")
    T = pd.read_csv(R / FILES[1])
    Mi = M.set_index("dir")
    falls, plateaus = [], []
    for d_, g in T.groupby("dir"):
        g = g.sort_values("step")
        fall = 1 - g.twonn.iloc[-1] / g.twonn.max()
        (falls if Mi.loc[d_, "grokked"] else plateaus).append(fall)
    chk("median fall, grokked", 0.33, float(np.median(falls)), tol=0.01)
    chk("min fall, grokked", 0.25, float(np.min(falls)), tol=0.01)
    chk("max fall, grokked", 0.50, float(np.max(falls)), tol=0.01)
    chk("fall, non-grokked", 0.00, float(plateaus[0]), tol=0.005)
    chk("median plateau value", 19.1, float(np.median(
        [g.twonn.max() for _, g in T.groupby("dir")])), tol=0.1)

    print("\n=== attribution, grokked seeds with appreciable change ===")
    for est, med, lo, hi, first, second in [("ph_dim", 124, 109, 156, 103, 149),
                                            ("twonn", 21, 3, 39, 105, -72)]:
        P = pd.read_csv(R / "brown_parts" / f"attr_perseed_unaligned_{est}.csv")
        G = P[P.grokked & P.appreciable]
        chk(f"{est} n seeds", 9, len(G), tol=0)
        chk(f"{est} share %", med, 100 * G.shapley_between.median(), tol=0.6)
        chk(f"  seed min", lo, 100 * G.shapley_between.min(), tol=0.6)
        chk(f"  seed max", hi, 100 * G.shapley_between.max(), tol=0.6)
        chk(f"  order first", first, 100 * G.centroid_first.median(), tol=0.6)
        chk(f"  order second", second, 100 * G.centroid_second.median(), tol=0.6)
    Pp = pd.read_csv(R / "brown_parts" / "attr_perseed_unaligned_ph_dim.csv")
    v = Pp[Pp.grokked & Pp.appreciable].shapley_between.values
    rng = np.random.default_rng(0)
    boots = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(10_000)]
    jack = [np.median(np.delete(v, i)) for i in range(len(v))]
    chk("ph_dim pct lo", 116, float(np.percentile(boots, 2.5)) * 100, tol=0.6)
    chk("ph_dim pct hi", 151, float(np.percentile(boots, 97.5)) * 100, tol=0.6)
    chk("ph_dim loso lo", 123, 100 * min(jack), tol=0.6)
    chk("ph_dim loso hi", 130, 100 * max(jack), tol=0.6)
    chk("ph_dim orders first min", 100, 100 * Pp[Pp.grokked].centroid_first.min(), tol=0.6)
    chk("ph_dim orders first max", 108, 100 * Pp[Pp.grokked].centroid_first.max(), tol=0.6)
    chk("ph_dim orders second min", 115, 100 * Pp[Pp.grokked].centroid_second.min(), tol=0.6)
    chk("ph_dim orders second max", 207, 100 * Pp[Pp.grokked].centroid_second.max(), tol=0.6)
    chk("non-grokked ph share %", -43,
        100 * Pp[~Pp.grokked].shapley_between.iloc[0], tol=0.6)
    chk("all appreciable", 10, int(Pp.appreciable.sum()), tol=0)
    for est, med, lo, hi in [("eff_rank", 80, 79, 81), ("mle_k10", 83, 81, 86)]:
        P = pd.read_csv(R / "brown_parts" / f"attr_perseed_unaligned_{est}.csv")
        G = P[P.grokked & P.appreciable]
        j = [np.median(np.delete(G.shapley_between.values, i)) for i in range(len(G))]
        chk(f"{est} share %", med, 100 * G.shapley_between.median(), tol=0.6)
        chk(f"  loso lo", lo, 100 * min(j), tol=0.6)
        chk(f"  loso hi", hi, 100 * max(j), tol=0.6)

    print("\n=== matched plateau-to-convergence decomposition (Sec. 4, Appendix D.6) ===")
    PL = pd.read_csv(R / "brown_transformer_plateau.csv")
    Gp = PL[(PL.frame == "unaligned") & PL.grokked]
    chk("plateau seeds", 9, Gp.seed.nunique(), tol=0)
    chk("plateau step min", 8500, int(Gp.step_plateau.min()), tol=0)
    chk("plateau step max", 20000, int(Gp.step_plateau.max()), tol=0)
    chk("L2 at plateau", 0.98, float(Gp.L2_plateau.median()), tol=0.005)
    chk("L2 at convergence", 0.0, float(Gp.L2_final.median()), tol=1e-9)
    Tw = Gp[Gp.estimator == "twonn"]
    chk("TwoNN raw decline median", 5.63, float(Tw.raw_change.median()), tol=0.006)
    chk("  decline min", 2.94, float(Tw.raw_change.min()), tol=0.006)
    chk("  decline max", 9.06, float(Tw.raw_change.max()), tol=0.006)
    chk("TwoNN declines in all 9 seeds", 9, int((Tw.raw_change > 0).sum()), tol=0)
    w_first = 1 - Tw.centroid_first_share
    w_second = 1 - Tw.centroid_second_share
    w_sym = 1 - Tw.shapley_between_share
    chk("TwoNN within, first", 131, 100 * float(w_first.median()), tol=0.6)
    chk("  first min", 98, 100 * float(w_first.min()), tol=0.6)
    chk("  first max", 261, 100 * float(w_first.max()), tol=0.6)
    chk("TwoNN within, second", 100, 100 * float(w_second.median()), tol=0.6)
    chk("  second min", 97, 100 * float(w_second.min()), tol=0.6)
    chk("TwoNN within, symmetric", 115, 100 * float(w_sym.median()), tol=0.6)
    chk("  symmetric min (>= 97 claim)", 99, 100 * float(w_sym.min()), tol=0.6)
    chk("min within share over BOTH orders", 97,
        100 * min(w_first.min(), w_second.min()), tol=0.6)
    chk("centroid-second |contribution| max", 0.09,
        float(Tw.centroid_second_raw.abs().max()), tol=0.006)
    chk("  seeds with |second| < 0.01", 7,
        int((Tw.centroid_second_raw.abs() < 0.01).sum()), tol=0)
    chk("  seeds with negative first contribution", 7,
        int((Tw.centroid_first_raw < 0).sum()), tol=0)
    Php = Gp[Gp.estimator == "ph_dim"]
    wp = 1 - Php.shapley_between_share
    chk("PH raw decline median", 14.61, float(Php.raw_change.median()), tol=0.006)
    chk("  decline min", 12.94, float(Php.raw_change.min()), tol=0.006)
    chk("  decline max", 15.28, float(Php.raw_change.max()), tol=0.006)
    chk("PH within, symmetric", 92, 100 * float(wp.median()), tol=0.6)
    chk("  within min", 64, 100 * float(wp.min()), tol=0.6)
    chk("  within max", 107, 100 * float(wp.max()), tol=0.6)
    chk("PH within, first", 99, 100 * float((1 - Php.centroid_first_share).median()),
        tol=0.6)
    chk("PH within, second", 82, 100 * float((1 - Php.centroid_second_share).median()),
        tol=0.6)
    chk("PH plateau value (approx 17)", 17.2, float(Php.f00.median()), tol=0.2)
    Ap = PL[(PL.frame == "aligned") & PL.grokked]
    for est, pts in (("twonn", 1.0), ("ph_dim", 2.0)):
        u = Gp[Gp.estimator == est].shapley_between_share.median()
        a = Ap[Ap.estimator == est].shapley_between_share.median()
        chk(f"{est} plateau alignment delta (pts)", pts, 100 * abs(u - a), tol=0.06)

    print("\n=== alignment control ===")
    S = pd.read_csv(R / FILES[5]).set_index(["frame", "estimator"])
    chk("ph_dim aligned-unaligned (pts)", 0.2,
        abs(100 * (S.loc[("aligned", "ph_dim"), "between_median"]
                   - S.loc[("unaligned", "ph_dim"), "between_median"])), tol=0.06)
    chk("twonn aligned-unaligned (pts)", 0.4,
        abs(100 * (S.loc[("aligned", "twonn"), "between_median"]
                   - S.loc[("unaligned", "twonn"), "between_median"])), tol=0.06)

    print("\n=== L2 and the intervention on converged clouds ===")
    L = pd.read_csv(R / FILES[2])
    Mi2 = M.set_index("dir")
    lags = []
    for d_, g in L.groupby("dir"):
        g = g.sort_values("step")
        if not Mi2.loc[d_, "grokked"]:
            chk("non-grokked final L2", 0.99, float(g.L2.iloc[-1]), tol=0.005)
            continue
        z = g[g.L2 <= 1e-9]
        lags.append(int(z.step.iloc[0]) - int(Mi2.loc[d_, "t_grok_step"]))
    chk("L2 init", 0.99, float(L.groupby("dir").L2.first().median()), tol=0.005)
    chk("L2=0 lag after t_grok, min", 1100, min(lags), tol=0)
    chk("L2=0 lag after t_grok, max", 2900, max(lags), tol=0)
    IV = pd.read_csv(R / FILES[3])
    IVg = IV[IV.seed != int(C[~C.grokked].seed.iloc[0])]
    fold = IVg.groupby("seed").apply(
        lambda x: x[x.alpha == 0].ph_dim.iloc[0] / x[x.alpha == 8].ph_dim.iloc[0])
    tw = IVg.groupby("seed").apply(
        lambda x: x[x.alpha == 0].twonn.iloc[0] / x[x.alpha == 8].twonn.iloc[0])
    chk("intervention PH fold", 13.8, float(fold.median()), tol=0.06)
    chk("  fold min", 10.4, float(fold.min()), tol=0.06)
    chk("  fold max", 17.8, float(fold.max()), tol=0.06)
    chk("intervention TwoNN fold", 1.43, float(tw.median()), tol=0.006)
    chk("  TwoNN fold min", 1.06, float(tw.min()), tol=0.006)
    chk("  TwoNN fold max", 1.58, float(tw.max()), tol=0.006)
    med = IV.groupby("alpha")[["L2", "twonn", "ph_dim"]].median()
    chk("sweep L2 at alpha 0", 0.96, float(med.loc[0.0, "L2"]), tol=0.005)
    chk("sweep L2 at alpha 1", 0.0, float(med.loc[1.0, "L2"]), tol=1e-6)
    chk("TwoNN constant alpha 1..8", 0.0,
        float(med.loc[[1.0, 2.0, 4.0, 8.0], "twonn"].std()), tol=1e-4)
    chk("TwoNN value on flat stretch", 13.07, float(med.loc[1.0, "twonn"]), tol=0.006)
    chk("PH further fall alpha 1->8", 2.2,
        float(med.loc[1.0, "ph_dim"] / med.loc[8.0, "ph_dim"]), tol=0.06)

print()
if bad:
    print(f"{len(bad)} PROBLEM(S): {bad}")
    sys.exit(1)
print("ALL BROWN-STYLE CHECKS OK")
