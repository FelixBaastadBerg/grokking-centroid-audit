"""
Headline-number check for "Auditing Geometric Change in Grokking: Centroid Configuration versus
Within-Class Geometry" (NeurReps 2026).

Recomputes every load-bearing number quoted in the main text, and the appendix numbers they rest
on, directly from the result CSVs in RESULTS*/, and compares each against the value printed in the
paper. Prints OK / MISMATCH per item and exits nonzero on any mismatch.

The Transformer (modular-division) numbers are checked by paper/check_transformer_numbers.py,
which this script runs at the end.

Run from the repository root:

    python paper/check_headline_numbers.py

Needs only numpy and pandas (plus matplotlib, because two figure-input checks import
paper/make_figures.py). No training runs, checkpoints or activations are read.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "paper"))

import numpy as np
import pandas as pd

R = REPO / "RESULTS"
R4 = REPO / "RESULTS_v6_4"
R8 = REPO / "RESULTS_v6_8"
RC = REPO / "RESULTS_camera_ready"
bad, missing = [], []


def chk(name, claimed, actual, tol=0.02, rel=False):
    """Compare a value printed in the paper (`claimed`) with the value recomputed from a CSV."""
    if actual is None or (isinstance(actual, float) and not np.isfinite(actual)):
        print(f"  ??       {name}: could not recompute")
        missing.append(name)
        return
    d = abs(claimed - actual) / max(abs(actual), 1e-9) if rel else abs(claimed - actual)
    ok = d <= tol
    print(f"  {'OK  ' if ok else 'MISMATCH'} {name}: paper {claimed}  recomputed {actual:.4g}")
    if not ok:
        bad.append(name)


# ============================================================ populations (Sec. 2, App. B.2)
print("=== populations (Sec. 2, Appendix B.2) ===")
t11 = pd.read_csv(R / "v3_t11_run_table.csv")
corpus = t11[~t11.random_labels.astype(bool)]
chk("MLP corpus runs", 96, len(corpus), tol=0)
chk("grokked (mean-of-last-5 rule)", 53, int(corpus.grokked.astype(bool).sum()), tol=0)
chk("non-generalising memorisers", 43, int((~corpus.grokked.astype(bool)).sum()), tol=0)
chk("random-label controls", 3, int(t11.random_labels.astype(bool).sum()), tol=0)

K = pd.read_csv(R4 / "v6_4_count_resolution.csv", index_col=0)["value"]
chk("MLP corpus runs (ledger)", 96, float(K["n_mlp_corpus"]), tol=0)
chk("grokked (ledger)", 53, float(K["n_grokked_mean5_rule"]), tol=0)
chk("memorisers (ledger)", 43, float(K["n_memorisers"]), tol=0)

M = pd.read_csv(R4 / "fullgrid_meta.csv")
chk("primary experiment: points n", 529, float(M.n_points.median()), tol=0)
chk("primary experiment: classes K", 23, float(M.n_classes.median()), tol=0)
chk("primary experiment: points per class m", 23, float(M.points_per_class.median()), tol=0)
chk("primary experiment: distinct seeds", 9, len(M), tol=0)

ZA = pd.read_csv(R8 / "z59_run_audit.csv")
inv = ZA[ZA.in_corpus_inventory]
rec = ZA[ZA.in_recombination]
chk("Z59 corpus census runs", 12, len(inv), tol=0)
chk("Z59 corpus census distinct seeds", 2, inv.seed.nunique(), tol=0)
chk("Z59 corpus census grokked", 7, int(inv.grokked.sum()), tol=0)
chk("Z59 replication runs", 9, len(rec), tol=0)
chk("Z59 replication distinct seeds", 5, rec.seed.nunique(), tol=0)
chk("Z59 replication grokked", 9, int(rec.grokked.sum()), tol=0)
chk("Z59 replication runs also in the census", 6, int(rec.dir.isin(inv.dir).sum()), tol=0)
for t_, n_ in ((0.4, 2), (0.6, 2), (0.8, 5)):
    chk(f"Z59 replication runs at train fraction {t_:g}", n_,
        int((rec.train_fraction == t_).sum()), tol=0)
for name, sub in (("Z59 census", inv), ("Z59 replication", rec)):
    chk(f"{name}: runs counted twice", 0, int(sub.dir.duplicated().sum()), tol=0)

# ============================================================ figure inputs
print("\n=== figure-generating inputs ===")
import make_figures as F  # noqa: E402

rt = F.run_table().set_index("dir")
W = pd.read_csv(R / "v3_within_series.csv")
W = W[(W.layer == "last_hidden") & (W.dir.isin(rt.index))]
cnt = {"grokked": 0, "memoriser": 0, "random labels": 0}
for d_, _ in W.groupby("dir"):
    r_ = rt.loc[d_]
    cnt["random labels" if bool(r_.random_labels)
        else "grokked" if bool(r_.grokked) else "memoriser"] += 1
chk("within-class series: grokked runs", 53, cnt["grokked"], tol=0)
chk("within-class series: memorisers", 43, cnt["memoriser"], tol=0)
chk("within-class series: random-label runs", 3, cnt["random labels"], tol=0)
Sw = pd.read_csv(R4 / "fullgrid_sweep.csv")
Sw = Sw[Sw.pipeline == "raw"]
chk("Figure 1 shading: smallest alpha with L2 = 0 in every seed", 1.5,
    float(F.containment_alpha(Sw)), tol=0)

# ============================================================ centroid intervention (Sec. 3)
print("\n=== centroid intervention, complete input grid (Sec. 3, Figure 1) ===")
med = Sw.groupby("alpha")[["ph_dim", "twonn", "L2"]].median()
seg = {("ph_dim", 0.0, 1.0): 3.00, ("ph_dim", 1.0, 8.0): 3.47, ("ph_dim", 0.0, 8.0): 10.40,
       ("twonn", 0.0, 1.0): 1.18, ("twonn", 1.0, 8.0): 1.00, ("twonn", 0.0, 8.0): 1.18}
for (e, a, b), claimed in seg.items():
    chk(f"{e} fold alpha {a:g} -> {b:g} (ratio of seed medians)", claimed,
        med[e][a] / med[e][b], tol=0.006)
for e, a, v in (("ph_dim", 0.0, 15.5), ("ph_dim", 1.0, 5.2), ("ph_dim", 8.0, 1.5),
                ("twonn", 0.0, 17.4), ("twonn", 1.0, 14.8), ("twonn", 8.0, 14.8)):
    chk(f"{e} seed median at alpha = {a:g}", v, med[e][a], tol=0.06)
Pw = Sw.pivot_table(index="alpha", columns="seed", values="ph_dim")
for a, b, lo, hi in ((0.0, 1.0, 2.7, 3.3), (1.0, 8.0, 3.2, 3.7)):
    r_ = Pw.loc[a] / Pw.loc[b]
    chk(f"PH-dim per-seed fold {a:g} -> {b:g}, min", lo, r_.min(), tol=0.06)
    chk(f"PH-dim per-seed fold {a:g} -> {b:g}, max", hi, r_.max(), tol=0.06)
Tw = Sw.pivot_table(index="alpha", columns="seed", values="twonn")
rt_ = Tw.loc[0.0] / Tw.loc[1.0]
chk("TwoNN per-seed fold 0 -> 1, min", 1.02, rt_.min(), tol=0.006)
chk("TwoNN per-seed fold 0 -> 1, max", 1.28, rt_.max(), tol=0.006)
rel = (Tw.loc[8.0] - Tw.loc[1.0]).abs() / Tw.loc[1.0]
chk("largest per-seed relative TwoNN change above alpha = 1, %", 0.97, 100 * rel.max(), tol=0.06)
L2w = Sw.pivot_table(index="alpha", columns="seed", values="L2")
chk("seeds with L2 > 0 at alpha = 1", 3, int((L2w.loc[1.0] > 0).sum()), tol=0)
chk("seeds with L2 = 0 at alpha = 1.5", 9, int((L2w.loc[1.5] == 0).sum()), tol=0)
ph_ratio = Sw[Sw.alpha >= 1.5].groupby("seed").ph_dim.agg(["max", "min"])
ph_ratio["r"] = ph_ratio["max"] / ph_ratio["min"]
chk("PH-dim further fall over the contained stretch, min", 2.4, ph_ratio.r.min(), tol=0.02)
chk("PH-dim further fall over the contained stretch, max", 2.6, ph_ratio.r.max(), tol=0.03)
NC = pd.read_csv(R4 / "near_containment_exact.csv")
chk("TwoNN max abs deviation once L2 = 0 (units, x1e6)", 3.2,
    1e6 * NC[NC.estimator == "twonn"].max_abs_dev_after_zero.max(), tol=0.3)
chk("MLE-ID max abs deviation once Lq = 0 (units, x1e7)", 9.3,
    1e7 * NC[NC.estimator == "mle_k10"].max_abs_dev_after_zero.max(), tol=1.0)
B = pd.read_csv(R4 / "near_containment_bands.csv")
b = B[B.estimator == "twonn"].set_index("band")
chk("TwoNN median rel. deviation, 0 < L2 <= 0.01 (%)", 0.09,
    100 * b.loc["0 < Lq <= 0.01", "median_rel_dev"], tol=0.02)
chk("TwoNN max rel. deviation, 0 < L2 <= 0.01 (%)", 0.98,
    100 * b.loc["0 < Lq <= 0.01", "max_rel_dev"], tol=0.02)

print("\n=== exactness of the construction (Sec. 3, Appendix D.7) ===")
X = pd.read_csv(R4 / "exactness_summary.csv", index_col=0)["value"]
chk("same-class distance relative deviation (x1e7)", 2.7, 1e7 * X["c1_max_rel_dev"], tol=0.3)
chk("common-factor spread within a cloud (x1e7)", 7.4, 1e7 * X["c2_max_factor_spread"], tol=0.3)
chk("recombination diagonal vs checkpoint, PH-dim (x1e7)", 1.5, 1e7 * X["c4_max_ph_dim_dev"],
    tol=0.3)
chk("recombination diagonal vs checkpoint, TwoNN (x1e7)", 3.8, 1e7 * X["c4_max_twonn_dev"],
    tol=0.3)

# ============================================================ attribution (Sec. 4)
print("\n=== primary attribution, initialisation -> convergence (Sec. 4, Figure 2) ===")
Sh = pd.read_csv(R4 / "fullgrid_shapley.csv")
E = Sh[(Sh.pipeline == "raw") & (Sh.phase == "init->conv (endpoint)")]
P, T = E[E.estimator == "ph_dim"], E[E.estimator == "twonn"]
chk("PH-dim symmetric centroid share %", 90, 100 * P.between_share.median(), tol=0.6)
chk("  seed min %", 84, 100 * P.between_share.min(), tol=0.6)
chk("  seed max %", 91, 100 * P.between_share.max(), tol=0.6)
chk("PH-dim centroids-first share %", 123, 100 * P.centroid_first_share.median(), tol=0.6)
chk("  seed min %", 121, 100 * P.centroid_first_share.min(), tol=0.6)
chk("  seed max %", 124, 100 * P.centroid_first_share.max(), tol=0.6)
chk("PH-dim centroids-second share %", 58, 100 * P.centroid_second_share.median(), tol=0.6)
chk("  seed min %", 46, 100 * P.centroid_second_share.min(), tol=0.6)
chk("  seed max %", 59, 100 * P.centroid_second_share.max(), tol=0.6)
chk("seeds below half under the centroids-second order", 3,
    int((P.centroid_second_share <= 0.5).sum()), tol=0)
chk("PH-dim raw centroid contribution", 16.5, P.phi_between_raw.median(), tol=0.06)
chk("PH-dim raw residual contribution", 1.8, P.phi_within_raw.median(), tol=0.06)
for nm, v in (("f(C0,R0)", 23.5), ("f(C1,R0)", 1.0), ("f(C0,R1)", 15.5), ("f(C1,R1)", 5.2)):
    col = {"f(C0,R0)": "f00", "f(C1,R0)": "f10", "f(C0,R1)": "f01", "f(C1,R1)": "f11"}[nm]
    chk(f"PH-dim 2x2 cell {nm}", v, P[col].median(), tol=0.06)
chk("TwoNN symmetric centroid share %", -2, 100 * T.between_share.median(), tol=0.6)
chk("  seed min %", -9, 100 * T.between_share.min(), tol=0.6)
chk("  seed max %", 13, 100 * T.between_share.max(), tol=0.6)
chk("TwoNN diagonal change (units)", 12.2, T.diag.median(), tol=0.06)

G = pd.read_csv(R4 / "fullgrid_summary.csv")
g = G[(G.pipeline == "raw") & (G.phase == "init->conv (endpoint)")].set_index("estimator")
chk("PH-dim seed-bootstrap interval lo %", 86, 100 * g.loc["ph_dim", "boot_lo"], tol=0.6)
chk("PH-dim seed-bootstrap interval hi %", 91, 100 * g.loc["ph_dim", "boot_hi"], tol=0.6)
chk("TwoNN seed-bootstrap interval lo %", -6, 100 * g.loc["twonn", "boot_lo"], tol=0.6)
chk("TwoNN seed-bootstrap interval hi %", 6, 100 * g.loc["twonn", "boot_hi"], tol=0.6)
V = pd.read_csv(R4 / "fullgrid_vs_400.csv")
Ve = V[(V.phase == "init->conv (endpoint)") & (V.estimator == "ph_dim")]
chk("400-point subsample PH-dim share %", 89, 100 * Ve.between_share_train400.median(), tol=0.6)
chk("paired full-grid vs 400-point share difference (pp)", 0.1,
    abs(Ve.share_diff_pp.median()), tol=0.15)

print("\n=== endpoint interaction, per seed (Sec. 4, Table 3) ===")
EI = pd.read_csv(R8 / "endpoint_interaction.csv")
P8, T8 = EI[EI.estimator == "ph_dim"], EI[EI.estimator == "twonn"]
chk("PH-dim interaction I, median", 12.1, P8.interaction_raw.median(), tol=0.06)
chk("  seed min", 11.1, P8.interaction_raw.min(), tol=0.06)
chk("  seed max", 15.1, P8.interaction_raw.max(), tol=0.06)
chk("PH-dim diagonal change", 18.3, P8.diag.median(), tol=0.06)
chk("PH-dim I / diagonal, median", 0.66, P8.interaction_over_diag.median(), tol=0.006)
chk("seeds with centroid > residual (symmetric allocation)", 9,
    int((P8.phi_centroid_raw > P8.phi_residual_raw).sum()), tol=0)
chk("TwoNN interaction I", -4.44, T8.interaction_raw.median(), tol=0.06)
chk("TwoNN I / diagonal", -0.36, T8.interaction_over_diag.median(), tol=0.006)
for nm, col, v in (("f(C0,R0)", "f00", 26.5), ("f(C1,R0)", "f10", 28.9),
                   ("f(C0,R1)", "f01", 17.4), ("f(C1,R1)", "f11", 14.8)):
    chk(f"TwoNN 2x2 cell {nm}", v, T8[col].median(), tol=0.06)

print("\n=== four-segment checkpoint-partition allocation (Sec. 4, Table 4) ===")
CP = pd.read_csv(R8 / "checkpoint_path_summary.csv")
cp, ct = CP[CP.estimator == "ph_dim"], CP[CP.estimator == "twonn"]
chk("segment diagonals telescope to the endpoint diagonal", 0.0,
    float((cp.path_diag_raw - cp.endpoint_diag_raw).abs().max()), tol=1e-9)
chk("PH-dim four-segment centroid share %", 180, 100 * cp.path_share.median(), tol=0.6)
chk("  seed min %", 171, 100 * cp.path_share.min(), tol=0.6)
chk("  seed max %", 187, 100 * cp.path_share.max(), tol=0.6)
chk("TwoNN four-segment centroid share %", 28, 100 * ct.path_share.median(), tol=0.6)
chk("  seed min %", 11, 100 * ct.path_share.min(), tol=0.6)
chk("  seed max %", 38, 100 * ct.path_share.max(), tol=0.6)
chk("PH-dim centroid-dominant under both partitions (seeds)", 9,
    int(((cp.path_share > 0.5) & (cp.endpoint_share > 0.5)).sum()), tol=0)
chk("TwoNN residual-dominant under both partitions (seeds)", 9,
    int(((ct.path_share < 0.5) & (ct.endpoint_share < 0.5)).sum()), tol=0)

print("\n=== separation scale at the endpoints (Sec. 4, Appendix D.2) ===")
f_sr = RC / "scale_ratio.csv"
if f_sr.exists():
    SR = pd.read_csv(f_sr)
    SR = SR[SR.exact_reproduction.astype(bool)]
    chk("seeds used (exact reproductions)", 8, len(SR), tol=0)
    chk("centroid/residual RMS ratio at initialisation, median", 0.116,
        SR.ratio_init.median(), tol=0.0006)
    chk("  seed min", 0.114, SR.ratio_init.min(), tol=0.0006)
    chk("  seed max", 0.118, SR.ratio_init.max(), tol=0.0006)
    chk("centroid/residual RMS ratio at convergence, median", 2.58,
        SR.ratio_conv.median(), tol=0.006)
    chk("  seed min", 2.27, SR.ratio_conv.min(), tol=0.006)
    chk("  seed max", 3.11, SR.ratio_conv.max(), tol=0.006)
else:
    missing.append("scale ratio (RESULTS_camera_ready/scale_ratio.csv)")

print("\n=== first hidden layer (Appendix D.3) ===")
LA = pd.read_csv(R8 / "layer_audit.csv")
lp = LA[(LA.layer == "pre_logit") & (LA.estimator == "ph_dim")]
chk("layer audit reproduces the PH-dim diagonal", P8.diag.median(), lp.diag.median(), tol=0.02)
chk("layer audit reproduces the PH-dim interaction", P8.interaction_raw.median(),
    lp.interaction_raw.median(), tol=0.02)
chk("primary cell distinct seeds in the layer audit", 9, LA.seed.nunique(), tol=0)
lh = LA[(LA.layer == "hidden1") & (LA.estimator == "ph_dim")]
chk("first hidden layer, L2 at initialisation", 1.0, lh.L2_init.median(), tol=1e-9)
chk("first hidden layer, L2 at convergence", 1.0, lh.L2_conv.median(), tol=1e-9)
chk("first hidden layer, silhouette at convergence", 0.011, lh.sil_conv.median(), tol=0.002)
chk("first hidden layer, PH-dim share seed min %", -159, 100 * lh.symmetric_share.min(), tol=1.0)
chk("first hidden layer, PH-dim share seed max %", 349, 100 * lh.symmetric_share.max(), tol=1.0)

print("\n=== MLP replications (Sec. 4, Appendix D.4, Table 5) ===")
XC = pd.read_csv(R / "v3_crosscell_summary.csv")
XC = XC[XC.frame == "unaligned"].set_index(["cell", "estimator"])
chk("S4 PH-dim centroid share %", 89, 100 * XC.loc[("S4_perm_mlp", "ph_dim"), "between_share"],
    tol=0.6)
chk("S4 TwoNN centroid share %", -13, 100 * XC.loc[("S4_perm_mlp", "twonn"), "between_share"],
    tol=0.6)
chk("S4 runs", 5, int(XC.loc[("S4_perm_mlp", "ph_dim"), "n_runs"]), tol=0)
Z59 = pd.read_csv(R8 / "z59_by_train_fraction.csv")
Z59["share"] = Z59.phi_centroid_raw / Z59.diag
zs = Z59.groupby(["estimator", "train_fraction"]).share.median()
for tf, v in ((0.4, 106), (0.6, 103), (0.8, 105)):
    chk(f"Z59 PH-dim centroid share at train fraction {tf:g} %", v,
        100 * zs[("ph_dim", tf)], tol=0.6)
tw59 = 100 * zs["twonn"]
chk("Z59 TwoNN centroid share by fraction, min %", 33, tw59.min(), tol=0.6)
chk("Z59 TwoNN centroid share by fraction, max %", 37, tw59.max(), tol=0.6)

# ============================================================ estimator support (Sec. 5)
print("\n=== PH-dimension mixture account on real clouds (Sec. 5, Figure 4, Table 9) ===")
RM = pd.read_csv(R8 / "real_mst_summary.csv")
hold, fail = RM[RM.alpha >= 1.0], RM[RM.alpha <= 0.5]
chk("cross-class MST edges / (K-1) at alpha >= 1, max", 1.05,
    float(hold.cross_over_Km1.max()), tol=0.006)
chk("residual d_hat - closed form at alpha >= 1, median", -0.21, float(hold.resid.median()),
    tol=0.006)
chk("  seed-alpha min", -0.46, float(hold.resid.min()), tol=0.006)
chk("  seed-alpha max", 0.13, float(hold.resid.max()), tol=0.006)
chk("cross-class MST edges / (K-1) at alpha <= 0.5, min", 1.87,
    float(fail.cross_over_Km1.min()), tol=0.02)
chk("  max", 14.07, float(fail.cross_over_Km1.max()), tol=0.02)
MX = pd.read_csv(R / "v3_mixture_grid.csv").dropna(subset=["d_hat", "d_pred_closedform"])
r2 = 1 - (((MX.d_hat - MX.d_pred_closedform) ** 2).sum()
          / ((MX.d_hat - MX.d_hat.mean()) ** 2).sum())
chk("synthetic mixture: closed-form R^2", 0.92, r2, tol=0.005)
chk("synthetic mixture: grid cells", 288, len(MX), tol=0)

print("\n=== GRIDE forced-leakage order and crossover (Appendix E.4, Table 10) ===")
C = pd.read_csv(R4 / "gride_real_crossover_dense.csv")
for pop, forced, leak, cross in (("fullgrid", 12, 12, 8), ("train400", 7, 7, 6)):
    q = C[C.population == pop]
    chk(f"{pop}: forced order (min over classes)", forced, float(q.forced_n1_min.median()), tol=0)
    chk(f"{pop}: first measured leakage", leak, float(q.n1_first_leak_measured.median()), tol=0)
    chk(f"{pop}: response crossover", cross, float(q.n1_cross_measured.median()), tol=0)
SY = pd.read_csv(R4 / "gride_synth_crossover_dense.csv")
for m, forced, leak in ((8, 4, 4), (16, 8, 8), (32, 16, 16)):
    q = SY[SY.m == m]
    chk(f"synthetic m={m}: forced order", forced, float(q.n1_forced_corrected.median()), tol=0)
    chk(f"synthetic m={m}: first leakage", leak, float(q.n1_first_leak_measured.median()), tol=0)

# ============================================================ group structure (App. F)
print("\n=== separation, centroid rank and group structure (Appendix F) ===")
DS = pd.read_csv(R / "v3_dissociate_summary.csv").set_index("family")
chk("equivariance order parameter, Fourier torus", 1.00, DS.loc["fourier3", "equiv_R2"],
    tol=0.006)
chk("equivariance order parameter, permuted", 0.025, DS.loc["fourier3_perm", "equiv_R2"],
    tol=0.002)
chk("PH-dim, Fourier torus", 3.04, DS.loc["fourier3", "ph_dim"], tol=0.006)
chk("PH-dim, permuted", 3.04, DS.loc["fourier3_perm", "ph_dim"], tol=0.006)

# ============================================================ Transformer, Brown-exact rule
print("\n=== Transformer: Brown-exact TwoNN fitting rule (Appendix D.6) ===")
A = pd.read_csv(R4 / "brown_exact_attribution.csv")
piv = A.pivot_table(index="seed", columns="rule", values="within_share")
chk("all seeds residual-dominated under both fitting rules", 1,
    int(bool((piv[["paper", "brown_min"]] > 0.5).all().all())), tol=0)
BH = pd.read_csv(R8 / "brown_hybrid_leakage.csv")
bg = BH[BH.grokked]
chk("grokked confirmation seeds", 9, len(bg), tol=0)
chk("seeds contained in both clouds of the centroids-second leg", 4,
    int(((bg.L2_Cplateau_Rconv == 0) & (bg.L2_Cconv_Rconv == 0)).sum()), tol=0)
chk("L2 at f(C_conv, R_conv), max", 0.0, float(bg.L2_Cconv_Rconv.max()), tol=1e-9)
chk("L2 at the hybrid f(C_plateau, R_conv), max", 0.0269, float(bg.L2_Cplateau_Rconv.max()),
    tol=0.0006)

# ============================================================ summary
print("\n" + "=" * 72)
if bad:
    print(f"FAIL: {len(bad)} mismatch(es): " + "; ".join(bad))
if missing:
    print(f"NOTE: {len(missing)} value(s) could not be recomputed: " + ", ".join(missing))
if not bad and not missing:
    print("ALL HEADLINE NUMBERS (MLP, intervention, attribution, mechanism) VERIFIED")

print("\n### Transformer numbers: paper/check_transformer_numbers.py ###\n", flush=True)
rc = subprocess.call([sys.executable, str(REPO / "paper" / "check_transformer_numbers.py")])
sys.exit(1 if (bad or missing or rc) else 0)
