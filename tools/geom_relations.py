"""
STEP 3 -- the relations. This is the intended novel core, so it is written to be falsifiable.

R1  HOW MANY INDEPENDENT AXES? Factor the measures (correlation -> PCA on within-run z-scores,
    so a shared monotone trend in training time cannot manufacture a single giant factor).
    Report variance explained and which measures load together. Names the redundancies.

R2  THE CENTRAL CONTRAST (wedge W1): does geometry move where homology does not, ON THE SAME
    checkpoints? Quantified as total within-run |z| excursion of each measure, with ambient
    H1 carried through the identical pipeline as the null-by-construction reference.

R3  DIMENSION vs CLUSTERING (from GATE 7/11): ph_dim and MLE-k20 are driven to ~1 and ~0 by
    clustering alone. So for every dimension measure, regress its within-run change on the
    clusteredness change and ask what survives. A measure whose motion is fully explained by
    gap_ratio is reporting clustering, not dimension -- which is the critique that applies to
    the "dimensionality collapse precedes generalization" literature.

R4  FRACTAL vs LINEAR DIVERGENCE (wedge W2): track (linear ID - local/fractal ID) through the
    transition, having verified on the Swiss roll that the estimators DO reproduce a known
    divergence (+0.82) when one exists.

R5  everything against generalization and against equiv_R2.

Discovery/confirmation split is respected: patterns are found on discovery and only then
evaluated on confirmation.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import spearmanr

OUT = REPO / "RESULTS"

DIMS_FRACTAL = ["ph_dim", "corr_dim"]
DIMS_LOCAL = ["twonn", "mle_id_k5", "mle_id_k10"]
DIMS_LINEAR = ["pca_pr", "eff_rank"]
CLUSTER = ["gap_ratio", "nn1_ratio", "nc1", "nc_wb"]
SHAPE = ["lid_sd", "lid_skew", "lid_bimod", "spec_decay"]
OTHER = ["orc_mean", "orc_frac_neg", "mag_t1", "mag_t5", "nc2_etf"]
HOM = ["amb_H1_max"]
ANCHOR = ["equiv_R2"]
ALL = DIMS_FRACTAL + DIMS_LOCAL + DIMS_LINEAR + CLUSTER + SHAPE + OTHER + HOM + ANCHOR


def zwithin(D, cols):
    """z-score each measure WITHIN each run, so cross-run scale differences cannot dominate."""
    Z = D.copy()
    for c in cols:
        if c in Z:
            Z[c] = Z.groupby("dir")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return Z


if __name__ == "__main__":
    D = pd.read_csv(OUT / "geom_series.csv").sort_values(["dir", "epoch"])
    cols = [c for c in ALL if c in D.columns]
    pd.set_option("display.width", 220)
    print(f"loaded {D.shape}: {D.dir.nunique()} runs, {len(cols)} measures\n")

    # ---------------------------------------------------------------- R2 first: excursion
    print("=== R2: does the measure actually MOVE? (geometry vs the homological reference) ===")
    print("    (max-min)/sd was tried first and discarded: it returns ~4 for ANY noisy 15-point")
    print("    series, so it scored ambient H1 the same as every geometric measure. Replaced by")
    print("    two statistics that have a null which can fire:")
    print("      trend  |spearman(value, epoch)|, against a TIME-SHUFFLED null (-> ~0)")
    print("      snr    |net change| / (1.4826 * median |step-to-step difference|)\n")
    rng = np.random.default_rng(0)
    exc = []
    for c in cols:
        tr, sn, nul = [], [], []
        for _, g in D.groupby("dir"):
            s = g[[c, "epoch"]].dropna().sort_values("epoch")
            if len(s) < 6 or s[c].std() == 0:
                continue
            v = s[c].values
            tr.append(abs(spearmanr(v, s.epoch.values).statistic))
            d = np.abs(np.diff(v))
            md = np.median(d)
            sn.append(abs(v[-1] - v[0]) / (1.4826 * md) if md > 0 else np.nan)
            nul.append(abs(spearmanr(rng.permutation(v), s.epoch.values).statistic))
        if tr:
            exc.append({"measure": c, "n": len(tr), "trend": float(np.median(tr)),
                        "trend_null": float(np.median(nul)),
                        "snr": float(np.nanmedian(sn)),
                        "family": ("fractal" if c in DIMS_FRACTAL else "local" if c in DIMS_LOCAL
                                   else "linear" if c in DIMS_LINEAR else "cluster" if c in CLUSTER
                                   else "shape" if c in SHAPE else "homology" if c in HOM
                                   else "anchor" if c in ANCHOR else "other")})
    E = pd.DataFrame(exc).sort_values("trend", ascending=False).round(3)
    print(E.to_string(index=False))
    E.to_csv(OUT / "geom_excursion.csv", index=False)

    # ---------------------------------------------------------------- R1 factor structure
    print("\n=== R1: how many INDEPENDENT axes? (PCA on within-run z-scores) ===")
    Z = zwithin(D, cols)
    disc = Z[Z.split == "discovery"][cols].dropna()
    if len(disc) > 30:
        from sklearn.decomposition import PCA
        A = disc.values
        A = (A - A.mean(0)) / (A.std(0) + 1e-12)
        pca = PCA().fit(A)
        ev = pca.explained_variance_ratio_
        print("  variance explained: " + "  ".join(f"PC{i+1}={x:.2f}" for i, x in enumerate(ev[:6])))
        print(f"  components for 90% variance: {int(np.searchsorted(np.cumsum(ev), 0.90) + 1)}"
              f"  (of {len(cols)} measures)")
        L = pd.DataFrame(pca.components_[:3].T, index=disc.columns, columns=["PC1", "PC2", "PC3"])
        print("\n  loadings (|.|>0.25 shown):")
        for pc in ("PC1", "PC2", "PC3"):
            s = L[pc][L[pc].abs() > 0.25].sort_values()
            print(f"    {pc}: " + ", ".join(f"{i}{v:+.2f}" for i, v in s.items()))
        L.to_csv(OUT / "geom_loadings.csv")

        print("\n  redundancy: |spearman| > 0.9 pairs on discovery")
        C = disc.corr(method="spearman").abs()
        seen = set()
        for a in C.columns:
            for b in C.columns:
                if a < b and C.loc[a, b] > 0.9:
                    print(f"    {a:12s} ~ {b:12s} {C.loc[a,b]:.3f}")
                    seen.add((a, b))
        if not seen:
            print("    (none)")

    # ---------------------------------------------------------------- R3 dim vs clustering
    print("\n=== R3: is the dimension drop explained by CLUSTERING? ===")
    print("    per-run spearman of each dimension measure vs gap_ratio, and vs test_acc\n")
    rows = []
    for c in DIMS_FRACTAL + DIMS_LOCAL + DIMS_LINEAR:
        if c not in D:
            continue
        rg, rt, rp = [], [], []
        for _, g in D.groupby("dir"):
            s = g[[c, "gap_ratio", "test_acc"]].dropna()
            if len(s) < 6:
                continue
            rg.append(spearmanr(s[c], s.gap_ratio).statistic)
            rt.append(spearmanr(s[c], s.test_acc).statistic)
            # partial: correlation of residuals after removing gap_ratio (rank-linear)
            x = s[c].rank(); z = s.gap_ratio.rank(); yv = s.test_acc.rank()
            bx = np.polyfit(z, x, 1); by = np.polyfit(z, yv, 1)
            ex = x - np.polyval(bx, z); ey = yv - np.polyval(by, z)
            if ex.std() > 1e-9 and ey.std() > 1e-9:
                rp.append(np.corrcoef(ex, ey)[0, 1])
        rows.append({"measure": c, "n": len(rg),
                     "rho_vs_clustering": np.median(rg), "rho_vs_testacc": np.median(rt),
                     "partial_vs_testacc_given_clustering": np.median(rp) if rp else np.nan})
    R3 = pd.DataFrame(rows).round(3)
    print(R3.to_string(index=False))
    R3.to_csv(OUT / "geom_dim_vs_cluster.csv", index=False)

    # ---------------------------------------------------------------- R4 divergence
    print("\n=== R4: fractal/local vs linear dimension divergence through the transition ===")
    D["dim_local"] = D[[c for c in DIMS_LOCAL if c in D]].mean(axis=1)
    D["dim_linear"] = D[[c for c in DIMS_LINEAR if c in D]].mean(axis=1)
    D["divergence"] = D.dim_linear - D.dim_local
    D["ratio_lin_loc"] = D.dim_linear / D.dim_local.replace(0, np.nan)
    D["side"] = np.where(D.t < D.t_grok, "pre", np.where(D.t > D.t_grok, "post", "at"))
    g = D[D.t_grok >= 0].groupby(["split", "side"]).agg(
        n=("divergence", "count"), dim_local=("dim_local", "median"),
        dim_linear=("dim_linear", "median"), divergence=("divergence", "median"),
        ratio=("ratio_lin_loc", "median")).round(3)
    print(g.to_string())

    # ---------------------------------------------------------------- R5 vs generalization
    print("\n=== R5: relation to generalization and to equiv_R2 (within-run spearman medians) ===")
    rows = []
    for c in cols:
        if c in ("equiv_R2",):
            continue
        ra, re = [], []
        for _, gg in D.groupby("dir"):
            s = gg[[c, "test_acc", "equiv_R2"]].dropna()
            if len(s) < 6 or s[c].std() == 0:
                continue
            ra.append(spearmanr(s[c], s.test_acc).statistic)
            re.append(spearmanr(s[c], s.equiv_R2).statistic)
        if ra:
            rows.append({"measure": c, "n": len(ra), "rho_test_acc": np.median(ra),
                         "rho_equiv_R2": np.median(re)})
    R5 = pd.DataFrame(rows).round(3).sort_values("rho_test_acc")
    print(R5.to_string(index=False))
    R5.to_csv(OUT / "geom_vs_generalization.csv", index=False)
    D.to_csv(OUT / "geom_series_derived.csv", index=False)
