"""
v6.4 -- one authoritative run ledger for every claim the paper makes.

Every displayed count in v6.4 is derived from this file. Nothing is hand-counted in LaTeX.

The v6.3 text carried four counts that appeared to disagree (96 MLP runs / 54 grokked in the
corpus table; 53 grokked + 44 memorisers in the within-class analysis; 49 grokked + 40
non-grokked at the operating point; a per-cell silhouette table whose grokked rows sum to 54).
This script recomputes all of them from source and writes the resolution, so the disagreement
is settled by the data rather than by editing one number.

Grokking rule, applied uniformly here: a run is grokked when its FINAL TEST ACCURACY, defined
as the mean over the last five checkpoints, reaches 0.9. That is the same "final value"
definition the paper already uses for t_grok, so v6.4 uses one definition throughout.
RESULTS/_inventory.csv instead stores the LAST-CHECKPOINT test accuracy, which is what
produced the 54 count; exactly one run differs between the two rules.

Analysis-only. Reads RESULTS/, writes RESULTS_v6_4/.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

SRC = REPO / "RESULTS"
OUT = REPO / "RESULTS_v6_4"
OUT.mkdir(exist_ok=True)

GROK_THRESH = 0.9
MEMORISER_TRAIN = 0.95


# --------------------------------------------------------------------------- corpus census
def corpus_census():
    """MLP corpus under the two competing grokking rules, and the runs that differ."""
    inv = pd.read_csv(SRC / "_inventory.csv")
    t11 = pd.read_csv(SRC / "v3_t11_run_table.csv")
    mlp = inv[inv.arch == "mlp"].copy()
    real = t11[~t11.random_labels.astype(bool)].copy()          # excludes random-label controls
    rand = t11[t11.random_labels.astype(bool)].copy()

    m = mlp.merge(real[["dir", "fin_test", "fin_train", "grokked"]], on="dir", how="left")
    m["grok_last_ckpt"] = m.test >= GROK_THRESH                 # inventory rule
    m["grok_mean5"] = m.fin_test >= GROK_THRESH                 # paper's stated rule
    disagree = m[m.grok_last_ckpt != m.grok_mean5]

    memoriser = (~m.grok_mean5) & (m.fin_train >= MEMORISER_TRAIN)
    return dict(
        n_mlp=len(m),
        n_grok_last=int(m.grok_last_ckpt.sum()),
        n_grok_mean5=int(m.grok_mean5.sum()),
        n_memoriser=int(memoriser.sum()),
        n_nongrok=int((~m.grok_mean5).sum()),
        n_randlabel=len(rand),
        disagree=disagree[["dir", "test", "fin_test", "fin_train"]],
        table=m,
    )


def operating_point_census(m):
    """The 89-run operating-point subset, and which MLP runs are absent from it."""
    op = pd.read_csv(SRC / "operating_point.csv")
    corpus = op[op.source == "corpus"].copy()
    tf = pd.read_csv(SRC / "bridge_tf.csv")
    have = corpus.merge(m[["dir", "grok_last_ckpt", "grok_mean5", "fin_test"]],
                        on="dir", how="left")
    missing = m[~m.dir.isin(set(corpus.dir))]
    return dict(
        n_op=len(corpus),
        n_grok_last=int(have.grok_last_ckpt.sum()),
        n_grok_mean5=int(have.grok_mean5.sum()),
        n_missing=len(missing),
        missing=missing[["dir", "test", "fin_test"]],
        have=have,
        n_tf_bridge=int(tf.dir.nunique()) if "dir" in tf.columns else np.nan,
    )


def silhouette_table(have):
    """Per-cell final silhouette of grokked runs, under the single mean-5 rule."""
    g = have[have.grok_mean5.fillna(False)]
    rows = []
    for (task, p), q in g.groupby(["task", "p"]):
        rows.append({"architecture": "MLP", "task": task, "p": int(p), "runs": len(q),
                     "median": float(q.sil_final.median()),
                     "lo": float(q.sil_final.min()), "hi": float(q.sil_final.max())})
    return pd.DataFrame(rows).sort_values(["task", "p"])


# --------------------------------------------------------------------------- ledger rows
def ledger():
    cc = corpus_census()
    m = cc["table"]
    op = operating_point_census(m)

    inv = pd.read_csv(SRC / "_inventory.csv")
    from tools.b3_seed_alpha import select
    prim = select(inv)
    prim_seeds = sorted(int(s) for s in prim.seed)

    C = pd.read_csv(SRC / "c1_recombine.csv")
    XC = pd.read_csv(SRC / "v3_crosscell.csv")
    BM = pd.read_csv(SRC / "brown_transformer_run_manifest.csv")
    t11 = pd.read_csv(SRC / "v3_t11_run_table.csv")

    bp = BM[BM.tag == "pilot"]
    bp2 = BM[BM.tag == "pilot2"]
    bc = BM[BM.tag == "confirm"]

    R = []

    def row(**kw):
        base = dict(claim="", arch="", task="", group="", config="", train_fraction="",
                    n_candidate="", n_included="", n_excluded="", exclusion_reason="",
                    seeds="", n_grokked="", n_nongrokked="", n_memoriser="",
                    input_set="", n_points="", points_per_class="", checkpoints="",
                    statistic="", implementation="", source_file="", consumed_by="")
        base.update(kw)
        R.append(base)

    # ---- primary experiment ------------------------------------------------
    row(claim="primary_attribution_endpoint", arch="MLP", task="modular addition",
        group="Z_23", config="2x200 hidden, bias=False, full batch, lr 1e-2, wd 0.03",
        train_fraction="0.8",
        n_candidate=int(((inv.arch == "mlp") & (inv.task == "add") & (inv.p == 23)
                         & (inv.tf == 0.8)).sum()),
        n_included=len(prim),
        n_excluded=int(((inv.arch == "mlp") & (inv.task == "add") & (inv.p == 23)
                        & (inv.tf == 0.8)).sum()) - len(prim),
        exclusion_reason="batch-size variants (bs != 0) and duplicate logging schedules of the "
                         "same seed are pseudo-replicates; one run per distinct seed at the "
                         "full-batch configuration is kept (tools/b3_seed_alpha.py:select)",
        seeds=";".join(str(s) for s in prim_seeds),
        n_grokked=len(prim), n_nongrokked=0, n_memoriser=0,
        input_set="fixed subsample of 400 training inputs, seeded rng(0)",
        n_points=400, points_per_class="17.4 (400/23, unequal)",
        checkpoints="5: init, pre-grok, t_grok, post-grok midpoint, convergence",
        statistic="PH-dimension, TwoNN, +battery", implementation="analysis/geom.py",
        source_file="RESULTS/c1_recombine.csv", consumed_by="Sec. attribution; Fig. 2a")

    row(claim="primary_attribution_fullgrid", arch="MLP", task="modular addition",
        group="Z_23", config="as above", train_fraction="0.8",
        n_candidate=len(prim), n_included=len(prim), n_excluded=0, exclusion_reason="none",
        seeds=";".join(str(s) for s in prim_seeds),
        n_grokked=len(prim), n_nongrokked=0, n_memoriser=0,
        input_set="complete Z_23 x Z_23 input grid (train and held-out inputs)",
        n_points=529, points_per_class="23 (exact)",
        checkpoints="same 5 definitions",
        statistic="PH-dimension, TwoNN", implementation="analysis/geom.py",
        source_file="RESULTS_v6_4/fullgrid_recombine.csv",
        consumed_by="Sec. attribution (primary); Fig. 2a")

    row(claim="centroid_intervention_sweep", arch="MLP", task="modular addition", group="Z_23",
        config="converged checkpoint only", train_fraction="0.8",
        n_candidate=len(prim), n_included=len(prim), n_excluded=0, exclusion_reason="none",
        seeds=";".join(str(s) for s in prim_seeds), n_grokked=len(prim),
        input_set="fixed subsample of 400 training inputs",
        n_points=400, points_per_class="17.4",
        checkpoints="convergence", statistic="alpha sweep, 13 points",
        implementation="tools/b3_seed_alpha.py",
        source_file="RESULTS/b3_seed_alpha.csv", consumed_by="Sec. intervention; Fig. 1")

    # ---- MLP replications --------------------------------------------------
    for cell, task, grp, tf_ in (("S4_perm_mlp", "permutation composition", "S_4", "0.8"),
                                 ("Z59_add_mlp", "modular addition", "Z_59", "0.4")):
        q = XC[XC.cell == cell]
        n = q.dir.nunique()
        p = 24 if grp == "S_4" else 59
        k = 24 if grp == "S_4" else 59
        row(claim="mlp_replication", arch="MLP", task=task, group=grp,
            config="2x200 hidden, full batch", train_fraction=tf_,
            n_candidate=n, n_included=n, n_excluded=0,
            exclusion_reason="grokked runs of the cell with retained endpoint checkpoints",
            seeds=";".join(str(int(s)) for s in sorted(q.seed.unique())),
            n_grokked=n, n_nongrokked=0,
            input_set="fixed subsample of 400 training inputs",
            n_points=400, points_per_class=f"{400/k:.1f} (400/{k})",
            checkpoints="2: init, convergence",
            statistic="PH-dimension, TwoNN, +battery", implementation="analysis/geom.py",
            source_file="RESULTS/v3_crosscell.csv, v3_crosscell_summary.csv",
            consumed_by="Sec. attribution (replication); Fig. 2a")

    # ---- Transformer -------------------------------------------------------
    row(claim="transformer_pilot", arch="Transformer (2 layer, 4 head, d=128)",
        task="modular division", group="Z_97",
        config="round 1: AdamW loop (Sea-Snell); round 2: coupled-decay Adam (Brown-corrected)",
        train_fraction="0.15",
        n_candidate=len(bp) + len(bp2), n_included=len(bp) + len(bp2), n_excluded=0,
        exclusion_reason="none; both pilot rounds reported",
        seeds="round1: 0;1 at wd 0.001 and wd 0.01 | round2: 0;1 at wd 0.001 and wd 0.01",
        n_grokked=int(BM[BM.tag.isin(["pilot", "pilot2"])].grokked.sum()),
        n_nongrokked=int((~BM[BM.tag.isin(["pilot", "pilot2"])].grokked).sum()),
        input_set="complete validation set", n_points=int(bc.n_val.iloc[0]),
        points_per_class=f"{bc.n_val.iloc[0]/97:.1f} (n_val/97)",
        checkpoints="configuration selection only; enters no reported interval",
        statistic="—", implementation="tools/brown_transformer/brown_train.py",
        source_file="RESULTS/brown_transformer_run_manifest.csv",
        consumed_by="Appendix (configuration selection)")

    row(claim="transformer_confirmation", arch="Transformer (2 layer, 4 head, d=128)",
        task="modular division", group="Z_97",
        config="coupled-decay Adam, wd 0.001 (selected setting)", train_fraction="0.15",
        n_candidate=len(bc), n_included=len(bc), n_excluded=0,
        exclusion_reason="none; all ten seeds reported including the one that did not grok",
        seeds=";".join(str(int(s)) for s in sorted(bc.seed)),
        n_grokked=int(bc.grokked.sum()), n_nongrokked=int((~bc.grokked).sum()),
        input_set="complete validation set (7916 points)", n_points=int(bc.n_val.iloc[0]),
        points_per_class=f"{bc.n_val.iloc[0]/97:.1f}",
        checkpoints="init, memorisation plateau, t_grok, convergence",
        statistic="PH-dimension, TwoNN", implementation="analysis/geom.py; "
                  "tools/brown_transformer/brown_analyze.py",
        source_file="RESULTS/brown_transformer_attribution_summary.csv, "
                    "brown_transformer_recombination_raw.csv",
        consumed_by="Sec. attribution (boundary); Fig. 2a; Appendix")

    # ---- corpus-level analyses --------------------------------------------
    row(claim="operating_point", arch="MLP", task="all corpus tasks", group="mixed",
        config="corpus", train_fraction="0.3-0.9",
        n_candidate=cc["n_mlp"], n_included=op["n_op"], n_excluded=op["n_missing"],
        exclusion_reason="no saved final-activation measurement for the 7 later __ss777 runs",
        seeds="corpus-wide; see RESULTS/operating_point.csv",
        n_grokked=op["n_grok_mean5"], n_nongrokked=op["n_op"] - op["n_grok_mean5"],
        input_set="pre-logit activations, 400-point training subsample",
        n_points=400, points_per_class="varies by K",
        checkpoints="init, t_grok, convergence",
        statistic="label silhouette", implementation="sklearn silhouette_score",
        source_file="RESULTS/operating_point.csv",
        consumed_by="Sec. setup; Appendix operating point")

    row(claim="within_class_series", arch="MLP", task="all corpus tasks", group="mixed",
        config="corpus + 3 random-label controls", train_fraction="0.3-0.9",
        n_candidate=len(t11), n_included=len(t11), n_excluded=0,
        exclusion_reason="none",
        seeds="corpus-wide; see RESULTS/v3_t11_run_table.csv",
        n_grokked=cc["n_grok_mean5"], n_nongrokked=cc["n_nongrok"],
        n_memoriser=cc["n_memoriser"],
        input_set="pre-logit activations, full trajectory",
        n_points=400, points_per_class="varies by K",
        checkpoints="all retained (76-401 per run)",
        statistic="within-class effective rank", implementation="analysis/geom.py",
        source_file="RESULTS/v3_t11_run_table.csv, v3_within_series.csv",
        consumed_by="Appendix within-class boundary")

    row(claim="random_label_control", arch="MLP", task="modular addition (shuffled labels)",
        group="Z_23", config="corpus", train_fraction="0.8",
        n_candidate=cc["n_randlabel"], n_included=cc["n_randlabel"], n_excluded=0,
        exclusion_reason="none", seeds="see RESULTS/v3_t11_run_table.csv",
        n_grokked=0, n_nongrokked=cc["n_randlabel"], n_memoriser=cc["n_randlabel"],
        input_set="pre-logit activations", n_points=400, points_per_class="17.4",
        checkpoints="all retained", statistic="within-class effective rank",
        implementation="analysis/geom.py",
        source_file="RESULTS/v2_randlabel.csv, v3_t11_run_table.csv",
        consumed_by="Appendix within-class boundary")

    # ---- mechanism / synthetic --------------------------------------------
    mg = pd.read_csv(SRC / "v3_mixture_grid.csv")
    row(claim="ph_mixture_closed_form", arch="—", task="synthetic mixture", group="—",
        config="grid over (d, K, m, gamma)", train_fraction="—",
        n_candidate=len(mg), n_included=len(mg), n_excluded=0,
        exclusion_reason="none", seeds="synthetic grid cells, NOT training replicates",
        input_set="synthetic separated mixtures",
        n_points="varies", points_per_class="m in grid",
        checkpoints="—", statistic="PH-dimension vs closed form",
        implementation="tools/v3_t13_mixture.py",
        source_file="RESULTS/v3_mixture_grid.csv",
        consumed_by="Sec. mechanism; Fig. 3a,b")

    gr = pd.read_csv(SRC / "b1_gride_ratio.csv")
    row(claim="gride_support_crossover", arch="—", task="synthetic mixture + real activations",
        group="—", config="m in {8,16,32,64}", train_fraction="—",
        n_candidate=len(gr), n_included=len(gr), n_excluded=0, exclusion_reason="none",
        seeds="synthetic grid cells, NOT training replicates",
        input_set="synthetic mixtures; real 400-point clouds",
        n_points="varies", points_per_class="8/16/32/64 synthetic; 17.4 real",
        checkpoints="—", statistic="GRIDE at order n1",
        implementation="tools/b1_gride.py",
        source_file="RESULTS/b1_gride_ratio.csv, RESULTS_v6_4/gride_threshold.csv",
        consumed_by="Sec. mechanism; Fig. 3c")

    ds = pd.read_csv(SRC / "v3_dissociate_summary.csv")
    row(claim="symmetry_permutation_control", arch="—", task="synthetic group-structured cloud",
        group="Z_p Fourier construction", config="fourier3 vs fourier3_perm",
        train_fraction="—",
        n_candidate=len(ds), n_included=2, n_excluded=len(ds) - 2,
        exclusion_reason="the permutation contrast uses the fourier3 / fourier3_perm pair; "
                         "other families are context",
        seeds="construction families, NOT training replicates",
        input_set="synthetic", n_points="—", points_per_class="—", checkpoints="—",
        statistic="equivariance order parameter, PH-dim, TwoNN, effective rank",
        implementation="tools/v3_t12_dissociate.py",
        source_file="RESULTS/v3_dissociate_summary.csv",
        consumed_by="Discussion (NeurReps implication); Appendix")

    return pd.DataFrame(R), cc, op


FIELD_LABEL = [
    ("arch", "architecture"), ("task", "task"), ("group", "group / prime"),
    ("config", "configuration"), ("train_fraction", "train fraction"),
    ("n_candidate", "candidate runs"), ("n_included", "included"), ("n_excluded", "excluded"),
    ("exclusion_reason", "exclusion reason"), ("seeds", "seed identifiers"),
    ("n_grokked", "grokked"), ("n_nongrokked", "non-grokked"), ("n_memoriser", "memorisers"),
    ("input_set", "input set measured"), ("n_points", "points"),
    ("points_per_class", "points per class"), ("checkpoints", "checkpoints / interval"),
    ("statistic", "statistic"), ("implementation", "implementation"),
    ("source_file", "source file"), ("consumed_by", "consumed by"),
]


def write_markdown(L, cc, op, sil, path):
    """Readable rendering of the same rows the CSV holds. Never hand-edited."""
    out = ["# v6.4 claim / run ledger", "",
           "Machine-generated by `tools/v6_4/build_ledger.py`. Every count displayed in the",
           "paper is derived from this file; no count is maintained by hand in LaTeX.", "",
           "**Grokking rule.** A run is grokked when its final test accuracy, defined as the",
           "mean over the last five checkpoints, reaches 0.9. This is the same \"final value\"",
           "definition the paper uses for $t_{\\text{grok}}$, applied uniformly.", "",
           "**Unit of replication.** The training seed. Checkpoints, subsamples, points and",
           "synthetic grid cells are never treated as independent training replicates.", "",
           "---", "", "## Count resolution", "",
           "| quantity | value |", "| --- | --- |"]
    for k, v in {
        "MLP corpus runs": cc["n_mlp"],
        "grokked (mean-of-last-five rule, used in v6.4)": cc["n_grok_mean5"],
        "grokked (last-checkpoint rule, used by _inventory.csv)": cc["n_grok_last"],
        "non-generalising memorisers": cc["n_memoriser"],
        "random-label controls": cc["n_randlabel"],
        "runs in the operating-point measurement": op["n_op"],
        "…of which grokked (v6.4 rule)": op["n_grok_mean5"],
        "MLP runs with no saved final-activation measurement": op["n_missing"],
    }.items():
        out.append(f"| {k} | {v} |")
    out += ["", "Runs whose grokked flag depends on the rule:", "", "```"]
    out.append(cc["disagree"].to_string(index=False))
    out += ["```", "", "MLP runs absent from the operating-point measurement:", "", "```"]
    out.append(op["missing"].to_string(index=False))
    out += ["```", "",
            "Final label-silhouette of grokked MLP runs, by cell, under the single rule:",
            "", "```", sil.to_string(index=False), "```", "", "---", "", "## Rows", ""]

    for _, r in L.iterrows():
        out += [f"### `{r['claim']}`", "", "| field | value |", "| --- | --- |"]
        for key, label in FIELD_LABEL:
            v = r[key]
            if v == "" or (isinstance(v, float) and pd.isna(v)):
                continue
            out.append(f"| {label} | {v} |")
        out.append("")
    Path(path).write_text("\n".join(out), encoding="utf-8")


def main():
    L, cc, op = ledger()
    L.to_csv(OUT / "v6_4_claim_run_ledger.csv", index=False)

    m = cc["table"]
    sil = silhouette_table(op["have"])
    sil.to_csv(OUT / "v6_4_silhouette_by_cell.csv", index=False)

    # per-cell corpus census, for the appendix corpus table
    cen = (m.assign(grokked=m.grok_mean5)
             .groupby(["task", "p"])
             .agg(runs=("dir", "size"), grokked=("grokked", "sum"))
             .reset_index())
    cen["memoriser"] = cen.runs - cen.grokked
    cen.to_csv(OUT / "v6_4_corpus_census.csv", index=False)

    res = {
        "n_mlp_corpus": cc["n_mlp"],
        "n_grokked_last_ckpt_rule": cc["n_grok_last"],
        "n_grokked_mean5_rule": cc["n_grok_mean5"],
        "n_memorisers": cc["n_memoriser"],
        "n_randlabel_controls": cc["n_randlabel"],
        "n_operating_point": op["n_op"],
        "n_op_grokked_last_ckpt_rule": op["n_grok_last"],
        "n_op_grokked_mean5_rule": op["n_grok_mean5"],
        "n_op_missing": op["n_missing"],
        "n_silhouette_table_mlp_rows": int(sil.runs.sum()),
    }
    pd.Series(res).to_csv(OUT / "v6_4_count_resolution.csv", header=["value"])
    write_markdown(L, cc, op, sil, REPO / "v6_4_claim_run_ledger.md")

    print("=== v6.4 count resolution ===")
    for k, v in res.items():
        print(f"  {k:<34} {v}")
    print("\n--- runs whose grokked flag depends on the rule ---")
    print(cc["disagree"].to_string(index=False))
    print("\n--- MLP runs absent from the operating-point measurement ---")
    print(op["missing"].to_string(index=False))
    print("\n--- per-cell silhouette of grokked runs (mean-5 rule) ---")
    print(sil.to_string(index=False))
    print(f"\nwrote {OUT/'v6_4_claim_run_ledger.csv'}  ({len(L)} rows)")


if __name__ == "__main__":
    main()
