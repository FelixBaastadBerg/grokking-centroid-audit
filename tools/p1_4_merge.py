"""
Fold the P1-4 external-validity cells into the two corpus tables the paper reads from, tagged so
they can always be separated back out, and print what they say about the universality claim.

Extends RESULTS/operating_point.csv (label-silhouette init -> convergence) and
RESULTS/e1_centroid.csv (the Table 1 alpha sweep). Idempotent: re-running replaces the tagged rows
rather than appending duplicates.

The question this has to answer honestly is whether clustering weakens at larger p, where there are
more output classes and fewer points per class. If it does, that is a scope refinement on our own
claim and it gets reported as one.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
R = REPO / "RESULTS"
TAG = "P1-4"


def merge(base_name: str, new_frames: list[pd.DataFrame]) -> pd.DataFrame | None:
    new = [d for d in new_frames if d is not None and len(d)]
    if not new:
        return None
    add = pd.concat(new, ignore_index=True)
    add["source"] = TAG
    base_path = R / base_name
    if base_path.exists():
        base = pd.read_csv(base_path)
        if "source" not in base.columns:
            base["source"] = "corpus"
        base = base[base.source != TAG]                     # idempotent
        out = pd.concat([base, add], ignore_index=True)
    else:
        out = add
    bak = R / (base_name.replace(".csv", "_pre_p1_4.csv"))
    if base_path.exists() and not bak.exists():
        pd.read_csv(base_path).to_csv(bak, index=False)     # keep the pre-merge table once
    out.to_csv(base_path, index=False)
    return out


def load(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


if __name__ == "__main__":
    op_mod, e1_mod = load("p1_4_operating_modular.csv"), load("p1_4_centroid_modular.csv")
    op_par, e1_par = load("p1_4_operating_parity.csv"), load("p1_4_centroid_parity.csv")
    op_mat, e1_mat = load("p1_4_operating_matched.csv"), load("p1_4_centroid_matched.csv")
    if op_mod is None and op_par is None and op_mat is None:
        sys.exit("no P1-4 results yet")

    # tag which subsample each row came from: "capped" is the fixed 400-point budget every corpus
    # cell uses, "matched" holds points-per-class at the real operating point (m = 20)
    for d, tag in ((op_mod, "capped"), (e1_mod, "capped"),
                   (op_par, "capped"), (e1_par, "capped")):
        if d is not None and "subsample" not in d.columns:
            d["subsample"] = tag

    op = merge("operating_point.csv", [op_mod, op_par, op_mat])
    e1 = merge("e1_centroid.csv", [e1_mod, e1_par, e1_mat])
    print(f"operating_point.csv -> {len(op)} rows ({int((op.source == TAG).sum())} tagged {TAG})")
    if e1 is not None:
        print(f"e1_centroid.csv     -> {len(e1)} rows "
              f"({int((e1.source == TAG).sum())} tagged {TAG})")

    print("\n--- silhouette init -> convergence, and grokked/total, by new cell ---")
    new = op[op.source == TAG]
    # capped and matched are different clouds from the same networks; never pool their silhouettes
    okeys = ["task", "p", "subsample"] if "subsample" in new.columns else ["task", "p"]
    for key, g in new.groupby(okeys):
        ng, nt = int(g.grokked.sum()), len(g)
        gg = g[g.grokked] if ng else g
        lab = "/".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
        print(f"  {lab:<22} grokked {ng}/{nt}   "
              f"sil {gg.sil_init.median():+.3f} -> {gg.sil_final.median():+.3f}   "
              f"final test {g.final_test.median():.3f}")

    print("\n--- for comparison, the existing corpus cells ---")
    old = op[op.source != TAG]
    if "grokked" in old.columns and len(old):
        og = old[old.grokked.astype(str).str.lower().isin(["true", "1"])] if \
            old.grokked.dtype == object else old[old.grokked.astype(bool)]
        print(f"  corpus grokked median sil_final = {og.sil_final.median():+.3f} (n={len(og)})")

    # the alpha-sweep folds: does the estimator separation survive at larger p?
    if e1 is not None and len(e1[e1.source == TAG]):
        n = e1[e1.source == TAG]
        # same convention as disc_conf_table / Table 1: per run, max over the alpha sweep
        # divided by min. Reported per cell as the median over seeds.
        print("\n--- Table 1 alpha-sweep folds on the new cells (per-run max/min over alpha) ---")
        keys = ["task", "p", "subsample"] if "subsample" in n.columns else ["task", "p"]
        for key, g in n.groupby(keys):
            row = []
            for est in ["ph_dim", "corr_dim", "H1_total", "twonn", "mle_k10"]:
                if est not in g.columns:
                    continue
                folds = []
                for _, x in g.groupby("seed"):
                    m0 = x.groupby("alpha")[est].mean()
                    if m0.notna().sum() > 3 and abs(m0.min()) > 1e-9:
                        folds.append(m0.max() / abs(m0.min()))
                if folds:
                    row.append(f"{est} {np.median(folds):5.2f}x")
            lab = "/".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
            print(f"  {lab:<22} " + "  ".join(row))

        # the same folds on the original corpus, so the comparison is like for like
        print("\n--- the corpus cells, same statistic ---")
        c = e1[(e1.source != TAG) & (e1.sweep == "alpha") & (e1.norm == "trace")]
        row = []
        for est in ["ph_dim", "corr_dim", "H1_total", "twonn", "mle_k10"]:
            folds = []
            for _, x in c.groupby("dir"):
                m0 = x.groupby("alpha")[est].mean()
                if m0.notna().sum() > 3 and abs(m0.min()) > 1e-9:
                    folds.append(m0.max() / abs(m0.min()))
            if folds:
                row.append(f"{est} {np.median(folds):5.2f}x")
        print(f"  {'corpus (p=23, m~17)':<22} " + "  ".join(row))
