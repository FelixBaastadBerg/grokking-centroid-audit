"""
v6.8 -- resolve the Z_59 population contradiction carried by v6.7.

v6.7 states two things that cannot both describe one corpus:

  * Appendix Table 1 (the 116-network census): Z_59 MLP = 12 runs, 2 distinct seeds, 7 grokked.
  * Main text / B.1 / claim table / D.3: the Z_59 replication = 9 grokked runs across train
    fractions 0.4-0.8, apparently seeds 0-4.

This script enumerates every Z_59 addition MLP run directory on disk, reads each run's own
metadata and curves rather than any summary file, and classifies it. Nothing is inferred from a
directory name that can be read from the run itself; the name is recorded so a reader can match
the two.

For each run:
  dir, stable id, seed, train fraction, width, lr, weight decay, epochs, batch,
  final train/test accuracy under the paper's single rule (mean of the last five checkpoints),
  grokked flag, whether endpoint checkpoints exist, whether it appears in the corpus inventory
  (RESULTS/v3_t11_run_table.csv), and whether it entered the reported recombination analysis
  (RESULTS/v3_crosscell.csv).

Output: RESULTS_v6_8/z59_run_audit.csv and z59_population_summary.csv, plus a printed
reconciliation of the two disputed counts.

Analysis-only: retained checkpoints, no training.
"""
from __future__ import annotations
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from analysis.lsrun import LandscapeRun

OUT = REPO / "RESULTS_v6_8"
OUT.mkdir(exist_ok=True)
RUNS = REPO / "runs_topo"

GROK_ACC = 0.9          # the paper's single rule
FINAL_K = 5             # final value = mean over the last five checkpoints


def candidates():
    """Every run directory whose own metadata says Z_59 addition on an MLP."""
    out = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        meta_f = d / "run_meta.json"
        if not meta_f.exists():
            continue
        try:
            meta = json.loads(meta_f.read_text())
        except Exception:
            continue
        if meta.get("task") == "add" and int(meta.get("p", -1)) == 59 \
                and meta.get("arch") == "mlp":
            out.append((d, meta))
    return out


def main():
    corpus = pd.read_csv(REPO / "RESULTS" / "v3_t11_run_table.csv")
    corpus_dirs = set(corpus.dir)
    cross = pd.read_csv(REPO / "RESULTS" / "v3_crosscell.csv")
    used_dirs = set(cross[cross.cell == "Z59_add_mlp"].dir)

    rows = []
    for d, meta in candidates():
        rec = {
            "dir": d.name,
            "seed": meta.get("seed"),
            "train_fraction": meta.get("train_fraction"),
            "width": meta.get("hidden"),
            "lr": meta.get("lr"),
            "weight_decay": meta.get("weight_decay"),
            "epochs": meta.get("num_epochs"),
            "batch": meta.get("batch_size", 0),
            "suffix": d.name.split("__")[-1] if "__" in d.name else "",
            "has_trajectory": (d / "trajectory.npz").exists(),
            "in_corpus_inventory": d.name in corpus_dirs,
            "in_recombination": d.name in used_dirs,
        }
        # the run's own curves, under the paper's rule
        try:
            r = LandscapeRun(d)
            te = r.curves.test_acc.values
            tr = r.curves.train_acc.values
            rec["n_checkpoints"] = int(len(r.epochs))
            rec["final_test"] = float(te[-FINAL_K:].mean())
            rec["final_train"] = float(tr[-FINAL_K:].mean())
            rec["grokked"] = bool(rec["final_test"] >= GROK_ACC)
        except Exception as e:
            rec["n_checkpoints"] = 0
            rec["final_test"] = np.nan
            rec["final_train"] = np.nan
            rec["grokked"] = False
            rec["load_error"] = f"{type(e).__name__}: {e}"
        rows.append(rec)

    D = pd.DataFrame(rows).sort_values(["train_fraction", "seed", "suffix"])
    D.to_csv(OUT / "z59_run_audit.csv", index=False)

    print(f"=== every Z_59 addition MLP run directory on disk: {len(D)} ===")
    cols = ["dir", "seed", "train_fraction", "width", "weight_decay", "n_checkpoints",
            "final_test", "grokked", "has_trajectory", "in_corpus_inventory",
            "in_recombination"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 62):
        print(D[cols].to_string(index=False))

    print("\n=== reconciliation of the two disputed counts ===")
    inv = D[D.in_corpus_inventory]
    rec_ = D[D.in_recombination]
    print(f"  in the 116-network corpus inventory : {len(inv):>2} runs, "
          f"{inv.seed.nunique()} distinct seeds, {int(inv.grokked.sum())} grokked, "
          f"train fractions {sorted(set(inv.train_fraction))}")
    print(f"  in the reported recombination      : {len(rec_):>2} runs, "
          f"{rec_.seed.nunique()} distinct seeds, {int(rec_.grokked.sum())} grokked, "
          f"train fractions {sorted(set(rec_.train_fraction))}")
    both = D[D.in_corpus_inventory & D.in_recombination]
    only_rec = D[D.in_recombination & ~D.in_corpus_inventory]
    only_inv = D[D.in_corpus_inventory & ~D.in_recombination]
    print(f"  in both                            : {len(both):>2}")
    print(f"  recombination only (outside census): {len(only_rec):>2}  "
          + ";".join(sorted(only_rec.dir)))
    print(f"  census only (not decomposed)       : {len(only_inv):>2}  "
          + ";".join(sorted(only_inv.dir)))

    print("\n=== the recombination population, by train fraction ===")
    if len(rec_):
        g = (rec_.groupby("train_fraction")
                 .agg(runs=("dir", "size"), seeds=("seed", "nunique"),
                      grokked=("grokked", "sum"),
                      seed_ids=("seed", lambda s: ";".join(str(x) for x in sorted(s)))))
        print(g.to_string())

    summ = []
    for name, sub in (("corpus_inventory", inv), ("recombination", rec_),
                      ("all_on_disk", D)):
        summ.append({
            "population": name, "runs": len(sub), "distinct_seeds": sub.seed.nunique(),
            "grokked": int(sub.grokked.sum()),
            "train_fractions": ";".join(f"{v:g}" for v in sorted(set(sub.train_fraction))),
            "seed_ids": ";".join(str(v) for v in sorted(set(sub.seed))),
        })
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "z59_population_summary.csv", index=False)
    print("\nwrote RESULTS_v6_8/z59_run_audit.csv, z59_population_summary.csv")


if __name__ == "__main__":
    main()
