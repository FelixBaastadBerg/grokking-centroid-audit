"""
v6.4 Phase 2.5 -- remove configuration-selection ambiguity as far as the stored runs allow.

v6.3's Transformer evidence is a ten-seed confirmation at one weight decay (0.001), chosen from
pilots at two settings. The corrected implementation's SECOND setting (weight decay 0.01) has
stored checkpoints for two seeds, and both grokked. Those runs exist, so they are analysed here
with no further selection and every seed is reported, rather than left as a gap.

This is two seeds, not ten. It is reported as such and supports no confirmatory claim. Running
ten predeclared seeds at the second setting would need ten fresh 40k-step trainings, which was
outside this revision's compute budget; the worklog records that.

Exactly the rules already fixed elsewhere are applied, with nothing re-tuned:
  * grokked, t_grok      from RESULTS/brown_transformer_run_manifest.csv
  * plateau checkpoint   last SAVED checkpoint before t_grok with val <= 0.10 and train >= 0.99
  * attribution          the 2x2 recombination, both factor orders and the symmetric value
  * estimators           this paper's PH-dimension and TwoNN, unchanged

Analysis-only. Writes RESULTS_v6_4/brown_second_config.csv.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "brown_transformer"))

import numpy as np
import pandas as pd

from analysis import geom
from tools.v6_4.brown_exact import (load_run, parts, trace_norm, steps_of, plateau_step,
                                    leakage, RUNS)

OUT = REPO / "RESULTS_v6_4"
OUT.mkdir(exist_ok=True)


def estimators(Z):
    return {"ph_dim": geom.ph_dim(Z)[0], "twonn": geom.twonn(Z)}


def decompose(clouds, est):
    v = {k: estimators(trace_norm(Z))[est] for k, Z in clouds.items()}
    diag = v["00"] - v["11"]
    first = v["00"] - v["10"]
    second = v["01"] - v["11"]
    phi_b = 0.5 * (first + second)
    return {"f00": v["00"], "f10": v["10"], "f01": v["01"], "f11": v["11"],
            "raw_change": diag,
            "centroid_first_raw": first, "centroid_second_raw": second,
            "phi_between_raw": phi_b, "phi_within_raw": diag - phi_b,
            "between_share": phi_b / diag if abs(diag) > 1e-9 else np.nan,
            "within_share": 1 - phi_b / diag if abs(diag) > 1e-9 else np.nan,
            "centroid_first_share": first / diag if abs(diag) > 1e-9 else np.nan,
            "centroid_second_share": second / diag if abs(diag) > 1e-9 else np.nan}


def one_run(d, rec):
    from brown_analyze import activations
    meta, ds, Xva, y = load_run(d)
    curves = pd.read_csv(d / "curves.csv")
    steps = steps_of(d)
    tg = int(rec.t_grok_step)
    sf = steps[-1]
    s0 = steps[0]
    sp = plateau_step(curves, tg, set(steps)) if tg > 0 else -1

    H = {s: activations(d, s, ds, Xva) for s in {s0, sf} | ({sp} if sp > 0 else set())}
    C, Rr = {}, {}
    for s, h in H.items():
        C[s], Rr[s] = parts(h, y)

    rows = []
    intervals = [("init->conv", s0, sf)]
    if sp > 0:
        intervals.append(("plateau->conv", sp, sf))
    for label, a, b in intervals:
        clouds = {"00": C[a] + Rr[a], "10": C[b] + Rr[a],
                  "01": C[a] + Rr[b], "11": C[b] + Rr[b]}
        for est in ("ph_dim", "twonn"):
            rows.append({"dir": d.name, "seed": meta["seed"],
                         "weight_decay": float(rec.weight_decay), "tag": rec.tag,
                         "grokked": bool(rec.grokked), "t_grok_step": tg,
                         "interval": label, "step_a": a, "step_b": b,
                         "L2_a": leakage(trace_norm(H[a]), y, 2),
                         "L2_b": leakage(trace_norm(H[b]), y, 2),
                         "estimator": est, **decompose(clouds, est)})
    return rows


def main():
    t0 = time.time()
    man = pd.read_csv(REPO / "RESULTS" / "brown_transformer_run_manifest.csv")
    # the corrected implementation is the pilot2 round; both weight-decay settings, all seeds
    sel = man[man.tag == "pilot2"].sort_values(["weight_decay", "seed"])
    print(f"=== v6.4 second-configuration analysis: {len(sel)} corrected pilot runs ===")
    print(f"    weight decays {sorted(sel.weight_decay.unique())}, "
          f"seeds {sorted(sel.seed.unique())}, all reported", flush=True)

    rows = []
    for _, rec in sel.iterrows():
        rows += one_run(RUNS / rec.dir, rec)
        print(f"  {rec.dir.split('__')[-2]} wd={rec.weight_decay} done "
              f"({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "brown_second_config.csv", index=False)

    print("\n--- per run, both weight decays, every seed reported ---")
    for est in ("ph_dim", "twonn"):
        print(f"\n  {est}")
        print(f"    {'wd':>6}{'seed':>6}{'interval':>16}{'change':>9}"
              f"{'between':>9}{'1st':>8}{'2nd':>8}")
        for _, r in D[D.estimator == est].iterrows():
            print(f"    {r.weight_decay:>6}{r.seed:>6}{r.interval:>16}"
                  f"{r.raw_change:>9.2f}{r.between_share:>9.0%}"
                  f"{r.centroid_first_share:>8.0%}{r.centroid_second_share:>8.0%}")
    print(f"\nwrote brown_second_config.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
