"""
Item 1, gate 2 -- what is the largest class-conditional dimension m points per class can support?

The trajectory run reads conditional TwoNN at ~27 early in training and ~15 at convergence. The
first validation only covered true d in {2, 5, 8}, so both numbers are outside the range where the
estimator was shown calibrated, and one hard fact makes that urgent: a class holding m points spans
an affine subspace of dimension at most m - 1, so at the corpus operating point of m = 17 the true
class-conditional dimension CANNOT exceed 16. Any reading above that is off-scale by construction.

This sweeps true d up to and past that bound at several m, and reports where the estimator
saturates. If it saturates below the observed readings, the observed decline cannot be reported as a
dimensional compression, and the honest conclusion is that the direct measurement is out of reach at
this m -- which is itself a result, and the one the paper's own discipline demands.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from tools.v2_conditional import cond_twonn, cond_mle, residuals, spectral, cloud

OUT = REPO / "RESULTS"
DS = [2, 4, 6, 8, 12, 16, 24, 32, 48]
MS = [17, 32, 64, 128]
AMB = 200          # the real hidden width, so the ambient ceiling matches the networks


def main():
    print("=== CEILING: what can m points per class actually resolve? ===")
    print(f"    ambient {AMB} (the real hidden width); a class of m points spans at most m-1 "
          f"dimensions\n")
    rows = []
    for m in MS:
        for d in DS:
            if d >= AMB:
                continue
            for s in (0, 1, 2):
                X, y = cloud(d, 23, m, 0.05, seed=s, amb=AMB)
                R = residuals(X, y)
                rows.append({"m": m, "true_d": d, "seed": s,
                             "cond_twonn": cond_twonn(X, y),
                             "cond_mle_k10": cond_mle(X, y, 10),
                             "within_eff_rank": spectral(R)["eff_rank"],
                             "ceiling": m - 1})
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "v2_conditional_ceiling.csv", index=False)

    for est in ("cond_twonn", "cond_mle_k10", "within_eff_rank"):
        P = C.pivot_table(index="m", columns="true_d", values=est)
        print(f"  {est} (rows m, cols true d):")
        print("   m \\ d " + "".join(f"{d:>7}" for d in P.columns))
        for m_, r in P.iterrows():
            print(f"   {m_:>5} " + "".join(f"{v:>7.1f}" for v in r.values)
                  + f"    | ceiling m-1 = {m_-1}")
        print()

    print("  saturation: the largest value the estimator returns at each m, over all true d")
    for est in ("cond_twonn", "cond_mle_k10", "within_eff_rank"):
        s = C.groupby("m")[est].max()
        print(f"    {est:18s} " + "  ".join(f"m={m}: {v:.1f}" for m, v in s.items()))

    print("\n  usable range: largest true d whose estimate is within 25% of the truth")
    for est in ("cond_twonn", "cond_mle_k10", "within_eff_rank"):
        out = []
        for m_, g in C.groupby("m"):
            ok = [d for d, gg in g.groupby("true_d")
                  if abs(gg[est].median() - d) / d <= 0.25]
            out.append(f"m={m_}: {max(ok) if ok else '-'}")
        print(f"    {est:18s} " + "  ".join(out))


if __name__ == "__main__":
    main()
