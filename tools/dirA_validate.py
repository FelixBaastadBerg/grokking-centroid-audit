"""
DIR A instrument gate. Every landscape here has an ANALYTIC answer.

  convex bowl        sublevel sets convex  -> b0=1, b1=0; barrier(a,b) = max(F(a),F(b))
  double well        F=(x^2-1)^2+y^2       -> minima 0 at (+-1,0), saddle EXACTLY 1 at origin
  tunable barrier    F=h(x^2-1)^2+y^2      -> barrier must come back as h, for every h
  ring valley        F=(r-1)^2             -> annulus: b1=1 for t<1, b1=0 for t>=1
  three wells                              -> b0=3 below the lowest saddle
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from analysis.landscape import minimax_barrier, betti_sublevel

N = 401
ax = np.linspace(-2, 2, N)
X, Y = np.meshgrid(ax, ax, indexing="xy")


def idx(x, y):
    return int(np.argmin(np.abs(ax - y))), int(np.argmin(np.abs(ax - x)))


if __name__ == "__main__":
    print("=== DIR A GATE 1: minimax barrier must equal the analytic saddle height ===\n")
    rows = []
    for h in (0.25, 0.5, 1.0, 2.0, 4.0):
        F = h * (X ** 2 - 1) ** 2 + Y ** 2
        got = minimax_barrier(F, idx(-1, 0), idx(1, 0))
        rows.append({"landscape": f"double well h={h}", "true": h, "measured": round(got, 4),
                     "err": round(abs(got - h), 4)})
    Fc = X ** 2 + Y ** 2
    a, b = (-1.0, 0.5), (1.0, -0.5)
    true_c = max(a[0] ** 2 + a[1] ** 2, b[0] ** 2 + b[1] ** 2)
    got = minimax_barrier(Fc, idx(*a), idx(*b))
    rows.append({"landscape": "convex bowl", "true": true_c, "measured": round(got, 4),
                 "err": round(abs(got - true_c), 4)})
    V = pd.DataFrame(rows)
    print(V.to_string(index=False))
    print(f"\n  max error over all barrier tests: {V.err.max():.4f}  "
          f"(grid resolution is {ax[1]-ax[0]:.4f})")

    print("\n=== DIR A GATE 2: Betti curves of sublevel sets ===\n")
    lv = np.linspace(0.02, 2.0, 25)

    Fr = (np.sqrt(X ** 2 + Y ** 2) - 1) ** 2
    b0, b1 = betti_sublevel(Fr, lv)
    ok_lo = all(b1[lv < 0.95] == 1)
    ok_hi = all(b1[lv > 1.05] == 0)
    print(f"  ring valley   b1=1 while t<1 : {ok_lo}   b1=0 once t>1 : {ok_hi}")
    print(f"                b1 over t: {list(b1)}")

    b0c, b1c = betti_sublevel(Fc, lv)
    print(f"  convex bowl   b0 always 1: {all(b0c == 1)}   b1 always 0: {all(b1c == 0)}")

    F3 = np.ones_like(X) * 3.0
    for cx, cy in ((-1.2, 0.0), (1.2, 0.0), (0.0, 1.2)):
        F3 = np.minimum(F3, 3.0 * ((X - cx) ** 2 + (Y - cy) ** 2))
    b03, _ = betti_sublevel(F3, lv)
    print(f"  three wells   b0=3 at the lowest level: {b03[0] == 3}  (b0 curve {list(b03[:6])} ...)")

    passed = V.err.max() < 0.02 and ok_lo and ok_hi and all(b0c == 1) and b03[0] == 3
    print(f"\n  GATE {'PASSED' if passed else 'FAILED'}")
