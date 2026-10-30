"""
DIR C gate. RTD must satisfy its own definition before it is allowed near real data.

Hard requirements (any failure kills the direction):
  RTD(A, A) == 0 exactly            -- the pair is trivial when the clouds agree
  isometry / rescale -> 0           -- it compares intrinsic geometry, not coordinates
  circle vs disk -> large           -- it must actually see a topological difference
  monotone in the size of the change
  survives at the point counts we would use
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np
from analysis.rtd import rtd, dmat

if __name__ == "__main__":
    r = np.random.default_rng(0)
    n = 120
    t = np.sort(r.uniform(0, 2 * np.pi, n))
    circ = np.c_[np.cos(t), np.sin(t)]
    Q, _ = np.linalg.qr(r.normal(size=(2, 2)))
    disk = r.normal(size=(n, 2)) * 0.35
    line = np.c_[t / (2 * np.pi), np.zeros(n)]

    print(f"=== DIR C GATE: RTD_1 on known answers (n={n}) ===\n")
    cases = [("IDENTICAL              -> must be 0", circ, circ, "zero"),
             ("rotated copy           -> must be 0", circ, circ @ Q, "zero"),
             ("rescaled x3            -> must be 0", circ, circ * 3, "zero"),
             ("circle vs +2% noise    -> small", circ, circ + r.normal(size=circ.shape) * 0.02, None),
             ("circle vs +10% noise   -> larger", circ, circ + r.normal(size=circ.shape) * 0.10, None),
             ("circle vs CUT circle   -> large", circ, line, "big"),
             ("circle vs filled disk  -> large", circ, disk, "big")]
    vals = {}
    ok = True
    for nm, A, B, req in cases:
        v = rtd(dmat(A), dmat(B))
        vals[nm] = v
        flag = ""
        if req == "zero" and abs(v) > 1e-9:
            flag = "  <-- FAIL"; ok = False
        if req == "big" and v < 1.0:
            flag = "  <-- FAIL"; ok = False
        print(f"  {nm:38s} RTD_1 = {v:9.5f}{flag}")

    print("\n=== monotonicity in the magnitude of the perturbation ===")
    prev, mono = -1.0, True
    for eps in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40):
        v = rtd(dmat(circ), dmat(circ + np.random.default_rng(3).normal(size=circ.shape) * eps))
        print(f"  noise={eps:4.2f}  RTD_1 = {v:9.5f}")
        if v + 1e-9 < prev:
            mono = False
        prev = v
    print(f"  monotone: {mono}")

    print("\n=== point-count behaviour (circle vs disk, matched index) ===")
    for n2 in (40, 80, 120, 160):
        tt = np.linspace(0, 2 * np.pi, n2, endpoint=False)
        c = np.c_[np.cos(tt), np.sin(tt)]
        d = np.random.default_rng(1).normal(size=(n2, 2)) * 0.35
        s = rtd(dmat(c), dmat(d))
        nl = rtd(dmat(c), dmat(c + np.random.default_rng(2).normal(size=c.shape) * 0.05))
        print(f"  n={n2:4d}  signal={s:8.4f}  noise-null={nl:8.4f}  ratio={s/max(nl,1e-9):7.1f}x")

    print(f"\n  GATE {'PASSED' if ok and mono else 'FAILED'}")
