"""
P0-1 -- two training-time interventions that each move ONE factor, and their validity gate.

The Kazanskii regulariser moves everything at once (E9). To make our own reading causal we need
levers that dissociate:

  L_clust   tightens within-class scatter RELATIVE to between-class separation, leaving the
            within-class covariance SHAPE untouched. Isotropic shrinkage of within-class residuals
            does not change the dimension of the within-class point set, so the target is
            silhouette UP with TwoNN flat.

  L_dim     lowers the within-class intrinsic dimension without moving centroids, via a
            scale-invariant soft rank (nuclear / Frobenius) on the WITHIN-CLASS-CENTRED feature
            matrix. Scale invariance is what keeps it from acting as a clustering lever by the back
            door. Target: TwoNN DOWN with silhouette flat.

GATE (run before any training): optimise a synthetic cloud of KNOWN true dimension directly under
each loss and confirm each moves its target factor and not the other. A loss that fails this is not
used, exactly as with the 17 prior validity gates.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from sklearn.metrics import silhouette_score
from analysis import geom

OUT = REPO / "RESULTS"


# ---------------------------------------------------------------- the two losses
def within_between(H, y, n_cls):
    """Class means and within-class residuals, differentiable."""
    oh = torch.zeros(len(y), n_cls, dtype=H.dtype)
    oh[torch.arange(len(y)), y] = 1.0
    cnt = oh.sum(0).clamp_min(1.0)
    M = (oh.T @ H) / cnt[:, None]                      # (n_cls, d)
    W = H - M[y]                                       # within-class residuals
    B = M - H.mean(0, keepdim=True)                    # between-class offsets
    return W, B, M


def loss_clust(H, y, n_cls, target=6.0, eps=1e-8):
    """
    Increase class separation relative to a DETACHED within-class scale.

    Two earlier versions failed this gate. Minimising within/between without bound collapsed every
    class to a point (silhouette 1.000, residuals numerically zero). A hinge on the same ratio still
    distorted within-class shape, because the per-point gradient is free to move points relative to
    one another (TwoNN 4.86 -> 7.58).

    The fix is to let the loss depend on H *only through the class means*, with the within-class
    scale detached. The gradient on each point is then its class mean's gradient divided by the
    class count -- identical for every point in a class -- so the update is a RIGID TRANSLATION of
    each class. Within-class geometry, and hence its dimension, is untouched by construction. This
    is the training-time analogue of the alpha sweep in E1.
    """
    W, B, _ = within_between(H, y, n_cls)
    sep = B.pow(2).sum() / len(H)
    scale = (W.pow(2).sum() / len(H)).detach()          # detached: no gradient into residuals
    return (sep / (scale + eps) - target).pow(2)


def loss_dim(H, y, n_cls, eps=1e-8):
    """Scale-invariant soft rank of the within-class-centred features: nuclear / Frobenius.
    Bounded by sqrt(rank), so minimising it lowers the within-class dimension without shrinking
    scale and without moving centroids."""
    W, _, _ = within_between(H, y, n_cls)
    s = torch.linalg.svdvals(W)
    return s.sum() / (s.pow(2).sum().sqrt() + eps)


# ---------------------------------------------------------------- gate
def measure(X, y):
    Xn = X if isinstance(X, np.ndarray) else X.detach().numpy()
    W = Xn - np.stack([Xn[y == c].mean(0) for c in y])
    # dimension is scale-free in principle but not numerically: after isotropic shrinkage the
    # residual distances underflow the estimators. Normalise residuals to unit RMS before
    # measuring, so a pure scale change cannot masquerade as a dimension change.
    rms = np.sqrt((W ** 2).sum(1).mean())
    Wn = W / rms if rms > 1e-12 else W
    return {"silhouette": float(silhouette_score(Xn, y)),
            "twonn_within": float(geom.twonn(Wn)),
            "mle_within": float(geom.mle_id(Wn, 10)),
            "twonn_all": float(geom.twonn(Xn)),
            "ph_dim_all": float(geom.ph_dim(Xn)[0])}


def gate(steps=400, lr=0.05, seed=0):
    print("=== P0-1 GATE: do the two losses dissociate on a KNOWN cloud? ===")
    print("    8 classes, within-class manifold of true dimension 5, 400 points\n")
    g = np.random.default_rng(seed)
    d, K, per, amb = 5, 8, 50, 40
    P = g.normal(size=(d, amb))
    cen = g.normal(size=(K, amb)) * 2.0
    y = np.repeat(np.arange(K), per)
    X0 = np.vstack([cen[k] + (g.normal(size=(per, d)) @ P) * 0.5 for k in range(K)])
    yt = torch.as_tensor(y, dtype=torch.long)

    rows = [{"loss": "none", "step": 0, **measure(X0, y)}]
    print(f"  {'loss':10s} {'silhouette':>11s} {'TwoNN within':>13s} {'MLE within':>11s}")
    print(f"  {'start':10s} {rows[0]['silhouette']:>11.3f} {rows[0]['twonn_within']:>13.2f} "
          f"{rows[0]['mle_within']:>11.2f}")
    for name, fn in (("L_clust", loss_clust), ("L_dim", loss_dim)):
        H = torch.tensor(X0, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([H], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            fn(H, yt, K).backward()
            opt.step()
        m = measure(H, y)
        rows.append({"loss": name, "step": steps, **m})
        print(f"  {name:10s} {m['silhouette']:>11.3f} {m['twonn_within']:>13.2f} "
              f"{m['mle_within']:>11.2f}")
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "p0_1_gate.csv", index=False)

    s0, sc, sd = G.silhouette.values
    t0, tc, td = G.twonn_within.values
    clust_ok = (sc - s0) > 0.15 and abs(tc - t0) / max(t0, 1e-9) < 0.25
    dim_ok = (t0 - td) / max(t0, 1e-9) > 0.20 and abs(sd - s0) < 0.15
    print(f"\n  L_clust moves silhouette {s0:+.3f} -> {sc:+.3f} "
          f"({'PASS' if clust_ok else 'FAIL'}: TwoNN {t0:.2f} -> {tc:.2f})")
    print(f"  L_dim   moves TwoNN     {t0:.2f} -> {td:.2f} "
          f"({'PASS' if dim_ok else 'FAIL'}: silhouette {s0:+.3f} -> {sd:+.3f})")
    ok = clust_ok and dim_ok
    print(f"\n  GATE {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    gate()
