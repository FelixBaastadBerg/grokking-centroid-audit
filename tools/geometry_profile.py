"""
T3.5 -- the geometry-profile diagnostic: turns the paper's warning into a tool.

Given a point cloud X (n x D) and class labels y, prints a GEOMETRY PROFILE and, for each
common geometric statistic, an automatic reading of whether it may be interpreted as
class-conditional (within-class) geometry or must be read as between-class geometry, applying
the decision rules established in the paper:

  m            points per class (median). m <~ 20 puts local estimators at risk and caps any
               conditional estimate at m - 1.
  L_q          support leakage: fraction of points whose first q nearest neighbours leave the
               class. L_q = 0 (both endpoints, if comparing two clouds) is the exact-invariance
               condition of the fixed-order proposition.
  silhouette   true-label silhouette; >~ 0.5 is the clustered regime.
  gamma        separation-to-scale: RMS centroid distance from the grand mean over the
               within-class RMS extent (per the sqrt(d)-normalised convention).
  u            between/within MST-energy ratio at alpha = 1 -- the mixture variable governing
               PH-dimension; the fitted value collapses toward alpha as u grows,
               d_hat = alpha (1+u) / (u + alpha/d).

Usage:
    python tools/geometry_profile.py                          # demo on synthetic clouds
    from tools.geometry_profile import geometry_profile
    report = geometry_profile(X, y)     # dict; report["text"] is the printable reading

Anonymised, dependency-light (numpy, scipy); ships with the paper.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

ALPHA = 1.0


def _leakage(D, y, q):
    n = len(y)
    L = 0
    order = np.argsort(D, axis=1)
    for i in range(n):
        nb = [j for j in order[i] if j != i][:q]
        L += any(y[j] != y[i] for j in nb)
    return L / n


def geometry_profile(X, y, q=2):
    X = np.asarray(X, float); y = np.asarray(y)
    n, _ = X.shape
    cls, cnt = np.unique(y, return_counts=True)
    K, m = len(cls), float(np.median(cnt))
    D = squareform(pdist(X))

    # silhouette (simplified, exact)
    sil = np.nan
    if K > 1 and n <= 4000:
        s_vals = []
        for i in range(n):
            same = y == y[i]; same[i] = False
            if not same.any():
                continue
            a = D[i][same].mean()
            b = min(D[i][y == c].mean() for c in cls if c != y[i])
            s_vals.append((b - a) / max(a, b))
        sil = float(np.mean(s_vals))

    mus = {c: X[y == c].mean(0) for c in cls}
    mu = X.mean(0)
    R_rms = float(np.sqrt(np.mean([np.sum((mus[c] - mu) ** 2) for c in cls])))
    res = X - np.stack([mus[c] for c in y])
    within_rms = float(np.sqrt(np.mean(np.sum(res ** 2, 1))))
    gamma = R_rms / max(within_rms, 1e-12)

    # within-class spectral rank (calibrated to K(m-1)) and its ceiling
    s = np.linalg.svd(res - res.mean(0), compute_uv=False) ** 2
    s = s[s > 1e-30]; p_ = s / s.sum()
    w_er = float(np.exp(-(p_ * np.log(p_)).sum()))

    # MST energy split
    T = minimum_spanning_tree(D)
    Tc = T.tocoo()
    w = Tc.data ** ALPHA
    cross = y[Tc.row] != y[Tc.col]
    E_b, E_w = float(w[cross].sum()), float(w[~cross].sum())
    u = E_b / max(E_w, 1e-12)

    Lq = _leakage(D, y, q)

    clustered = (sil >= 0.5) if np.isfinite(sil) else (gamma >= 2.0)
    weakly = (0.2 <= sil < 0.5) if np.isfinite(sil) else (1.0 <= gamma < 2.0)
    flags = {
        "small_m": m < 20,
        "clustered": clustered,
        "leaky": Lq > 0,
        # u is a mixture ratio only when the labels actually partition the cloud; on an
        # unclustered cloud most MST edges cross labels by chance and u means nothing
        "mixture_dominated": clustered and u > 0.2,
    }
    readings = {}
    readings["PH-dimension / correlation dimension / total persistence"] = (
        "read as BETWEEN-class geometry (report K alongside)" if flags["mixture_dominated"]
        else ("clustered but within-term dominates; check u across subsample sizes" if clustered
              else "cloud is not clustered by these labels: readable as cloud geometry"
                   + (" (weak clustering: interpret with care)" if weakly else "")))
    readings["TwoNN / fixed-order local estimators"] = (
        "invariance holds exactly (L_%d = 0)" % q if Lq == 0 else
        "L_%d = %.2f > 0: fixed-order invariance NOT guaranteed" % (q, Lq))
    if m < 20:
        readings["TwoNN / fixed-order local estimators"] += \
            "; m < 20: small-m regime, local estimators unreliable"
    readings["conditional (within-class) estimators"] = (
        "ceiling: a class of m points resolves at most d ~ m/2 (hard cap m-1 = %d); "
        "%s" % (int(m) - 1,
                "usable" if m >= 40 else "at m = %.0f treat nonlinear conditional estimates "
                "as unreliable; use within-class spectra (budget K(m-1) = %d)"
                % (m, int(K * (m - 1)))))
    readings["global spectral statistics (eff. rank, PR, entropy)"] = (
        "mixture-dominated (u = %.2f): global spectrum reflects centroid arrangement plus "
        "within-class spectrum; decompose before interpreting" % u if flags["mixture_dominated"]
        else ("within-term dominates the spectrum (u = %.2f)" % u if clustered else
              "labels do not partition the cloud; global spectrum is cloud geometry"))

    d_apparent = ALPHA * (1 + u) / (u + ALPHA / max(w_er, 1e-9))
    text = [
        "GEOMETRY PROFILE",
        f"  n = {n}, K = {K}, m (median points/class) = {m:.0f}",
        f"  true-label silhouette = {sil:+.3f}" if np.isfinite(sil) else
        "  silhouette: skipped (n too large)",
        f"  gamma (separation-to-scale) = {gamma:.2f}",
        f"  L_{q} (support leakage) = {Lq:.3f}",
        (f"  MST energy ratio u (between/within) = {u:.3f}"
         + (f"  -> mixture-model PH-dim would read ~{d_apparent:.1f}" if clustered
            else "  (not a mixture ratio: labels do not partition the cloud)")),
        f"  within-class effective rank = {w_er:.1f}",
        "READINGS",
    ] + [f"  {k}: {v}" for k, v in readings.items()]
    return {"n": n, "K": K, "m": m, "sil": sil, "gamma": gamma, "Lq": Lq, "u": u,
            "within_eff_rank": w_er, "flags": flags, "readings": readings,
            "text": "\n".join(text)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("--- demo 1: clustered mixture (the grokked regime) ---")
    K, m, d = 23, 17, 5
    cen = rng.normal(size=(K, 60)) * 3
    P = rng.normal(size=(d, 60))
    X = np.vstack([cen[k] + rng.normal(size=(m, d)) @ P * 0.1 for k in range(K)])
    y = np.repeat(np.arange(K), m)
    print(geometry_profile(X, y)["text"])
    print("\n--- demo 2: unclustered cloud, same points ---")
    X2 = np.vstack([rng.normal(size=(m, d)) @ P for _ in range(K)])
    print(geometry_profile(X2, y)["text"])
