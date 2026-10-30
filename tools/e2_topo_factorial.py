"""
E2 -- topology x clustering factorial. The control that decides whether an H1 claim is defensible.

2 x 2 design with ambient dimension, scale, noise and n held fixed:

                    LOW clustering                    HIGH clustering
    NO LOOP    uniform disk                      K tight clumps in a blob
    ONE LOOP   uniform annulus (a real H1 class) K tight clumps arranged ON a circle

Three questions, each of which the headline needs answered:
  Q1  At matched clustering, does H1 detect the real loop at all?
  Q2  Is the clustering effect LARGER than the true topology effect?
  Q3  When H1 responds to clustering, is it one dominant bar (a genuine global loop) or many
      short bars (the signature of gaps between clumps)?

Significance rule, stated once and used everywhere: a bar counts as significant if its lifetime
exceeds the 95th percentile of the longest-bar lifetime over `n_null` label-free nulls formed by
resampling points from a Gaussian matched to the cloud's covariance. "No persistent H1 class
above the calibrated threshold" means: no bar passes this rule. This replaces the earlier
shorthand "b1 = 0".
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from ripser import ripser

OUT = REPO / "RESULTS"
N, AMB = 300, 40


def _prep(X):
    X = np.asarray(X, float) - np.mean(X, 0)
    D = np.sqrt(np.maximum(((X[:, None] - X[None]) ** 2).sum(-1), 0))
    d = D.max()
    return D / d if d > 0 else D


def h1_full(X, maxdim=1):
    D = _prep(X)
    dg = ripser(D, distance_matrix=True, maxdim=maxdim)["dgms"][1]
    if not len(dg):
        return {"H1_max": 0.0, "H1_total": 0.0, "n_bars": 0, "dom_ratio": 0.0, "life": np.array([])}
    dg = dg[np.isfinite(dg[:, 1])]
    life = np.sort(dg[:, 1] - dg[:, 0])[::-1] if len(dg) else np.array([])
    if not len(life):
        return {"H1_max": 0.0, "H1_total": 0.0, "n_bars": 0, "dom_ratio": 0.0, "life": life}
    return {"H1_max": float(life[0]), "H1_total": float(life.sum()), "n_bars": int(len(life)),
            "dom_ratio": float(life[0] / life.sum()), "life": life}


def null_threshold(X, n_null=12, seed=0, q=95):
    """95th percentile of the longest H1 bar over covariance-matched Gaussian nulls."""
    g = np.random.default_rng(seed)
    Xc = np.asarray(X, float) - np.mean(X, 0)
    C = np.cov(Xc, rowvar=False)
    C = C + 1e-9 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    L = np.linalg.cholesky(C)
    mx = []
    for _ in range(n_null):
        Z = g.normal(size=(len(X), C.shape[0])) @ L.T
        mx.append(h1_full(Z)["H1_max"])
    return float(np.percentile(mx, q)), float(np.median(mx))


# ---------------------------------------------------------------- constructions
def make(topology, clustering, n=N, amb=AMB, K=10, seed=0, noise=0.03, thick=0.10):
    g = np.random.default_rng(seed)
    if clustering == "low":
        if topology == "loop":
            t = g.uniform(0, 2 * np.pi, n)
            r = 1.0 + g.uniform(-thick, thick, n)
            P = np.c_[r * np.cos(t), r * np.sin(t)]
        else:
            z = g.normal(size=(n, 2)); z /= np.linalg.norm(z, axis=1, keepdims=True)
            P = z * g.uniform(size=(n, 1)) ** 0.5          # uniform disk, same radius scale
    else:
        per = int(np.ceil(n / K))
        if topology == "loop":
            ang = np.arange(K) / K * 2 * np.pi
            cen = np.c_[np.cos(ang), np.sin(ang)]
        else:
            z = g.normal(size=(K, 2)); z /= np.linalg.norm(z, axis=1, keepdims=True)
            cen = z * g.uniform(size=(K, 1)) ** 0.5
        P = np.vstack([c + g.normal(size=(per, 2)) * 0.02 for c in cen])[:n]
    P = P + g.normal(size=P.shape) * noise
    Q = g.normal(size=(2, amb))
    X = P @ Q
    X = X + g.normal(size=X.shape) * noise * np.abs(X).mean()
    return X


if __name__ == "__main__":
    t0 = time.time()
    pd.set_option("display.width", 220)
    print("=== E2: topology x clustering factorial (n=300, ambient 40, noise and scale fixed) ===")
    print("    significance rule: bar lifetime > 95th pct of longest bar over 12 covariance-\n"
          "    matched Gaussian nulls\n")
    rows = []
    for topo in ("noloop", "loop"):
        for clus in ("low", "high"):
            for seed in range(6):
                X = make(topo, clus, seed=seed)
                h = h1_full(X)
                thr, nul_med = null_threshold(X, seed=seed)
                nsig = int((h["life"] > thr).sum()) if len(h["life"]) else 0
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                lab = KMeans(n_clusters=10, n_init=3, max_iter=50, random_state=0).fit_predict(X)
                sil = silhouette_score(X, lab)
                rows.append({"topology": topo, "clustering": clus, "seed": seed,
                             "H1_max": h["H1_max"], "H1_total": h["H1_total"],
                             "n_bars": h["n_bars"], "dom_ratio": h["dom_ratio"],
                             "null_thr": thr, "null_med": nul_med, "n_signif": nsig,
                             "silhouette": sil})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "e2_factorial.csv", index=False)

    agg = D.groupby(["topology", "clustering"]).agg(
        sil=("silhouette", "mean"), H1_max=("H1_max", "mean"), H1_total=("H1_total", "mean"),
        n_bars=("n_bars", "mean"), dom_ratio=("dom_ratio", "mean"),
        thr=("null_thr", "mean"), n_signif=("n_signif", "mean")).round(3)
    print(agg.to_string())

    def m(t, c, col):
        return D[(D.topology == t) & (D.clustering == c)][col].mean()

    print("\n--- Q1: at matched clustering, does H1 detect the real loop? ---")
    for c in ("low", "high"):
        a, b = m("noloop", c, "H1_max"), m("loop", c, "H1_max")
        s = m("loop", c, "n_signif")
        print(f"  clustering={c:5s}  H1_max  no-loop {a:.3f} -> loop {b:.3f}  "
              f"({b/max(a,1e-9):.2f}x);  significant bars on the loop: {s:.1f}")

    print("\n--- Q2: which effect is larger, topology or clustering? ---")
    topo_eff_low = m("loop", "low", "H1_max") / max(m("noloop", "low", "H1_max"), 1e-9)
    topo_eff_high = m("loop", "high", "H1_max") / max(m("noloop", "high", "H1_max"), 1e-9)
    clus_eff_noloop = m("noloop", "high", "H1_max") / max(m("noloop", "low", "H1_max"), 1e-9)
    clus_eff_loop = m("loop", "high", "H1_max") / max(m("loop", "low", "H1_max"), 1e-9)
    print(f"  TOPOLOGY effect (loop vs no-loop):   low clustering {topo_eff_low:.2f}x, "
          f"high clustering {topo_eff_high:.2f}x")
    print(f"  CLUSTERING effect (high vs low):     no-loop {clus_eff_noloop:.2f}x, "
          f"loop {clus_eff_loop:.2f}x")
    for col in ("H1_total",):
        te = m("loop", "low", col) / max(m("noloop", "low", col), 1e-9)
        ce = m("noloop", "high", col) / max(m("noloop", "low", col), 1e-9)
        print(f"  same on {col}: topology {te:.2f}x, clustering {ce:.2f}x")

    print("\n--- Q3: one dominant bar, or many short ones? ---")
    print("  dominance = longest bar / total persistence (1.0 = a single global loop)")
    for t in ("loop", "noloop"):
        for c in ("low", "high"):
            print(f"  {t:7s} {c:5s}  dominance {m(t,c,'dom_ratio'):.3f}   "
                  f"bars {m(t,c,'n_bars'):.1f}   significant {m(t,c,'n_signif'):.1f}")
    print(f"\ndone in {time.time()-t0:.0f}s")
