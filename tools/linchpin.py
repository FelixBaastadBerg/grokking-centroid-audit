"""
LINCHPIN -- clustering control on Monod et al.'s exact statistic (arXiv 2605.06352).

Their object is the rows of the token embedding, and their headline topological quantity is
H1 persistence (max, and total) on that cloud. This asks one question: does the H1 signal carry
information about generalization BEYOND how clustered the embedding cloud is?

Four parts, in order:
  P0  GATES. (a) clustering metrics on clouds of known clusteredness; (b) the partial-correlation
      machinery itself, on synthetic data with a KNOWN common cause -- it must return ~0 when X
      and Y share only a common cause, and must NOT return ~0 when X has independent information.
      Without (b) a "partial correlation collapses" claim is unfalsifiable.
  P1  Reproduce their statistic and its behaviour: H1 max/total on token-embedding rows, and the
      accuracy-leads-topology lag via CCF of first differences (their own timing statistic).
  P2  The control: partial out clustering, see what survives.
  P3  Is the dominant H1 bar a RESOLVED b1=1? Bootstrap it. If a single long bar is a stable
      topological feature it should survive subsampling; if it is the largest gap in a sparse
      sample it should not. Cross-checked against cohomological circular coordinates, which
      recover multi-frequency winding where a single Rips loop does not exist.

Framing note (per the literature sweep): the multi-frequency/torus point is NOT claimed here --
Nanda 2301.05217, Chughtai 2302.03025 and Manifolds of Modular Addition 2512.25060 own it. P3 is
run as CONFIRMATION that our instruments recover the known picture, so that P2's control is
interpretable. The novel claim is P2 alone.
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
# sklearn KMeans thread startup dominates on a 23-point cloud: the k-sweep cost 2.2s per
# checkpoint multi-threaded and 0.02s single-threaded with bounded iterations. Set before
# sklearn/BLAS import or it has no effect.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ripser import ripser

from analysis.topo_runs import TopoRun

OUT = REPO / "RESULTS"


# ------------------------------------------------------------------ Monod's statistic
def monod_prep(E):
    """Their stated preprocessing is 'centered and normalized'; row-vs-global is unspecified in
    the paper, so we center and scale globally (row-normalisation was shown earlier to destroy a
    dimension). Distances are then scaled by the diameter so persistence is comparable."""
    X = np.asarray(E, float)
    X = X - X.mean(0)
    n = np.linalg.norm(X)
    return X / (n / np.sqrt(len(X)) + 1e-12)


def h1_stats(E, maxdim=1):
    X = monod_prep(E)
    D = np.sqrt(np.maximum(((X[:, None] - X[None]) ** 2).sum(-1), 0))
    diam = D.max()
    if diam <= 0:
        return {"H1_max": 0.0, "H1_total": 0.0, "H1_nbars": 0}
    dg = ripser(D / diam, distance_matrix=True, maxdim=maxdim)["dgms"][1]
    if not len(dg):
        return {"H1_max": 0.0, "H1_total": 0.0, "H1_nbars": 0}
    dg = dg[np.isfinite(dg[:, 1])]
    life = dg[:, 1] - dg[:, 0]
    return {"H1_max": float(life.max()) if len(life) else 0.0,
            "H1_total": float(life.sum()) if len(life) else 0.0,
            "H1_nbars": int((life > 0.05).sum())}


# ------------------------------------------------------------------ clustering, label-free
def clustering_metrics(E, kmax=6, seed=0):
    """
    Label-free clusteredness of the embedding cloud. NC1/CDNV need classes, and token embeddings
    have none (each token is one point), so the cloud-level analogues are used here; NC1 proper
    is co-measured on the ACTIVATIONS, where labels exist, in tools/geom_extract.py.
    """
    X = np.asarray(E, float)
    n = len(X)
    from scipy.spatial.distance import pdist
    pw = pdist(X)
    med = np.median(pw) if len(pw) else 1.0
    from sklearn.neighbors import NearestNeighbors
    k = min(10, n - 1)
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    out = {"gap_ratio": float(np.median(d[:, -1]) / med) if med > 0 else np.nan,
           "nn1_ratio": float(np.median(d[:, 1]) / med) if med > 0 else np.nan}
    best, bk = -1.0, np.nan
    for kk in range(2, min(kmax, n - 1) + 1):
        try:
            lab = KMeans(n_clusters=kk, n_init=3, max_iter=50, random_state=seed).fit_predict(X)
            if len(np.unique(lab)) < 2:
                continue
            s = silhouette_score(X, lab)
            if s > best:
                best, bk = s, kk
        except Exception:
            continue
    out["silhouette"] = float(best)
    out["silhouette_k"] = float(bk)
    return out


# ------------------------------------------------------------------ partialling
def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y given z (rank-linear residualisation)."""
    x, y, z = map(lambda v: pd.Series(v).rank().values.astype(float), (x, y, z))
    if np.std(z) < 1e-9:
        return np.nan
    ex = x - np.polyval(np.polyfit(z, x, 1), z)
    ey = y - np.polyval(np.polyfit(z, y, 1), z)
    if ex.std() < 1e-9 or ey.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def ccf_lag(v, acc, max_lag=20):
    """Monod's timing statistic: lag maximising |corr| of FIRST DIFFERENCES. >0 = v moves after."""
    v, acc = np.asarray(v, float), np.asarray(acc, float)
    ok = np.isfinite(v) & np.isfinite(acc)
    if ok.sum() < 10:
        return np.nan, np.nan
    dv, da = np.diff(v[ok]), np.diff(acc[ok])
    if np.std(dv) < 1e-12 or np.std(da) < 1e-12:
        return np.nan, np.nan
    best, bl = -2.0, np.nan
    for L in range(-max_lag, max_lag + 1):
        x, y = (dv[:L], da[-L:]) if L < 0 else ((dv[L:], da[:-L]) if L > 0 else (dv, da))
        if len(x) < 5 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        c = abs(np.corrcoef(x, y)[0, 1])
        if np.isfinite(c) and c > best:
            best, bl = c, L
    return bl, best


# ------------------------------------------------------------------ P0 gates
def gate_partialling():
    print("=== P0b GATE: does the partial-correlation machinery behave? ===")
    print("    (a claim that 'the partial collapses' is worthless unless the estimator can also")
    print("     NOT collapse when there is real independent signal)\n")
    g = np.random.default_rng(0)
    n = 200
    rows = []
    # case 1: pure common cause -- partial MUST go to ~0
    C = g.normal(size=n)
    X = C + g.normal(size=n) * 0.3
    Y = C + g.normal(size=n) * 0.3
    rows.append(("pure common cause (want ~0)", spearmanr(X, Y).statistic, partial_spearman(X, Y, C)))
    # case 2: common cause PLUS a direct link -- partial MUST stay well away from 0
    X2 = C + g.normal(size=n) * 0.3
    Y2 = C + 0.9 * X2 + g.normal(size=n) * 0.3
    rows.append(("common cause + direct link (want >>0)", spearmanr(X2, Y2).statistic,
                 partial_spearman(X2, Y2, C)))
    # case 3: no common cause, direct link only -- partial must stay high
    X3 = g.normal(size=n)
    Y3 = X3 + g.normal(size=n) * 0.3
    rows.append(("direct link, C irrelevant (want high)", spearmanr(X3, Y3).statistic,
                 partial_spearman(X3, Y3, C)))
    # case 4: monotone-nonlinear common cause -- rank partialling should still work
    X4 = np.exp(C) + g.normal(size=n) * 0.1
    Y4 = C ** 3 + g.normal(size=n) * 0.1
    rows.append(("nonlinear monotone common cause (want ~0)", spearmanr(X4, Y4).statistic,
                 partial_spearman(X4, Y4, C)))
    ok = abs(rows[0][2]) < 0.2 and rows[1][2] > 0.5 and rows[2][2] > 0.7 and abs(rows[3][2]) < 0.25
    for nm, r, p in rows:
        print(f"  {nm:42s} raw rho={r:+.3f}   partial={p:+.3f}")
    print(f"\n  GATE {'PASSED' if ok else 'FAILED'}\n")
    return ok


def gate_clustering():
    print("=== P0a GATE: clustering metrics on clouds of KNOWN clusteredness ===\n")
    g = np.random.default_rng(0)
    print(f"  {'case':34s} {'silhouette':>11s} {'gap_ratio':>10s} {'nn1_ratio':>10s}")
    for nm, sig in (("1 blob (no clusters)", None), ("5 clusters, loose", 0.60),
                    ("5 clusters, medium", 0.20), ("5 clusters, tight", 0.05)):
        if sig is None:
            X = g.normal(size=(120, 16))
        else:
            cen = g.normal(size=(5, 16)) * 3
            X = np.vstack([c + g.normal(size=(24, 16)) * sig for c in cen])
        m = clustering_metrics(X)
        print(f"  {nm:34s} {m['silhouette']:11.3f} {m['gap_ratio']:10.3f} {m['nn1_ratio']:10.3f}")
    print()


if __name__ == "__main__":
    ok = gate_partialling()
    gate_clustering()
    if not ok:
        print("PARTIALLING GATE FAILED -- stopping before any claim."); raise SystemExit(1)

    inv = pd.read_csv(OUT / "_inventory.csv")
    sel = inv[inv.arch.isin(["mlp", "transformer"])].copy()
    keep = []
    for (arch, task, p, tf), gg in sel.groupby(["arch", "task", "p", "tf"]):
        keep.extend(gg.sort_values("seed").head(2).to_dict("records"))
    sel = pd.DataFrame(keep)
    print(f"=== P1/P2: Monod statistic + clustering control on {len(sel)} runs ===\n", flush=True)

    rows = []
    for i, rec in enumerate(sel.to_dict("records")):
        try:
            r = TopoRun(REPO / "runs_topo" / rec["dir"])
        except Exception as e:
            continue
        n = len(r.traj["epochs"])
        # UNIFORM stride, not log-spaced. Monod's timing statistic is a cross-correlation of
        # first differences, which assumes a constant sampling interval; with log spacing the
        # conversion lag_in_checkpoints * median(delta_epoch) is not a number of steps at all.
        # That error produced a median lag of +14400 steps with IQR -13100..+16200 (i.e. noise).
        stride = max(1, n // 80)
        ts = np.arange(0, n, stride)
        for t in ts:
            try:
                E = r.embeddings("a", int(t))
            except Exception:
                continue
            h = h1_stats(E)
            c = clustering_metrics(E)
            rows.append({**h, **c, "t": int(t), "epoch": int(r.traj["epochs"][int(t)]),
                         "test_acc": float(r.curves.test_acc.values[int(t)]),
                         "train_acc": float(r.curves.train_acc.values[int(t)]),
                         "arch": rec["arch"], "task": rec["task"], "p": rec["p"],
                         "tf": rec["tf"], "seed": rec["seed"], "dir": rec["dir"]})
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(sel)}] runs done", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "linchpin_series.csv", index=False)
    print(f"\nwrote RESULTS/linchpin_series.csv {D.shape}\n")
