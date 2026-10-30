"""
E1 -- centroid intervention on REAL pre-logit activations.

The synthetic intervention constructs dimension and clustering independently. This does the same
thing on real network activations, which closes the synthetic->real bridge: it uses the network's
own within-class geometry and only rescales the class structure around it.

    x_i  =  mu  +  alpha (mu_{y_i} - mu)  +  beta (x_i - mu_{y_i})
                   ^ between-class            ^ within-class

alpha scales class SEPARATION; beta scales within-class SPREAD. The within-class manifold is the
network's real one and its intrinsic dimension is unchanged by alpha (a rigid rescaling of the
centroid offsets does not alter the dimension of the within-class point sets). So sweeping alpha
at fixed beta is a pure clustering manipulation on real geometry.

CONTROL: scaling either factor also changes total scale, which several statistics respond to on
its own. Every cloud is therefore re-normalised to fixed total variance (trace of covariance)
before measurement; a diameter-matched variant is reported alongside so the conclusion cannot be
an artifact of one normalisation choice.

GATE: the same decomposition applied to a SYNTHETIC cloud of known within-class dimension must
leave TwoNN/MLE near the truth as alpha varies. Run first.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from sklearn.metrics import silhouette_score
from analysis import geom
from analysis.lsrun import LandscapeRun
from tools.linchpin import h1_stats

OUT = REPO / "RESULTS"
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
BETAS = [0.25, 0.5, 1.0, 2.0]


def decompose(X, y):
    mu = X.mean(0)
    cls = np.unique(y)
    M = {c: X[y == c].mean(0) for c in cls}
    between = np.stack([M[c] for c in y]) - mu
    within = X - np.stack([M[c] for c in y])
    return mu, between, within


def recompose(mu, between, within, alpha, beta, norm="trace"):
    Z = mu + alpha * between + beta * within
    Zc = Z - Z.mean(0)
    if norm == "trace":
        v = np.sqrt((Zc ** 2).sum(1).mean())
        Z = Zc / (v + 1e-12)
    elif norm == "diameter":
        from scipy.spatial.distance import pdist
        d = pdist(Zc).max()
        Z = Zc / (d + 1e-12)
    return Z


def measure(Z, y):
    out = {}
    pdim, _ = geom.ph_dim(Z)
    cd, _ = geom.correlation_dim(Z)
    out.update({"ph_dim": pdim, "corr_dim": cd, "twonn": geom.twonn(Z),
                "mle_k10": geom.mle_id(Z, 10), "mle_k5": geom.mle_id(Z, 5),
                "pca_pr": geom.pca_participation_ratio(Z),
                "eff_rank": geom.effective_rank(Z)})
    Xc = Z - Z.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    s = s[s > 0]; p = s / s.sum()
    out["spec_entropy"] = float(-(p * np.log(p)).sum())
    h = h1_stats(Z)
    out["H1_max"] = h["H1_max"]; out["H1_total"] = h["H1_total"]; out["H1_nbars"] = h["H1_nbars"]
    out.update(geom.clusteredness(Z))
    try:
        out["sil_true"] = float(silhouette_score(Z, y)) if len(np.unique(y)) > 1 else np.nan
    except Exception:
        out["sil_true"] = np.nan
    return out


def gate():
    """Synthetic control: within-class dimension known and fixed while alpha sweeps."""
    print("=== E1 GATE: centroid sweep on a synthetic cloud of KNOWN within-class dimension ===")
    print("    8 classes, within-class manifold of true dimension 5; alpha scales separation\n")
    g = np.random.default_rng(0)
    d, K, per, amb = 5, 8, 50, 60
    P = g.normal(size=(d, amb))
    cen = g.normal(size=(K, amb)) * 3
    y = np.repeat(np.arange(K), per)
    X = np.vstack([cen[k] + (g.normal(size=(per, d)) @ P) * 0.35 for k in range(K)])
    mu, bet, wit = decompose(X, y)
    print(f"  {'alpha':>6} {'sil':>7} {'ph_dim':>8} {'corr':>7} {'twonn':>7} {'mle10':>7} "
          f"{'H1_tot':>8}   (true within-class d = 5)")
    rows = []
    for a in ALPHAS:
        Z = recompose(mu, bet, wit, a, 1.0)
        m = measure(Z, y)
        rows.append({"alpha": a, **m})
        print(f"  {a:>6.2f} {m['sil_true']:>7.3f} {m['ph_dim']:>8.2f} {m['corr_dim']:>7.2f} "
              f"{m['twonn']:>7.2f} {m['mle_k10']:>7.2f} {m['H1_total']:>8.3f}")
    G = pd.DataFrame(rows)
    tw = G.twonn.values
    ok = (np.nanmax(tw) / max(np.nanmin(tw), 1e-9)) < 2.0
    print(f"\n  TwoNN varies {np.nanmax(tw)/max(np.nanmin(tw),1e-9):.2f}x over the sweep -> "
          f"GATE {'PASSED' if ok else 'FAILED'}")
    G.to_csv(OUT / "e1_gate.csv", index=False)
    return ok


if __name__ == "__main__":
    if not gate():
        print("GATE FAILED -- not running on real data."); raise SystemExit(1)

    inv = pd.read_csv(OUT / "_inventory.csv")
    sel = inv[(inv.arch == "mlp") & (inv.test >= 0.9)].drop_duplicates(["task", "p"])
    # only the ss777 transformers were saved with full parameter state
    tfr = inv[(inv.arch == "transformer") & (inv.test >= 0.9)
              & inv.dir.str.contains("ss777")].drop_duplicates(["task", "p"])
    sel = pd.concat([sel, tfr])
    print(f"\n=== E1: centroid intervention on {len(sel)} real runs ===\n", flush=True)

    rows, t0 = [], time.time()
    for _, rec in sel.iterrows():
        try:
            if rec["arch"] == "mlp":
                r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
                n = len(r.epochs)
                r._load(r.flat(n - 1)); r.model.eval()
                with torch.no_grad():
                    r.model(r.Xtr, keep_activations=True)
                    H = r.model.activations[-2].numpy()
                y = r.Ytr.numpy()
            else:
                from analysis.tfrun import TransformerRun
                r = TransformerRun(REPO / "runs_topo" / rec["dir"])
                H = r.activations(len(r.epochs) - 1, r.Xtr)
                y = r.Ytr.numpy()
        except Exception as e:
            print(f"  skip {rec['dir'][:40]}: {type(e).__name__}", flush=True); continue
        sub = np.sort(np.random.default_rng(0).choice(len(y), min(400, len(y)), replace=False))
        H, y = H[sub], y[sub]
        mu, bet, wit = decompose(H, y)
        for norm in ("trace", "diameter"):
            for a in ALPHAS:
                Z = recompose(mu, bet, wit, a, 1.0, norm=norm)
                rows.append({"arch": rec["arch"], "task": rec["task"], "p": rec["p"],
                             "seed": rec["seed"], "dir": rec["dir"], "norm": norm,
                             "alpha": a, "beta": 1.0, "sweep": "alpha", **measure(Z, y)})
            for b in BETAS:
                Z = recompose(mu, bet, wit, 1.0, b, norm=norm)
                rows.append({"arch": rec["arch"], "task": rec["task"], "p": rec["p"],
                             "seed": rec["seed"], "dir": rec["dir"], "norm": norm,
                             "alpha": 1.0, "beta": b, "sweep": "beta", **measure(Z, y)})
        print(f"  {rec['arch']}/{rec['task']}/p{rec['p']} done ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "e1_centroid.csv", index=False)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "e1_centroid.csv", index=False)
    print(f"\nwrote RESULTS/e1_centroid.csv {D.shape}")
