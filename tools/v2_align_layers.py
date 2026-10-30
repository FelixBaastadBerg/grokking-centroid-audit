"""
Item 2 -- does the checkpoint-recombination result survive frame alignment?
Item 10 -- where in depth does the failure regime first appear?

ITEM 2. C1 builds x~ = mu_{y,s} + (x_t - mu_{y,t}), mixing centroids from checkpoint s with
residuals from checkpoint t. That is only meaningful if s and t share a coordinate frame, and hidden
representations rotate freely during training: nothing pins the basis. If the frames differ by a
rotation, the residuals are being added in the wrong directions and the attribution is measuring
misalignment rather than geometry.

The fix is to align every checkpoint to a common reference before recombining. Orthogonal Procrustes
gives the rotation R minimising ||H_t R - H_ref||_F, which is the right group here because the
statistics involved are all invariant to a global rotation -- so aligning cannot by itself change
any single-checkpoint measurement, only the cross-checkpoint mixture. We report alignment quality,
and the attribution both ways. Linear CKA is reported alongside as an alignment-free similarity.

ITEM 10. The same machinery gives the depth sweep for free: the MLP exposes hidden-1, hidden-2
(the paper's object) and the logits, so we measure clusteredness and the estimator response at each,
at several training stages, and ask where the confounded regime is first entered.

Analysis-only: retained checkpoints.
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
from tools.e1_centroid import measure
from tools.b3_seed_alpha import select
from tools.c1_recombine import checkpoints, parts, ESTS

OUT = REPO / "RESULTS"
N_SUB = 400
LAYERS = {"hidden1": 0, "hidden2": 1, "logits": 2}


# ---------------------------------------------------------------- alignment
def procrustes(A, B):
    """Rotation R minimising ||A R - B||_F, both column-centred. Returns R and the fit quality."""
    A0, B0 = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    R = U @ Vt
    num = np.linalg.norm(A0 @ R - B0) ** 2
    den = np.linalg.norm(B0) ** 2
    return R, float(1.0 - num / max(den, 1e-12))       # 1 = perfect, 0 = no better than the mean


def linear_cka(A, B):
    A0, B0 = A - A.mean(0), B - B.mean(0)
    num = np.linalg.norm(A0.T @ B0, "fro") ** 2
    den = np.linalg.norm(A0.T @ A0, "fro") * np.linalg.norm(B0.T @ B0, "fro")
    return float(num / max(den, 1e-12))


def acts(r, i, X, layer=1):
    r._load(r.flat(i)); r.model.eval()
    with torch.no_grad():
        r.model(X, keep_activations=True)
        return r.model.activations[layer].numpy()


# ---------------------------------------------------------------- item 2
def item2():
    inv = pd.read_csv(OUT / "_inventory.csv")
    sel = select(inv)
    print(f"=== Item 2: aligned vs unaligned checkpoint recombination, {len(sel)} seeds ===",
          flush=True)
    rows, qual, t0 = [], [], time.time()
    for i, (_, rec) in enumerate(sel.iterrows()):
        r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
        y_all = r.Ytr.numpy()
        sub = np.sort(np.random.default_rng(0).choice(len(y_all), min(N_SUB, len(y_all)),
                                                      replace=False))
        y = y_all[sub]
        idx = checkpoints(r)
        H = {k: acts(r, k, r.Xtr)[sub] for k in idx}
        ref = H[idx[-1]]                                   # align everything to convergence
        R, A = {}, {}
        for k in idx:
            Rk, q = procrustes(H[k], ref)
            R[k] = Rk
            A[k] = (H[k] - H[k].mean(0)) @ Rk + H[k].mean(0)
            qual.append({"dir": rec["dir"], "ckpt": idx.index(k), "epoch": int(r.epochs[k]),
                         "procrustes_r2": q, "cka": linear_cka(H[k], ref),
                         "procrustes_r2_self": procrustes(A[k], ref)[1]})
        for tag, D in (("unaligned", H), ("aligned", A)):
            cen, res = {}, {}
            for k in idx:
                cen[k], res[k] = parts(D[k], y)
            for s in idx:
                for t in idx:
                    Z = cen[s] + res[t]
                    Zc = Z - Z.mean(0)
                    Z = Zc / (np.sqrt((Zc ** 2).sum(1).mean()) + 1e-12)
                    rows.append({"dir": rec["dir"], "seed": int(rec["seed"]), "frame": tag,
                                 "s": idx.index(s), "t": idx.index(t), **measure(Z, y)})
        print(f"  [{i+1}/{len(sel)}] ({time.time()-t0:.0f}s)", flush=True)

    D = pd.DataFrame(rows); D.to_csv(OUT / "v2_align_recombine.csv", index=False)
    Q = pd.DataFrame(qual); Q.to_csv(OUT / "v2_align_quality.csv", index=False)

    print("\n  alignment quality (median over runs), by checkpoint index:")
    print(f"  {'ckpt':>5}{'epoch':>9}{'Procrustes R2 to final':>26}{'linear CKA':>13}")
    for c, g in Q.groupby("ckpt"):
        print(f"  {c:>5}{int(g.epoch.median()):>9}{g.procrustes_r2.median():>26.3f}"
              f"{g.cka.median():>13.3f}")

    print("\n  attribution: share of the diagonal change reproduced by advancing centroids alone")
    print(f"  {'estimator':<12}{'unaligned':>12}{'aligned':>12}{'  aligned IQR':>20}")
    summ = []
    for est in ESTS:
        row = {"estimator": est}
        for tag in ("unaligned", "aligned"):
            g0 = D[D.frame == tag]
            sh = []
            for d, g in g0.groupby("dir"):
                lo, hi = g.s.min(), g.s.max()
                def v(s_, t_):
                    q = g[(g.s == s_) & (g.t == t_)][est]
                    return float(q.iloc[0]) if len(q) else np.nan
                diag = v(lo, lo) - v(hi, hi)
                betw = v(lo, lo) - v(hi, lo)
                if np.isfinite(diag) and abs(diag) > 1e-9:
                    sh.append(betw / diag)
            row[tag] = float(np.median(sh)) if sh else np.nan
            row[tag + "_lo"] = float(np.percentile(sh, 25)) if sh else np.nan
            row[tag + "_hi"] = float(np.percentile(sh, 75)) if sh else np.nan
            row[tag + "_n"] = len(sh)
        summ.append(row)
        print(f"  {est:<12}{row['unaligned']:>11.0%}{row['aligned']:>12.0%}"
              f"   [{row['aligned_lo']:>4.0%},{row['aligned_hi']:>4.0%}]")
    pd.DataFrame(summ).to_csv(OUT / "v2_align_summary.csv", index=False)


# ---------------------------------------------------------------- item 10
def item10():
    inv = pd.read_csv(OUT / "_inventory.csv")
    sel = select(inv)
    print(f"\n=== Item 10: depth sweep, {len(sel)} seeds ===", flush=True)
    from tools.e1_centroid import decompose, recompose
    ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    rows, t0 = [], time.time()
    for i, (_, rec) in enumerate(sel.iterrows()):
        r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
        n = len(r.epochs); tg = r.t_grok()
        marks = {"init": 0, "pre": max(1, tg // 2), "t_grok": tg if tg > 0 else n // 2,
                 "final": n - 1}
        y_all = r.Ytr.numpy()
        sub = np.sort(np.random.default_rng(0).choice(len(y_all), min(N_SUB, len(y_all)),
                                                      replace=False))
        y = y_all[sub]
        for lname, li in LAYERS.items():
            for stage, k in marks.items():
                H = acts(r, k, r.Xtr, layer=li)[sub]
                base = {"dir": rec["dir"], "seed": int(rec["seed"]), "layer": lname,
                        "stage": stage, "epoch": int(r.epochs[k])}
                try:
                    sil = float(silhouette_score(H, y))
                except Exception:
                    sil = np.nan
                base["sil_true"] = sil
                base.update(geom.clusteredness(H))
                base["twonn"] = geom.twonn(H)
                base["ph_dim"] = geom.ph_dim(H)[0]
                # the alpha response, only at the stages where it is meaningful
                if stage in ("init", "final"):
                    mu, bet, wit = decompose(H, y)
                    vals = {}
                    for a in ALPHAS:
                        Z = recompose(mu, bet, wit, a, 1.0, norm="trace")
                        m_ = measure(Z, y)
                        for kk in ("ph_dim", "twonn", "H1_total"):
                            vals.setdefault(kk, []).append(m_[kk])
                    for kk, vv in vals.items():
                        vv = np.array([x for x in vv if np.isfinite(x)])
                        base[f"fold_{kk}"] = (float(vv.max() / max(abs(vv.min()), 1e-9))
                                              if len(vv) > 2 else np.nan)
                rows.append(base)
        print(f"  [{i+1}/{len(sel)}] ({time.time()-t0:.0f}s)", flush=True)

    L = pd.DataFrame(rows); L.to_csv(OUT / "v2_layers.csv", index=False)
    print("\n  clusteredness and estimator response by depth (median over seeds):")
    print(f"  {'layer':<9}{'stage':<8}{'silhouette':>12}{'gap ratio':>11}{'TwoNN':>9}"
          f"{'PH-dim':>9}{'fold PH':>10}{'fold TwoNN':>12}")
    for lname in LAYERS:
        for st in ("init", "pre", "t_grok", "final"):
            g = L[(L.layer == lname) & (L.stage == st)]
            if not len(g):
                continue
            fp = g.fold_ph_dim.median() if "fold_ph_dim" in g else np.nan
            ft = g.fold_twonn.median() if "fold_twonn" in g else np.nan
            print(f"  {lname:<9}{st:<8}{g.sil_true.median():>12.3f}{g.gap_ratio.median():>11.3f}"
                  f"{g.twonn.median():>9.2f}{g.ph_dim.median():>9.2f}"
                  + (f"{fp:>10.1f}x" if np.isfinite(fp) else f"{'--':>11}")
                  + (f"{ft:>11.2f}x" if np.isfinite(ft) else f"{'--':>12}"))


if __name__ == "__main__":
    t0 = time.time()
    item2()
    item10()
    print(f"\nwrote v2_align_*.csv, v2_layers.csv   ({time.time()-t0:.0f}s)")
