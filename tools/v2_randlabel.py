"""
Item 11 -- is any of this specific to grokking, or generic to terminal clustering?

Networks trained on PERMUTED labels cannot learn the group: the input-to-label map carries no
structure, so reaching zero training loss is memorisation and nothing else. Everything else is held
fixed -- same inputs, architecture, optimiser, hyperparameters, class balance and train fraction.
If those networks show the same terminal class separation and the same PH-dimension collapse, then
the confound this paper is about is a property of trained representations in general rather than of
grokking, and the paper's scope should say so.

That is a scope statement either way, so it is reported either way.

Compares, between grokking nets and random-label nets at matched training loss:
  * terminal label silhouette and NC1
  * the training-time change in PH-dim, effective rank and TwoNN
  * the alpha-sweep fold, i.e. the confound itself
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
from tools.e1_centroid import decompose, recompose, measure
from tools.b3_seed_alpha import select, ALPHAS

OUT = REPO / "RESULTS"
N_SUB = 400


def stages(r):
    n = len(r.epochs)
    tg = r.t_grok()
    tm = r.t_memorized()
    return {"init": 0, "memorised": tm if tm > 0 else max(1, n // 10),
            "final": n - 1}, tg


def analyse(dirs, label):
    rows = []
    for d in dirs:
        r = LandscapeRun(REPO / "runs_topo" / d)
        marks, tg = stages(r)
        y_all = r.Ytr.numpy()
        sub = np.sort(np.random.default_rng(0).choice(len(y_all), min(N_SUB, len(y_all)),
                                                      replace=False))
        y = y_all[sub]
        for st, k in marks.items():
            r._load(r.flat(k)); r.model.eval()
            with torch.no_grad():
                r.model(r.Xtr, keep_activations=True)
                H = r.model.activations[-2].numpy()[sub]
            rec = {"group": label, "dir": d, "stage": st, "epoch": int(r.epochs[k]),
                   "t_grok_idx": tg,
                   "train_acc": float(r.curves.train_acc.values[min(k, len(r.curves) - 1)]),
                   "test_acc": float(r.curves.test_acc.values[min(k, len(r.curves) - 1)]),
                   "ph_dim": geom.ph_dim(H)[0], "twonn": geom.twonn(H)}
            try:
                rec["sil_true"] = float(silhouette_score(H, y))
            except Exception:
                rec["sil_true"] = np.nan
            rec.update(geom.clusteredness(H))
            Zc = H - H.mean(0)
            s = np.linalg.svd(Zc, compute_uv=False) ** 2
            s = s[s > 1e-14]; p = s / s.sum()
            rec["eff_rank"] = float(np.exp(-(p * np.log(p)).sum()))
            rows.append(rec)
            if st == "final":
                mu, bet, wit = decompose(H, y)
                vals = {}
                for a in ALPHAS:
                    Z = recompose(mu, bet, wit, a, 1.0, norm="trace")
                    m_ = measure(Z, y)
                    for kk in ("ph_dim", "twonn", "H1_total"):
                        vals.setdefault(kk, []).append(m_[kk])
                for kk, vv in vals.items():
                    vv = np.array([x for x in vv if np.isfinite(x)])
                    rows[-1][f"fold_{kk}"] = (float(vv.max() / max(abs(vv.min()), 1e-9))
                                              if len(vv) > 2 else np.nan)
    return rows


def main():
    t0 = time.time()
    inv = pd.read_csv(OUT / "_inventory.csv")
    grok = list(select(inv).dir)
    rl = sorted(p.name for p in (REPO / "runs_topo").iterdir() if p.name.endswith("__randlab"))
    print("=== Item 11: random-label matched-memorisation control ===")
    print(f"    {len(grok)} grokking runs vs {len(rl)} random-label runs\n", flush=True)
    if not rl:
        print("    no random-label runs found -- train them first"); return

    rows = analyse(grok, "grokking") + analyse(rl, "random_labels")
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "v2_randlabel.csv", index=False)

    print(f"  {'group':<15}{'stage':<11}{'train':>7}{'test':>7}{'silhouette':>12}"
          f"{'gap ratio':>11}{'PH-dim':>9}{'TwoNN':>8}{'eff rank':>10}")
    for gname in ("grokking", "random_labels"):
        for st in ("init", "memorised", "final"):
            g = D[(D.group == gname) & (D.stage == st)]
            if not len(g):
                continue
            print(f"  {gname:<15}{st:<11}{g.train_acc.median():>7.2f}{g.test_acc.median():>7.2f}"
                  f"{g.sil_true.median():>12.3f}{g.gap_ratio.median():>11.3f}"
                  f"{g.ph_dim.median():>9.2f}{g.twonn.median():>8.2f}{g.eff_rank.median():>10.2f}")

    print("\n  training-time change (init -> final), median per group:")
    print(f"  {'group':<15}{'PH-dim':>12}{'eff rank':>12}{'TwoNN':>10}{'silhouette':>13}")
    summ = []
    for gname in ("grokking", "random_labels"):
        g = D[D.group == gname]
        rec = {"group": gname}
        line = f"  {gname:<15}"
        for c, w in (("ph_dim", 12), ("eff_rank", 12), ("twonn", 10), ("sil_true", 13)):
            ch = []
            for d, gg in g.groupby("dir"):
                a = gg[gg.stage == "init"][c]
                b = gg[gg.stage == "final"][c]
                if len(a) and len(b) and np.isfinite(a.iloc[0]) and abs(a.iloc[0]) > 1e-9:
                    ch.append((b.iloc[0] - a.iloc[0]) / abs(a.iloc[0]))
            v = float(np.median(ch)) if ch else np.nan
            rec[c] = v
            line += f"{v:>{w-1}.1%} " if np.isfinite(v) else f"{'--':>{w}}"
        summ.append(rec)
        print(line)

    print("\n  the confound itself -- alpha-sweep fold at convergence:")
    print(f"  {'group':<15}{'PH-dim fold':>14}{'TwoNN fold':>13}{'H1 total fold':>16}")
    for gname in ("grokking", "random_labels"):
        g = D[(D.group == gname) & (D.stage == "final")]
        print(f"  {gname:<15}{g.fold_ph_dim.median():>13.1f}x{g.fold_twonn.median():>12.2f}x"
              f"{g.fold_H1_total.median():>15.1f}x")
        summ_i = next(s for s in summ if s["group"] == gname)
        summ_i.update({"fold_ph_dim": float(g.fold_ph_dim.median()),
                       "fold_twonn": float(g.fold_twonn.median()),
                       "fold_H1_total": float(g.fold_H1_total.median()),
                       "sil_final": float(g.sil_true.median())})
    pd.DataFrame(summ).to_csv(OUT / "v2_randlabel_summary.csv", index=False)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
