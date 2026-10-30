"""
Analysis pipeline for the Brown-style modular-division Transformer runs.

Applies the paper's existing measurement pipeline (analysis/geom.py estimators, the
centroid intervention of tools/e1_centroid.py, and the 5x5 checkpoint recombination of
tools/c1_recombine.py) to the confirmation runs, without changing any estimator
definition. Representation: activations after the second Transformer block's MLP (the
residual stream leaving block 2, before the final LayerNorm), at the answer position, on
the COMPLETE fixed validation set (~7916 points, ~82 per output class at frac 0.15).

Phases (run per seed-subset in parallel, then collect):
  traj     TwoNN / PH-dim / effective rank / MLE-ID / silhouette / L2 / L10 trajectories
  five     the audited five-checkpoint activations: intervention sweep on the converged
           cloud + full 5x5 recombination grid, unaligned and Procrustes-aligned
  collect  merge parts -> the named RESULTS/brown_transformer_*.csv files

Estimator notes recorded for the paper: TwoNN is this paper's implementation (Facco et
al. ratio CDF fit with the top decile discarded, following Ansuini et al.) -- the same
implementation family Brown et al. state they follow; PH-dimension uses the paper's
schedule, which for clouds of >= 1000 points is 11 sizes in [200, 1000], 5 subsamples per
size, seed-0 paired streams. Total H1 persistence is not computed: Vietoris-Rips H1 on
~7900 points is out of reach for the existing pipeline (same limit as Appendix E.6).
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "brown_transformer"))

import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from analysis import geom
from brown_train import GrokkTransformer, ModDivisionDataset, P

RUNS = REPO / "runs_brown"
OUT = REPO / "RESULTS"
PARTS = OUT / "brown_parts"
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]   # same grid as tools/e1_centroid.py
TRAJ_STEPS = sorted(set(list(range(0, 2001, 200)) + list(range(2000, 40_001, 2000))))


# ------------------------------------------------------------------ shared helpers
def run_dirs(tag):
    return sorted(d for d in RUNS.iterdir() if d.is_dir() and d.name.endswith(tag))


def load_meta(d):
    return json.loads((d / "meta.json").read_text(encoding="utf-8"))


def t_grok_step(d):
    """The audited rule on the eval curve: final = mean of last five logged evals; t_grok
    = first logged step with val accuracy >= 0.5 * final. -1 if final < 0.5."""
    c = pd.read_csv(d / "curves.csv")
    fin = c.val_acc.values[-5:].mean()
    if fin < 0.5:
        return -1, fin
    return int(c.step.values[np.argmax(c.val_acc.values >= 0.5 * fin)]), fin


def five_checkpoints(d):
    """The paper's five-checkpoint rule in checkpoint-index space over the saved grid."""
    meta = load_meta(d)
    steps = [s for s in meta["ckpt_steps"] if (d / "ckpt" / f"step{s:06d}.pt").exists()]
    tg_step, _ = t_grok_step(d)
    n = len(steps)
    if tg_step < 0:
        idx = [0, max(1, n // 4), n // 2, (3 * n) // 4, n - 1]
    else:
        tg = int(np.argmax(np.array(steps) >= tg_step))
        idx = sorted(set([0, max(1, tg // 2), tg, (tg + n) // 2, n - 1]))
        while len(idx) < 5:
            idx.append(min(n - 1, max(idx) + 1))
            idx = sorted(set(idx))
    return [steps[i] for i in idx[:5]]


def activations(d, step, ds, Xva):
    meta = load_meta(d)
    from brown_train import build_model
    model = build_model(meta.get("impl", "port"), ds.n_vocab, ds.n_out)
    model.load_state_dict(torch.load(d / "ckpt" / f"step{step:06d}.pt",
                                     map_location="cpu"))
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xva), 2048):
            _, acts = model(Xva[i:i + 2048], return_block_acts=True)
            outs.append(acts[-1][:, -1, :].numpy())
    return np.concatenate(outs).astype(np.float64)


def leakage(H, y, q):
    nn = NearestNeighbors(n_neighbors=q + 1).fit(H)
    _, idx = nn.kneighbors(H)
    neigh = y[idx[:, 1:]]
    return float(np.mean((neigh != y[:, None]).any(axis=1)))


def norm_trace(Z):
    Zc = Z - Z.mean(0)
    return Zc / (np.sqrt((Zc ** 2).sum(1).mean()) + 1e-12)


def measure(Z, y):
    out = {"ph_dim": geom.ph_dim(Z)[0], "twonn": geom.twonn(Z),
           "mle_k10": geom.mle_id(Z, 10), "eff_rank": geom.effective_rank(Z),
           "corr_dim": geom.correlation_dim(Z)[0]}
    try:
        out["sil_true"] = float(silhouette_score(Z, y))
    except Exception:
        out["sil_true"] = np.nan
    return out


def parts_of(H, y):
    cls = np.unique(y)
    M = {c: H[y == c].mean(0) for c in cls}
    cen = np.stack([M[c] for c in y])
    return cen, H - cen


# ------------------------------------------------------------------------ phases
def phase_traj(dirs):
    for d in dirs:
        meta = load_meta(d)
        ds = ModDivisionDataset(P, meta["frac_train"], seed=meta["seed"])
        Xva, Yva = ds.tensors(ds.val_pairs)
        y = Yva.numpy()
        curves = pd.read_csv(d / "curves.csv").set_index("step")
        rows = []
        steps = [s for s in TRAJ_STEPS if (d / "ckpt" / f"step{s:06d}.pt").exists()]
        for s in steps:
            H = activations(d, s, ds, Xva)
            r = {"dir": d.name, "seed": meta["seed"], "step": s,
                 "train_acc": float(curves.train_acc.get(s, np.nan)),
                 "val_acc": float(curves.val_acc.get(s, np.nan)),
                 "L2": leakage(H, y, 2), "L10": leakage(H, y, 10),
                 "n": len(H), "m_per_class": len(H) / P, **measure(norm_trace(H), y)}
            rows.append(r)
            print(f"[traj {d.name}] step {s} done", flush=True)
        pd.DataFrame(rows).to_csv(PARTS / f"traj_{meta['seed']}_{meta['tag']}.csv",
                                  index=False)


def phase_five(dirs):
    for d in dirs:
        meta = load_meta(d)
        ds = ModDivisionDataset(P, meta["frac_train"], seed=meta["seed"])
        Xva, Yva = ds.tensors(ds.val_pairs)
        y = Yva.numpy()
        five = five_checkpoints(d)
        H = {s: activations(d, s, ds, Xva) for s in five}
        tg_step, fin = t_grok_step(d)

        # ---- centroid intervention on the converged cloud (exact e1 construction)
        rows = []
        Hc = H[five[-1]]
        mu = Hc.mean(0)
        cen, res = parts_of(Hc, y)
        for a in ALPHAS:
            Z = norm_trace(mu + a * (cen - mu) + res)
            rows.append({"dir": d.name, "seed": meta["seed"], "alpha": a,
                         "ckpt_step": five[-1], **measure(Z, y)})
            print(f"[interv {d.name}] alpha {a} done", flush=True)
        pd.DataFrame(rows).to_csv(PARTS / f"interv_{meta['seed']}_{meta['tag']}.csv",
                                  index=False)

        # ---- 5x5 recombination, unaligned and Procrustes-aligned to the converged frame
        rows = []
        for frame in ("unaligned", "aligned"):
            if frame == "aligned":
                ref = H[five[-1]] - H[five[-1]].mean(0)
                Ha = {}
                for s in five:
                    Xs = H[s] - H[s].mean(0)
                    R, _ = orthogonal_procrustes(Xs, ref)
                    Ha[s] = Xs @ R
                use = Ha
            else:
                use = H
            cenr = {s: parts_of(use[s], y) for s in five}
            for si, s in enumerate(five):
                for ti, t in enumerate(five):
                    Z = norm_trace(cenr[s][0] + cenr[t][1])
                    rows.append({"dir": d.name, "seed": meta["seed"], "frame": frame,
                                 "s": si, "t": ti, "step_s": s, "step_t": t,
                                 "t_grok_step": tg_step, "final_val": fin,
                                 **measure(Z, y)})
                print(f"[recomb {d.name}] {frame} s={si} done", flush=True)
        pd.DataFrame(rows).to_csv(PARTS / f"recomb_{meta['seed']}_{meta['tag']}.csv",
                                  index=False)


def phase_collect():
    conf_dirs = run_dirs("confirm")
    # both pilot rounds: "__pilot" (round 1) and "__pilot2" (round 2)
    pilot_dirs = run_dirs("pilot") + run_dirs("pilot2")

    # ---- manifest (pilot + confirmation; pilots never enter confirmatory summaries)
    rows = []
    for d in pilot_dirs + conf_dirs:
        meta = load_meta(d)
        tg, fin = t_grok_step(d)
        c = pd.read_csv(d / "curves.csv")
        tr99 = c[c.train_acc >= 0.99]
        rows.append({"dir": d.name, "tag": meta["tag"], "frac_train": meta["frac_train"],
                     "weight_decay": meta["weight_decay"], "seed": meta["seed"],
                     "steps": meta["steps"], "n_train": meta["n_train"],
                     "n_val": meta["n_val"], "final_val_acc": fin,
                     "grokked": fin >= 0.9, "t_grok_step": tg,
                     "t_train99_step": int(tr99.step.iloc[0]) if len(tr99) else -1})
    pd.DataFrame(rows).to_csv(OUT / "brown_transformer_run_manifest.csv", index=False)

    # ---- trajectories + leakage (confirmation only)
    T = pd.concat([pd.read_csv(f) for f in sorted(PARTS.glob("traj_*_confirm.csv"))])
    T.to_csv(OUT / "brown_transformer_trajectories.csv", index=False)
    T[["dir", "seed", "step", "L2", "L10", "n", "m_per_class", "sil_true"]].to_csv(
        OUT / "brown_transformer_leakage.csv", index=False)

    IV = pd.concat([pd.read_csv(f) for f in sorted(PARTS.glob("interv_*_confirm.csv"))])
    IV.to_csv(OUT / "brown_transformer_intervention.csv", index=False)

    RC = pd.concat([pd.read_csv(f) for f in sorted(PARTS.glob("recomb_*_confirm.csv"))])
    RC.to_csv(OUT / "brown_transformer_recombination_raw.csv", index=False)

    # ---- attribution summary
    ESTS = ["ph_dim", "twonn", "eff_rank", "mle_k10", "corr_dim"]
    out = []
    for frame, G in RC.groupby("frame"):
        for est in ESTS:
            per = []
            for d_, g in G.groupby("dir"):
                def f(s_, t_):
                    q = g[(g.s == s_) & (g.t == t_)][est]
                    return float(q.iloc[0]) if len(q) else np.nan
                f00, f10, f01, f11 = f(0, 0), f(4, 0), f(0, 4), f(4, 4)
                diag = f00 - f11
                rel = abs(diag) / max(abs(f00), 1e-12)
                appreciable = rel > 0.05
                rec = {"dir": d_, "seed": int(g.seed.iloc[0]), "estimator": est,
                       "frame": frame, "f00": f00, "f10": f10, "f01": f01, "f11": f11,
                       "diag": diag, "rel_change": rel, "appreciable": appreciable,
                       "grokked": bool(pd.read_csv(OUT / "brown_transformer_run_manifest.csv")
                                       .set_index("dir").loc[d_, "grokked"])}
                if appreciable and np.isfinite(diag) and abs(diag) > 1e-12:
                    first = (f00 - f10) / diag
                    second = (f01 - f11) / diag
                    rec.update({"centroid_first": first, "centroid_second": second,
                                "shapley_between": 0.5 * (first + second),
                                "interaction": first - second})
                    # init -> t_grok (grid indices 0 -> 2)
                    g00, g20, g02, g22 = f(0, 0), f(2, 0), f(0, 2), f(2, 2)
                    dg = g00 - g22
                    if np.isfinite(dg) and abs(dg) > 1e-12:
                        rec["init_tgrok_between"] = 0.5 * ((g00 - g20) + (g02 - g22)) / dg
                        rec["init_tgrok_relchange"] = abs(dg) / max(abs(f00 - f11), 1e-12)
                per.append(rec)
            Pdf = pd.DataFrame(per)
            v = Pdf.get("shapley_between")
            v = v.dropna().values if v is not None else np.array([])
            summ = {"frame": frame, "estimator": est, "n_runs": len(Pdf),
                    "n_grokked": int(Pdf.grokked.sum()),
                    "n_appreciable": int(Pdf.appreciable.sum()), "n_shares": len(v)}
            if len(v) >= 3:
                rng = np.random.default_rng(0)
                boots = [np.median(rng.choice(v, len(v), replace=True))
                         for _ in range(10_000)]
                jack = [np.median(np.delete(v, i)) for i in range(len(v))]
                summ.update({
                    "between_median": float(np.median(v)),
                    "between_min": float(v.min()), "between_max": float(v.max()),
                    "pct_lo": float(np.percentile(boots, 2.5)),
                    "pct_hi": float(np.percentile(boots, 97.5)),
                    "loso_lo": float(np.min(jack)), "loso_hi": float(np.max(jack)),
                    "first_median": float(Pdf.centroid_first.median()),
                    "second_median": float(Pdf.centroid_second.median()),
                    "init_tgrok_median": float(Pdf.init_tgrok_between.median())
                    if "init_tgrok_between" in Pdf else np.nan})
            out.append({**summ})
            Pdf.to_csv(PARTS / f"attr_perseed_{frame}_{est}.csv", index=False)
    pd.DataFrame(out).to_csv(OUT / "brown_transformer_attribution_summary.csv",
                             index=False)
    print("collect done; wrote the 6 CSV outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["traj", "five", "collect"], required=True)
    ap.add_argument("--tag", default="confirm")
    ap.add_argument("--seeds", default="", help="comma list; empty = all")
    args = ap.parse_args()
    torch.set_num_threads(3)
    PARTS.mkdir(parents=True, exist_ok=True)
    if args.phase == "collect":
        phase_collect()
        return
    dirs = run_dirs(args.tag)
    if args.seeds:
        want = {int(s) for s in args.seeds.split(",")}
        # parse the seed from the directory name: robust against run directories that
        # are still being created by concurrently training waves
        dirs = [d for d in dirs
                if (m := re.search(r"seed-(\d+)__", d.name)) and int(m.group(1)) in want]
    (phase_traj if args.phase == "traj" else phase_five)(dirs)


if __name__ == "__main__":
    main()
