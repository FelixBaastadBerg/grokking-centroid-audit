"""
v6.4 Phase 2.4 -- is the Transformer result a replication under Brown et al.'s own estimator?

v6.3 says its TwoNN is "the same implementation family" as Brown et al. That is weaker than a
replication claim needs, so the fitting and trimming rules are pinned down here.

What Brown et al. (arXiv:2211.13239) specify, from their Section 2 and Eq. (1):

  * estimator      TwoNN of Facco et al., "following the implementation described in
                   Ansuini et al."
  * statistic      mu = r^(2)/r^(1), the ratio of each point's second- to first-nearest
                   neighbour distance, Euclidean.
  * fit            "the slope of a straight line THROUGH THE ORIGIN, fit to the observed
                   coordinates {(log(mu_i), -log(1 - F^emp(mu_i)))}_i".
  * trimming       none stated.
  * population     validation data; LLID is the last layer. For the transformers the measured
                   object is "the activations after the second transformer block's MLP".

Two things follow. The fit form is identical to this paper's (both are least squares through
the origin, so both reduce to sum(x*y)/sum(x*x)). The trimming is NOT identical: this paper
discards the top decile of mu, Brown et al. state no discard. A literal no-discard fit with
F^emp = i/N is undefined, because the largest mu gives -log(0); some convention is forced. Two
defensible readings of "the observed coordinates" are implemented and both are reported, so the
comparison does not hinge on a choice this paper made on Brown et al.'s behalf:

  twonn_paper     F = i/N, top decile discarded          (this paper, analysis/geom.py)
  twonn_brown_min F = i/N, only the single final point dropped -- the least trimming that
                  leaves the fit defined
  twonn_brown_pl1 F = i/(N+1), nothing dropped -- the plotting-position convention that makes
                  every observed coordinate finite

Computed on every saved checkpoint of the ten confirmation seeds, then pushed through the same
plateau rule and the same 2x2 attribution, both factor orders.

Analysis-only: saved checkpoints, no training. Writes RESULTS_v6_4/.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "brown_transformer"))

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from analysis import geom

OUT = REPO / "RESULTS_v6_4"
OUT.mkdir(exist_ok=True)
RUNS = REPO / "runs_brown"
PARTS = OUT / "brown_exact_parts"


# --------------------------------------------------------------------------- estimators
def _mu(X):
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 0
    mu = r2[ok] / r1[ok]
    return np.sort(mu[mu > 1])


def twonn_variant(X, rule):
    """One TwoNN family, differing only in the empirical-CDF convention and the trimming."""
    mu = _mu(np.asarray(X, float))
    n = len(mu)
    if n < 10:
        return np.nan
    if rule == "paper":
        F = np.arange(1, n + 1) / n
        keep = int(n * 0.9)
    elif rule == "brown_min":
        F = np.arange(1, n + 1) / n
        keep = n - 1
    elif rule == "brown_pl1":
        F = np.arange(1, n + 1) / (n + 1)
        keep = n
    else:
        raise ValueError(rule)
    x, y = np.log(mu[:keep]), -np.log(1 - F[:keep])
    return float(np.sum(x * y) / np.sum(x * x))


RULES = ("paper", "brown_min", "brown_pl1")


def leakage(H, y, q):
    nn = NearestNeighbors(n_neighbors=q + 1).fit(H)
    _, idx = nn.kneighbors(H)
    return float(np.mean((y[idx[:, 1:q + 1]] != y[:, None]).any(1)))


# --------------------------------------------------------------------------- run handling
def load_run(d):
    from brown_analyze import load_meta
    from brown_train import ModDivisionDataset, P
    meta = load_meta(d)
    ds = ModDivisionDataset(P, meta["frac_train"], seed=meta["seed"])
    Xva, Yva = ds.tensors(ds.val_pairs)
    return meta, ds, Xva, np.asarray(Yva)


def parts(H, y):
    M = {c: H[y == c].mean(0) for c in np.unique(y)}
    cen = np.stack([M[c] for c in y])
    return cen, H - cen


def trace_norm(Z):
    Zc = Z - Z.mean(0)
    return Zc / (np.sqrt((Zc ** 2).sum(1).mean()) + 1e-12)


def steps_of(d):
    return sorted(int(p.stem[4:]) for p in (d / "ckpt").glob("step*.pt"))


def plateau_step(curves, t_grok_step, saved):
    """
    v6.3's committed rule: the last CHECKPOINT before t_grok with val <= 0.10 and train >= 0.99.
    The curve is logged more finely than checkpoints are saved, so the rule has to be evaluated
    on the saved steps; searching the whole curve picks steps with no stored weights.
    """
    c = curves[curves.step.isin(saved) & (curves.step < t_grok_step)
               & (curves.val_acc <= 0.10) & (curves.train_acc >= 0.99)]
    return int(c.step.max()) if len(c) else -1


def one_run(d):
    from brown_analyze import activations
    meta, ds, Xva, y = load_run(d)
    curves = pd.read_csv(d / "curves.csv")
    steps = steps_of(d)

    # ---- trajectories: every saved checkpoint, all three estimator rules
    traj = []
    for st in steps:
        H = activations(d, st, ds, Xva)
        Z = trace_norm(H)
        row = {"dir": d.name, "seed": meta["seed"], "step": st}
        for rule in RULES:
            row[f"twonn_{rule}"] = twonn_variant(Z, rule)
        # PH-dimension trajectories already exist in RESULTS/brown_transformer_trajectories.csv;
        # recomputing them here would triple the runtime for no new information
        row["L2"] = leakage(Z, y, 2)
        traj.append(row)
        del H, Z
    T = pd.DataFrame(traj)
    # written per worker: the trajectory pass is the expensive half, and a failure in the
    # attribution pass below must not discard it
    PARTS.mkdir(exist_ok=True)
    T.to_csv(PARTS / f"traj_{meta['seed']}.csv", index=False)

    # ---- plateau -> convergence attribution, both factor orders, all three rules
    man = pd.read_csv(REPO / "RESULTS" / "brown_transformer_run_manifest.csv")
    rec = man[man.dir == d.name].iloc[0]
    tg = int(rec.t_grok_step)
    grokked = bool(rec.grokked)
    attr = []
    if grokked and tg > 0:
        sp = plateau_step(curves, tg, set(steps))
        sf = steps[-1]
        if sp > 0:
            Hp = activations(d, sp, ds, Xva)
            Hf = activations(d, sf, ds, Xva)
            cp, rp = parts(Hp, y)
            cf, rf = parts(Hf, y)
            clouds = {"00": cp + rp, "10": cf + rp, "01": cp + rf, "11": cf + rf}
            for rule in RULES:
                v = {k: twonn_variant(trace_norm(Z), rule) for k, Z in clouds.items()}
                diag = v["00"] - v["11"]
                first = v["00"] - v["10"]          # centroids advanced first
                second = v["01"] - v["11"]         # centroids advanced second
                phi_b = 0.5 * (first + second)
                attr.append({"dir": d.name, "seed": meta["seed"], "rule": rule,
                             "step_plateau": sp, "step_final": sf, "t_grok_step": tg,
                             "L2_plateau": leakage(trace_norm(Hp), y, 2),
                             "L2_final": leakage(trace_norm(Hf), y, 2),
                             "f00": v["00"], "f10": v["10"], "f01": v["01"], "f11": v["11"],
                             "raw_change": diag,
                             "centroid_first_raw": first, "centroid_second_raw": second,
                             "phi_between_raw": phi_b, "phi_within_raw": diag - phi_b,
                             "between_share": phi_b / diag if abs(diag) > 1e-9 else np.nan,
                             "within_share": (1 - phi_b / diag) if abs(diag) > 1e-9 else np.nan,
                             "centroid_first_share": first / diag if abs(diag) > 1e-9 else np.nan,
                             "centroid_second_share": (second / diag if abs(diag) > 1e-9
                                                       else np.nan)})
            del Hp, Hf
    A = pd.DataFrame(attr)
    if len(A):
        A.to_csv(PARTS / f"attr_{meta['seed']}.csv", index=False)
    return T, A


def main():
    from multiprocessing import Pool
    t0 = time.time()
    dirs = sorted(p for p in RUNS.iterdir()
                  if p.is_dir() and p.name.endswith("__confirm"))
    print(f"=== v6.4 Brown-exact TwoNN on {len(dirs)} confirmation seeds ===", flush=True)
    TT, AA = [], []
    with Pool(processes=5) as pool:
        for i, (T, A) in enumerate(pool.imap_unordered(one_run, dirs), 1):
            TT.append(T); AA.append(A)
            print(f"  [{i}/{len(dirs)}] seed {T.seed.iloc[0]} "
                  f"({len(T)} ckpts, {time.time()-t0:.0f}s)", flush=True)

    T = pd.concat(TT); A = pd.concat(AA)
    T.to_csv(OUT / "brown_exact_trajectories.csv", index=False)
    A.to_csv(OUT / "brown_exact_attribution.csv", index=False)

    print("\n--- plateau->convergence TwoNN attribution, by estimator rule ---")
    print(f"  {'rule':<12}{'n':>4}{'decline':>10}{'within share':>15}"
          f"{'  range':>18}{'  cent-first':>13}{'  cent-second':>14}")
    for rule, g in A.groupby("rule"):
        print(f"  {rule:<12}{len(g):>4}{g.raw_change.median():>10.2f}"
              f"{g.within_share.median():>14.0%}"
              f"   [{g.within_share.min():>5.0%},{g.within_share.max():>6.0%}]"
              f"{g.centroid_first_share.median():>12.0%}"
              f"{g.centroid_second_share.median():>13.0%}")
    print(f"\ndone ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
