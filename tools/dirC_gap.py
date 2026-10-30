"""
DIR C -- RTD as a topological generalization gap.

The objects are input PAIRS, not group elements, so p=23 gives 529 points and the |G| ceiling
that killed P1-P3 does not apply.

For a set S of input pairs we compare two geometries on the SAME indexed points, which is
exactly what RTD needs:

    R_S : pairwise distances of the network's pre-logit representation of the pairs in S
    T_S : the task's own geometry -- distance 0 if two pairs share a label, 1 if they do not

RTD(R_S, T_S) is then how far the representation's topology sits from the task's topology on
S, and the topological generalization gap is

        RTD_gap(t) = RTD(R_test, T_test) - RTD(R_train, T_train)

Setup validation (run first, and reported): a representation that perfectly clusters by label
must give RTD ~ 0, and a random representation must give a large value. Without that the
number is uninterpretable, since T is highly degenerate (many zero distances).
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

import numpy as np, pandas as pd, torch
from analysis.rtd import rtd, dmat
from analysis.lsrun import LandscapeRun

OUT = REPO / "RESULTS"
NPTS = 90


def task_dmat(labels: np.ndarray) -> np.ndarray:
    """0 within a label, 1 across. Normalised to unit mean like dmat(), because RTD uses
    min(A,B) elementwise and would otherwise be dominated by whichever matrix has the
    smaller scale rather than by the difference in shape."""
    D = (labels[:, None] != labels[None, :]).astype(float)
    m = D[np.triu_indices_from(D, 1)].mean()
    return D / m if m > 0 else D


def rep_at(r: LandscapeRun, t: int, X: torch.Tensor) -> np.ndarray:
    r._load(r.flat(t)); r.model.eval()
    with torch.no_grad():
        r.model(X, keep_activations=True)
        return r.model.activations[-2].numpy()      # pre-logit hidden layer


def stratified(labels: np.ndarray, k: int, seed=0) -> np.ndarray:
    """Subsample keeping label coverage, so T is not accidentally all-ones or all-zeros."""
    rng = np.random.default_rng(seed)
    idx = []
    for lab in np.unique(labels):
        w = np.where(labels == lab)[0]
        idx.extend(rng.choice(w, min(len(w), max(1, k // len(np.unique(labels)) + 1)), replace=False))
    idx = np.array(sorted(set(idx)))
    return idx if len(idx) <= k else np.sort(rng.choice(idx, k, replace=False))


if __name__ == "__main__":
    print("=== DIR C SETUP VALIDATION: is RTD(rep, task-geometry) interpretable? ===\n")
    rng = np.random.default_rng(0)
    lab = np.repeat(np.arange(10), 9)
    T = task_dmat(lab)
    perfect = np.eye(10)[lab] * 5.0                              # exact label clusters
    noisy = perfect + rng.normal(size=perfect.shape) * 0.6
    randm = rng.normal(size=(len(lab), 10))
    for nm, R in (("perfect label clusters -> ~0", perfect),
                  ("mildly noisy clusters      ", noisy),
                  ("random representation      ", randm)):
        print(f"  {nm:30s} RTD(R, T) = {rtd(dmat(R), T):8.4f}")

    print("\n=== RTD generalization gap over training ===\n")
    inv = pd.read_csv(OUT / "_inventory.csv")
    G = inv[(inv.arch == "mlp") & (inv.test >= 0.9)].drop_duplicates(["task", "p", "tf"])
    G = G[G.tf.isin([0.7, 0.8])].head(4)
    rows = []
    for _, rec in G.iterrows():
        r = LandscapeRun(REPO / "runs_topo" / rec["dir"])
        tg, n = r.t_grok(), len(r.epochs)
        if tg < 0:
            continue
        ytr, yte = r.Ytr.numpy(), r.Yte.numpy()
        itr, ite = stratified(ytr, NPTS), stratified(yte, NPTS)
        Ttr, Tte = task_dmat(ytr[itr]), task_dmat(yte[ite])
        lg = np.unique(np.round(np.geomspace(1, n - 1, 9)).astype(int))
        ts = np.unique(np.concatenate([[0], lg, [max(0, tg - 2), tg, min(n - 1, tg + 2)], [n - 1]]))
        for t in ts:
            Rtr = rep_at(r, int(t), r.Xtr)[itr]
            Rte = rep_at(r, int(t), r.Xte)[ite]
            a = rtd(dmat(Rtr), Ttr, symmetrise=False)
            b = rtd(dmat(Rte), Tte, symmetrise=False)
            rows.append({"task": rec["task"], "p": rec["p"], "tf": rec["tf"],
                         "t": int(t), "epoch": int(r.epochs[int(t)]), "t_grok": tg,
                         "rtd_train": a, "rtd_test": b, "rtd_gap": b - a,
                         "test_acc": float(r.curves.test_acc.values[int(t)]),
                         "train_acc": float(r.curves.train_acc.values[int(t)])})
        d = pd.DataFrame(rows)
        d = d[(d.task == rec["task"]) & (d.tf == rec["tf"]) & (d.p == rec["p"])]
        print(f"  {rec['task']}/p{rec['p']}/tf{rec['tf']} t_grok=ck{tg}: "
              f"gap pre={d[d.t<tg].rtd_gap.median():7.3f}  post={d[d.t>tg].rtd_gap.median():7.3f}", flush=True)
    D = pd.DataFrame(rows); D.to_csv(OUT / "dirC_gap.csv", index=False)
    D["side"] = np.where(D.t < D.t_grok, "pre-grok", np.where(D.t > D.t_grok, "post-grok", "at-grok"))
    print("\n=== SUMMARY ===")
    print(D.groupby(["task", "p", "side"]).agg(
        n=("rtd_gap", "size"), rtd_train=("rtd_train", "median"), rtd_test=("rtd_test", "median"),
        rtd_gap=("rtd_gap", "median"), test_acc=("test_acc", "median")).round(3).to_string())
    from scipy.stats import spearmanr
    m = D.dropna(subset=["rtd_gap", "test_acc"])
    print(f"\n  spearman(rtd_gap, test_acc) = {spearmanr(m.rtd_gap, m.test_acc).statistic:+.3f} "
          f"(p={spearmanr(m.rtd_gap, m.test_acc).pvalue:.2e}, n={len(m)})")
