"""
Topological analysis of the new `runs_topo/` trajectories.

Two independent windows onto the same phenomenon, deliberately:

  1. PERSISTENT HOMOLOGY of the integer-embedding cloud  -> is there a loop?
  2. FOURIER CONCENTRATION of the same embeddings        -> is it a *circle*?

These are methodologically independent (one is metric/topological, the other is
harmonic), so agreement between them is real evidence rather than one statistic
restated. Nanda et al. 2023 predict that a grokked modular-addition network places
the integers on a circle whose embedding power concentrates on a few Fourier
frequencies; PH should then see exactly one dominant 1-cycle.

Layout note: for `nn.Linear(in, out)` PyTorch stores weight as (out, in), and the
model input is [onehot(a), onehot(b)]. So W0[:, a] is the embedding of integer a
(hidden-dim vector) and W0[:, p + b] the embedding of b.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .tda import cloud_persistence_features, prepare_cloud, persistence_diagrams, diagram_summaries, betti_curve
from .runlib import fit_logistic

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs_topo"


# --------------------------------------------------------------------------
class TopoRun:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "run_meta.json").read_text(encoding="utf-8"))
        self.curves = pd.read_csv(self.path / "curves.csv")
        self._traj = None

    @property
    def traj(self):
        if self._traj is None:
            self._traj = np.load(self.path / "trajectory.npz")
        return self._traj

    @property
    def epochs(self) -> np.ndarray:
        return self.traj["epochs"]

    @property
    def p(self) -> int:
        return int(self.meta["p"])

    def embeddings(self, which: str = "a", idx: int | None = None) -> np.ndarray:
        """
        Per-symbol embedding cloud at checkpoint `idx`: (p, d).

        MLP (one-hot input): column n of the first weight block IS emb(n).
        Transformer: the learned nn.Embedding weight is already (vocab, d_model),
        so the same group-structure analysis applies with no change of probe.
        """
        W0 = self.traj["W0"]
        W = W0[idx] if idx is not None else W0[-1]
        p = self.p
        if self.meta.get("arch") == "transformer":
            return W[:p]                          # (p, d_model), already row-indexed
        block = W[:, :p] if which == "a" else W[:, p:2 * p]
        return block.T                            # (p, hidden)

    @property
    def t_grok(self) -> float:
        """
        Onset = first epoch at which smoothed test accuracy reaches 50% of its FINAL
        value.

        The logistic midpoint (used for the legacy corpus) is deliberately NOT used
        here: these curves rise steeply and then creep, so the 4-parameter logistic
        collapses (it returned lo = -1.18, t0 = 0 on real runs). The threshold
        crossing is shape-agnostic. `t_grok_logistic` is kept as a secondary check.
        """
        c = self.curves.sort_values("epoch")
        acc = c["test_acc"].to_numpy(float)
        if acc.size < 5 or not np.isfinite(acc).any():
            return np.nan
        k = min(5, acc.size // 2 * 2 - 1)
        if k >= 3:
            acc = np.convolve(acc, np.ones(k) / k, mode="same")
        final = float(np.median(acc[-max(3, acc.size // 20):]))
        if final <= 0.05:
            return np.nan                      # never generalized
        hit = np.where(acc >= 0.5 * final)[0]
        return float(c["epoch"].to_numpy()[hit[0]]) if hit.size else np.nan

    @property
    def t_grok_logistic(self) -> float:
        s = pd.Series(self.curves["test_acc"].values, index=self.curves["epoch"].values)
        f = fit_logistic(s)
        return float(f["t0"]) if f else np.nan

    @property
    def t_memorize(self) -> float:
        """First epoch train accuracy reaches 0.99 -- the start of the delay."""
        c = self.curves.sort_values("epoch")
        hit = c[c["train_acc"] >= 0.99]
        return float(hit["epoch"].iloc[0]) if len(hit) else np.nan

    @property
    def groks(self) -> bool:
        return bool(self.curves["test_acc"].max() >= 0.90)


# --------------------------------------------------------------------------
def fourier_concentration(E: np.ndarray, top_k: int = 5) -> dict:
    """
    Nanda-style check that the embedding is CIRCULAR rather than merely loopy.

    E: (p, hidden) embeddings indexed by the integer. Take the DFT along the
    integer axis; a circular representation puts nearly all power on a small number
    of frequencies. Returns the fraction of power in the top-k nonzero frequencies
    and the normalised spectral entropy (low = concentrated = circle-like).
    """
    p = E.shape[0]
    Ec = E - E.mean(axis=0, keepdims=True)
    F = np.fft.rfft(Ec, axis=0)
    power = (np.abs(F) ** 2).sum(axis=1)          # power per frequency, summed over hidden units
    power = power[1:]                              # drop DC (removed by centring anyway)
    tot = power.sum()
    if tot <= 0 or power.size == 0:
        return {"fourier_topk_frac": np.nan, "fourier_entropy": np.nan,
                "fourier_top_freq": np.nan, "fourier_n_eff": np.nan}
    q = power / tot
    srt = np.sort(q)[::-1]
    ent = float(-(q * np.log(q + 1e-300)).sum() / np.log(len(q)))   # normalised to [0,1]
    return {
        "fourier_topk_frac": float(srt[:top_k].sum()),
        "fourier_entropy": ent,
        "fourier_top_freq": float(np.argmax(q) + 1),
        "fourier_n_eff": float(1.0 / (q ** 2).sum()),               # participation ratio of the spectrum
    }


def circularity_score(E: np.ndarray) -> dict:
    """
    Direct geometric test: project onto the top-2 PCs of the dominant Fourier pair
    and measure how close the p points sit to a circle traversed in order.
    """
    p = E.shape[0]
    Ec = E - E.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    P2 = Ec @ Vt[:2].T                       # (p,2)
    r = np.linalg.norm(P2, axis=1)
    if r.mean() <= 0:
        return {"circ_radius_cv": np.nan, "pc2_var_frac": np.nan}
    return {
        # low CV of radius => points lie on a common circle
        "circ_radius_cv": float(r.std() / r.mean()),
        "pc2_var_frac": float((S[:2] ** 2).sum() / (S ** 2).sum()),
    }


# --------------------------------------------------------------------------
def analyse_run(run: TopoRun, maxdim: int = 2, stride: int = 1,
                max_points: int = 400, scale: str = "diameter") -> pd.DataFrame:
    """Per-checkpoint topology of the integer-embedding cloud."""
    rows = []
    eps = run.epochs
    curves = run.curves.set_index("epoch")
    for i in range(0, len(eps), stride):
        E = run.embeddings("a", i)
        feats = cloud_persistence_features(E, maxdim=maxdim, max_points=max_points,
                                           scale=scale, seed=0)
        feats.update(fourier_concentration(E))
        feats.update(circularity_score(E))
        ep = int(eps[i])
        row = {"epoch": ep}
        if ep in curves.index:
            for c in ("train_acc", "test_acc", "train_loss", "test_loss", "w0_norm"):
                row[c] = float(curves.loc[ep, c])
        row.update(feats)
        rows.append(row)
    df = pd.DataFrame(rows)
    for k in ("arch", "task", "p", "seed", "weight_decay", "train_fraction", "group"):
        df[k] = run.meta.get(k)
    df["t_grok"] = run.t_grok
    df["run_id"] = run.path.name
    return df


def betti_curve_stack(run: TopoRun, dim: int = 1, n_grid: int = 60,
                      stride: int = 1, max_points: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CROCKER stack (H-I): Betti_dim as a function of (filtration value, epoch).
    Returns (epochs, grid, matrix[n_epoch, n_grid]).
    """
    eps, curves = run.epochs, []
    used = []
    grid = np.linspace(0.0, 1.0, n_grid)      # clouds are diameter-normalised
    for i in range(0, len(eps), stride):
        E = prepare_cloud(run.embeddings("a", i), max_points=max_points, scale="diameter")
        if E.shape[0] < 10:
            continue
        dgms = persistence_diagrams(E, maxdim=max(1, dim))
        curves.append(betti_curve(dgms[dim], grid))
        used.append(int(eps[i]))
    return np.array(used), grid, np.array(curves)


def load_runs(pattern: str = "*") -> list[TopoRun]:
    out = []
    for d in sorted(RUNS.glob(pattern)):
        if (d / "run_meta.json").exists() and (d / "curves.csv").exists() and (d / "trajectory.npz").exists():
            out.append(TopoRun(d))
    return out
