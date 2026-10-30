"""
Rebuild a trained MLP from its checkpoint trajectory and evaluate the loss at ARBITRARY
points in weight space, so the landscape between two checkpoints can be probed directly.

This works because train_topology.py builds the MLP with bias=False, so the saved
[W0, W1, W2] IS the complete parameter vector -- nothing about the model is unrecoverable.
The data split is regenerated from the recorded seed and train_fraction, exactly as training
did it, so "train loss" here means the same 423 examples it meant during training.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

from tools.train_topology import build_task
from models import MLP
from utils import stablemax_cross_entropy, softmax_cross_entropy


class LandscapeRun:
    def __init__(self, d: Path):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "run_meta.json").read_text())
        assert self.meta["arch"] == "mlp", "transformer needs full state dicts; see dirA_tf.py"
        import pandas as pd
        self.curves = pd.read_csv(self.dir / "curves.csv")
        tr = np.load(self.dir / "trajectory.npz")
        self.epochs = tr["epochs"]
        self.W = [tr["W0"], tr["W1"], tr["W2"]]
        self.shapes = [w.shape[1:] for w in self.W]
        self.sizes = [int(np.prod(s)) for s in self.shapes]

        p = self.meta["p"]
        pairs, y, n_sym, _ = build_task(self.meta["task"], p,
                                        sym_n=4 if (self.meta["task"] == "perm" and p == 24) else 5)
        N = len(y)
        X = torch.zeros(N, 2 * n_sym)
        X[torch.arange(N), torch.as_tensor(pairs[:, 0])] = 1.0
        X[torch.arange(N), n_sym + torch.as_tensor(pairs[:, 1])] = 1.0
        Y = torch.as_tensor(y, dtype=torch.long)
        g = torch.Generator().manual_seed(self.meta["seed"])
        perm = torch.randperm(N, generator=g)
        n_tr = int(self.meta["train_fraction"] * N)
        self.Xtr, self.Ytr = X[perm[:n_tr]], Y[perm[:n_tr]]
        self.Xte, self.Yte = X[perm[n_tr:]], Y[perm[n_tr:]]

        h = self.meta["hidden"]
        self.model = MLP(input_size=2 * n_sym, hidden_sizes=[h, h], output_size=n_sym, bias=False)
        self.loss_fn = (stablemax_cross_entropy if self.meta.get("loss_function") == "stablemax"
                        else softmax_cross_entropy)

    # ------------------------------------------------------------------ weights
    def flat(self, i: int) -> np.ndarray:
        return np.concatenate([w[i].ravel() for w in self.W]).astype(np.float64)

    def _load(self, v: np.ndarray):
        o = 0
        with torch.no_grad():
            for prm, s, n in zip(self.model.parameters(), self.shapes, self.sizes):
                prm.copy_(torch.as_tensor(v[o:o + n].reshape(s), dtype=prm.dtype))
                o += n

    def evaluate(self, v: np.ndarray) -> dict:
        self._load(v)
        self.model.eval()
        with torch.no_grad():
            otr, ote = self.model(self.Xtr), self.model(self.Xte)
            return {"train_loss": float(self.loss_fn(otr, self.Ytr, dtype=torch.float64).item()),
                    "test_loss": float(self.loss_fn(ote, self.Yte, dtype=torch.float64).item()),
                    "train_acc": float((otr.argmax(1) == self.Ytr).float().mean()),
                    "test_acc": float((ote.argmax(1) == self.Yte).float().mean())}

    # ------------------------------------------------------------------ landmarks
    def t_grok(self) -> int:
        """Index of the first checkpoint reaching 50% of final test accuracy (our locked rule)."""
        te = self.curves.test_acc.values
        fin = te[-5:].mean()
        if fin < 0.5:
            return -1
        return int(np.argmax(te >= 0.5 * fin))

    def t_memorized(self) -> int:
        """First checkpoint with train acc >= 0.99 (start of the memorization plateau)."""
        tr = self.curves.train_acc.values
        w = np.where(tr >= 0.99)[0]
        return int(w[0]) if len(w) else -1

    def path_losses(self, va: np.ndarray, vb: np.ndarray, n: int = 41) -> dict:
        """Losses along the straight line from va to vb."""
        out = {k: [] for k in ("train_loss", "test_loss", "train_acc", "test_acc")}
        for a in np.linspace(0, 1, n):
            r = self.evaluate((1 - a) * va + a * vb)
            for k in out:
                out[k].append(r[k])
        return {k: np.array(v) for k, v in out.items()}
