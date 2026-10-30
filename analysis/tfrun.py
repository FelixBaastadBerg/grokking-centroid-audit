"""
Rebuild a trained Transformer from full_traj.npz so the geometry battery can run on its
activations. Needed because trajectory.npz stores only [embedding.weight, param0, param1] for
transformers, which is not enough to reconstruct the model -- hence the --save_full flag.

Exposes the same interface the MLP path uses (Xtr/Ytr, flat(i), activations at checkpoint i),
so tools/geom_extract.py can treat both architectures identically. The point cloud is the
PRE-LOGIT encoder output at the last sequence position, the transformer analogue of the MLP's
last hidden layer.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tools.train_topology import build_task
from models import Transformer
from utils import stablemax_cross_entropy, softmax_cross_entropy


class TransformerRun:
    def __init__(self, d: Path):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "run_meta.json").read_text())
        assert self.meta["arch"] == "transformer"
        self.curves = pd.read_csv(self.dir / "curves.csv")
        tr = np.load(self.dir / "trajectory.npz")
        self.epochs = tr["epochs"]
        ft = np.load(self.dir / "full_traj.npz", allow_pickle=True)
        self.flat_all = ft["flat"]
        raw = ft["shapes"]
        self.shapes = [tuple(int(v) for v in row if v != -1) for row in raw]
        self.sizes = [int(np.prod(s)) for s in self.shapes]

        p = self.meta["p"]
        pairs, y, n_sym, _ = build_task(self.meta["task"], p,
                                        sym_n=4 if (self.meta["task"] == "perm" and p == 24) else 5)
        N = len(y)
        X = torch.as_tensor(pairs, dtype=torch.long)
        Y = torch.as_tensor(y, dtype=torch.long)
        g = torch.Generator().manual_seed(self.meta.get("split_seed", self.meta["seed"]))
        perm = torch.randperm(N, generator=g)
        n_tr = int(self.meta["train_fraction"] * N)
        self.Xtr, self.Ytr = X[perm[:n_tr]], Y[perm[:n_tr]]
        self.Xte, self.Yte = X[perm[n_tr:]], Y[perm[n_tr:]]

        self.model = Transformer(d_model=self.meta.get("d_model", 128),
                                 num_heads=self.meta.get("num_heads", 4),
                                 num_layers=self.meta.get("num_layers", 1),
                                 vocab_size=n_sym, seq_len=2, use_positional=True)
        self.loss_fn = (stablemax_cross_entropy if self.meta.get("loss_function") == "stablemax"
                        else softmax_cross_entropy)
        self.p = p

    def flat(self, i: int) -> np.ndarray:
        return self.flat_all[i].astype(np.float64)

    def _load(self, v: np.ndarray):
        o = 0
        with torch.no_grad():
            for prm, s, n in zip(self.model.parameters(), self.shapes, self.sizes):
                prm.copy_(torch.as_tensor(v[o:o + n].reshape(s), dtype=prm.dtype))
                o += n

    def activations(self, i: int, X: torch.Tensor) -> np.ndarray:
        """Pre-logit encoder output at the last position -- the transformer's last hidden layer."""
        self._load(self.flat(i))
        self.model.eval()
        with torch.no_grad():
            e = self.model.embedding(X)
            if self.model.pos_embedding is not None:
                e = e + self.model.pos_embedding(torch.arange(X.size(1))).unsqueeze(0)
            return self.model.transformer(e)[:, -1].numpy()

    def t_grok(self) -> int:
        te = self.curves.test_acc.values
        fin = te[-5:].mean()
        return -1 if fin < 0.5 else int(np.argmax(te >= 0.5 * fin))
