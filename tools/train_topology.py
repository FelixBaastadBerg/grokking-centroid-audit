"""
Training driver for the topology-of-grokking program.

Why a new driver rather than grokking_experiments.py: that script computes every
expensive metric INLINE at each checkpoint (PH, magnitude O(n^3), Hessian HVPs),
which dominates runtime and locks us into whatever summaries were chosen in 2025.
Here we do the opposite -- train fast, save the raw WEIGHT STATE per checkpoint,
and compute all topology offline. That decouples training cost from analysis cost
and lets any new topological feature be computed retrospectively without retraining.

Key design point (this is what makes H-A cheap and direct):
  For modular arithmetic with one-hot inputs x = [onehot(a), onehot(b)], the first
  layer computes W[:, :p] onehot(a) + W[:, p:] onehot(b). So COLUMN a of the first
  weight block IS the learned embedding of the integer a. The point cloud
  {emb(a) : a in Z_p} has exactly p points -- 113, not 12769 -- so persistent
  homology on it is cheap, and it is precisely the object in which Nanda et al.
  find the circular/Fourier structure. Saving weights therefore gives us the
  representation topology (H-A/H-B) AND the trajectory topology (H-F) for free.

CPU ONLY. device is forced to cpu; CUDA is never touched.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from models import MLP, Transformer            # noqa: E402
from utils import stablemax_cross_entropy, softmax_cross_entropy  # noqa: E402

OUT_ROOT = REPO / "runs_topo"


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------
def _perms(n: int) -> list[tuple]:
    from itertools import permutations
    return sorted(permutations(range(n)))


def build_task(task: str, p: int, sym_n: int = 5):
    """
    Returns (inputs_idx, targets, n_symbols, group_desc).
    inputs_idx is (N,2) integer pairs; the model one-hot encodes them.

    Cyclic tasks live on Z_p (abelian, cyclic -> Nanda's circular representation).
    perm: composition on S_sym_n (non-abelian, NOT cyclic) -- the H-B contrast.
    """
    if task in ("add", "sub", "mul", "div"):
        a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
        a, b = a.ravel(), b.ravel()
        if task == "add":
            y = (a + b) % p
        elif task == "sub":
            y = (a - b) % p
        elif task == "mul":
            y = (a * b) % p
        else:  # div: b != 0, Fermat inverse
            keep = b != 0
            a, b = a[keep], b[keep]
            y = np.array([(int(x) * pow(int(z), p - 2, p)) % p for x, z in zip(a, b)])
        return np.stack([a, b], 1), y, p, f"Z_{p} ({task}), cyclic abelian"

    if task == "perm":
        P = _perms(sym_n)
        idx = {q: i for i, q in enumerate(P)}
        n = len(P)
        A, B, Y = [], [], []
        for i, g in enumerate(P):
            for j, h in enumerate(P):
                comp = tuple(g[h[k]] for k in range(sym_n))
                A.append(i); B.append(j); Y.append(idx[comp])
        return np.stack([np.array(A), np.array(B)], 1), np.array(Y), n, f"S_{sym_n}, non-abelian"

    raise ValueError(task)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="add")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--sym_n", type=int, default=5)
    ap.add_argument("--arch", default="mlp", choices=["mlp", "transformer"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=0.003)
    ap.add_argument("--train_fraction", type=float, default=0.4)
    ap.add_argument("--num_epochs", type=int, default=40000)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--loss", default="stablemax", choices=["stablemax", "cross_entropy"])
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=0,
                    help="0 = full batch (default); >0 = minibatch SGD (H-T5)")
    ap.add_argument("--dense_proj", type=int, default=0,
                    help="if >0, also record a random projection of the full weight vector every step (H-T5b)")
    ap.add_argument("--split_seed", type=int, default=-1,
                    help="Seed for the train/test SPLIT only. Default -1 reuses --seed (old "
                         "behaviour). Setting it decouples the split from the init, so two runs "
                         "can share data and differ only in initialisation -- required for a fair "
                         "cross-seed mode-connectivity comparison.")
    ap.add_argument("--save_full", action="store_true",
                    help="Also save every parameter tensor per checkpoint (full_traj.npz), so "
                         "non-MLP architectures can be rebuilt for landscape probes.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--random_labels", action="store_true",
                    help="permute the targets, destroying the group structure while keeping the "
                         "inputs, class balance and everything else fixed (matched-memorisation "
                         "control)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(2)          # several runs share the box
    device = "cpu"

    pairs, y, n_sym, group_desc = build_task(args.task, args.p, args.sym_n)
    if args.random_labels:
        # Matched-memorisation control: identical inputs, architecture, optimiser and label
        # MARGINAL, but the input->label map carries no group structure. A network fitting this
        # must memorise, so any terminal class separation it shows cannot come from learning the
        # group. Permuting y rather than resampling keeps the class balance exactly.
        y = np.random.default_rng(10_000 + args.seed).permutation(y)
        group_desc += " [RANDOM LABELS: permuted, no group structure]"
    N = len(y)
    X = torch.zeros(N, 2 * n_sym)
    X[torch.arange(N), torch.as_tensor(pairs[:, 0])] = 1.0
    X[torch.arange(N), n_sym + torch.as_tensor(pairs[:, 1])] = 1.0
    Y = torch.as_tensor(y, dtype=torch.long)

    if args.arch == "transformer":
        X = torch.as_tensor(pairs, dtype=torch.long)      # (N,2) token ids

    split_seed = args.seed if args.split_seed < 0 else args.split_seed
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(N, generator=g)
    n_tr = int(args.train_fraction * N)
    tr, te = perm[:n_tr], perm[n_tr:]
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]

    if args.arch == "transformer":
        # Token sequence [a, b]; readout at the last position. The learned
        # nn.Embedding weight IS the per-symbol embedding, so the same
        # group-structure analysis applies with no change of probe.
        model = Transformer(d_model=args.d_model, num_heads=args.num_heads,
                            num_layers=args.num_layers, vocab_size=n_sym,
                            seq_len=2, use_positional=True).to(device)
    else:
        model = MLP(input_size=2 * n_sym, hidden_sizes=[args.hidden, args.hidden],
                    output_size=n_sym, bias=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, args.beta2),
                            eps=1e-25, weight_decay=args.weight_decay)
    loss_fn = stablemax_cross_entropy if args.loss == "stablemax" else softmax_cross_entropy

    rid = (f"arch-{args.arch}__task-{args.task}__p-{n_sym}__seed-{args.seed}"
           f"__lr-{args.lr}__wd-{args.weight_decay}__tf-{args.train_fraction}"
           + (f"__bs-{args.batch_size}" if args.batch_size else "")
           + (f"__{args.tag}" if args.tag else ""))
    out = OUT_ROOT / rid
    out.mkdir(parents=True, exist_ok=True)

    meta = {"arch": args.arch, "task": args.task, "p": n_sym, "group": group_desc,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "train_fraction": args.train_fraction, "seed": args.seed,
            "num_epochs": args.num_epochs, "save_every": args.save_every,
            "hidden": args.hidden, "loss_function": args.loss, "beta2": args.beta2,
            "batch_size_sgd": args.batch_size, "split_seed": split_seed,
            "n_train": int(n_tr), "n_test": int(N - n_tr), "n_total": int(N),
            "random_labels": bool(args.random_labels),
            "_source": "train_topology"}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ckpt_epochs, W0, W1, W2, curves = [], [], [], [], []
    full_rows = []
    dense_P, dense_rows = None, []
    if args.dense_proj:
        nparam = sum(q.numel() for q in model.parameters())
        g2 = torch.Generator().manual_seed(args.seed + 99)
        dense_P = torch.randn(nparam, args.dense_proj, generator=g2) / np.sqrt(args.dense_proj)
    t0 = time.time()
    gen = torch.Generator().manual_seed(args.seed + 1)
    for ep in range(args.num_epochs + 1):
        model.train()
        if args.batch_size and args.batch_size < len(Ytr):
            idx = torch.randperm(len(Ytr), generator=gen)[:args.batch_size]
            xb, yb = Xtr[idx], Ytr[idx]
        else:
            xb, yb = Xtr, Ytr
        opt.zero_grad()
        logits = model(xb)
        if args.arch == 'transformer': logits = logits[:, -1]
        loss = loss_fn(logits, yb, dtype=torch.float64)
        loss.backward(); opt.step()

        if dense_P is not None:
            with torch.no_grad():
                flat = torch.cat([q.detach().reshape(-1) for q in model.parameters()])
                dense_rows.append((flat @ dense_P).numpy().astype(np.float32))

        if ep % args.save_every == 0 or ep == args.num_epochs:
            model.eval()
            with torch.no_grad():
                otr, ote = model(Xtr), model(Xte)
                if args.arch == 'transformer': otr, ote = otr[:, -1], ote[:, -1]
                tr_acc = (otr.argmax(1) == Ytr).float().mean().item()
                te_acc = (ote.argmax(1) == Yte).float().mean().item()
                te_loss = loss_fn(ote, Yte, dtype=torch.float64).item()
                ws = [q.detach().cpu().numpy().astype(np.float32) for q in model.parameters()]
                if args.arch == 'transformer':
                    ws = [model.embedding.weight.detach().cpu().numpy().astype(np.float32)] + ws[:2]
            if args.save_full:
                with torch.no_grad():
                    full_rows.append(np.concatenate(
                        [q.detach().cpu().numpy().astype(np.float32).ravel()
                         for q in model.parameters()]))
            ckpt_epochs.append(ep); W0.append(ws[0]); W1.append(ws[1]); W2.append(ws[2])
            curves.append({"epoch": ep, "train_loss": float(loss.item()),
                           "test_loss": te_loss, "train_acc": tr_acc, "test_acc": te_acc,
                           "w0_norm": float(np.linalg.norm(ws[0])),
                           "w1_norm": float(np.linalg.norm(ws[1])),
                           "w2_norm": float(np.linalg.norm(ws[2]))})
            if ep % (args.save_every * 25) == 0:
                print(f"  ep {ep:>6} train {tr_acc:.3f} test {te_acc:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    import pandas as pd
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    if dense_rows:
        np.savez_compressed(out / "dense_traj.npz", proj=np.stack(dense_rows))
    if full_rows:
        shapes = [tuple(q.shape) for q in model.parameters()]
        names = [n for n, _ in model.named_parameters()]
        np.savez_compressed(out / "full_traj.npz", flat=np.stack(full_rows),
                            shapes=np.array([list(s) + [-1] * (4 - len(s)) for s in shapes]),
                            names=np.array(names))
    np.savez_compressed(out / "trajectory.npz",
                        epochs=np.array(ckpt_epochs),
                        W0=np.stack(W0), W1=np.stack(W1), W2=np.stack(W2))
    final = curves[-1]
    print(f"DONE {rid}  final train {final['train_acc']:.3f} test {final['test_acc']:.3f}  "
          f"{time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
