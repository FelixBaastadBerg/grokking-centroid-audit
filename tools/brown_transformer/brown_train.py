"""
Brown-style modular-division grokking run: a faithful port of the implementation Brown et
al. (arXiv:2211.13239) cite for their grokking experiment, Sea-Snell/grokking
(github.com/Sea-Snell/grokking), inspected at clone time (see ../../external/).

Specification followed (Brown et al., Sec. 3.2 and Appendix A; Sea-Snell defaults):
  task            modular division mod p = 97: c = a * b^(p-2) mod p, a in 0..96, b in 1..96
  sequence        [a, 'o', b, '='] (4 tokens), causal mask, prediction read at the last
                  position; vocab = ['o', '='] + residues; 97 output classes
  split           random shuffle of the 9312 (a, b) pairs; train = first frac_train
  model           Transformer decoder, 2 blocks, 4 heads, hidden 128, attn_dim 32,
                  intermediate 512, GELU, dropout 0.1, pre-norm, learned positions,
                  embedding scaled by sqrt(hidden), final LayerNorm, linear readout
  optimiser       AdamW, lr 1e-3 constant after 10 warmup steps, betas (0.9, 0.98),
                  weight decay per configuration (the cited implementation uses AdamW's
                  decoupled weight decay; Brown's text says "ADAM")
  batching        512 examples sampled from the train pairs with replacement, 40,000 steps

Recorded deviations from Sea-Snell (full list in the experiment README):
  * vocabulary order is fixed to ['o', '='] + [0..96] instead of Python set-iteration
    order (identical for CPython small ints);
  * train/val accuracy is evaluated on the COMPLETE train and validation sets at logging
    steps, not on 8 random validation batches;
  * the split, batch sampling, and torch RNGs are seeded per run for reproducibility
    (Sea-Snell leaves the global RNGs unseeded);
  * model state dicts are saved on a fixed checkpoint schedule for later activation
    extraction (activations after the second block's MLP, i.e. the residual stream after
    block 2 before the final LayerNorm, at the last token position, on validation data).

Run:  python tools/brown_transformer/brown_train.py --frac 0.15 --wd 0.001 --seed 0

PILOT ROUND 1 DIAGNOSIS (recorded in config_selection.md): with --impl port --optim adamw
(the Sea-Snell training loop), none of the four pilots generalised within 40k steps.
Brown et al.'s Appendix A specifies the model was "obtained from the huggingface
transformers library" and trained with "the ADAM optimizer" -- only the DATASET comes
from Sea-Snell. --impl hf therefore uses a HuggingFace GPT-2 backbone
(vocab 99, n_positions 5, n_embd 128, n_layer 2, n_head 4, default dropout 0.1 and
init std 0.02) with a linear classification head to the 97 output classes, and
--optim adam uses torch.optim.Adam with COUPLED weight decay, per Brown's text. The
representation for analysis is the block-2 output (residual stream after the second
block's MLP, before ln_f), captured by a forward hook, at the answer position.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
P = 97
MAX_LEN = 5
STEPS = 40_000
BSIZE = 512
LR = 1e-3
BETAS = (0.9, 0.98)
WARMUP = 10
EVAL_EVERY = 100
# state dicts: every 100 steps to 2000 (the fast early phase), then every 500
CKPT_STEPS = sorted(set(list(range(0, 2001, 100)) + list(range(2500, STEPS + 1, 500))))


# ------------------------------------------------------------------ dataset (Sea-Snell)
class ModDivisionDataset:
    def __init__(self, p: int, frac_train: float, seed: int):
        self.p = p
        self.elems1 = list(range(p))          # dividend a
        self.elems2 = list(range(1, p))       # divisor b != 0
        self.idx2vocab = ["o", "="] + list(range(p))
        self.vocab2idx = {v: i for i, v in enumerate(self.idx2vocab)}
        self.n_vocab = len(self.idx2vocab)    # 99
        self.n_out = p                        # 97 output classes
        idxs = list(range(len(self.elems1) * len(self.elems2)))
        random.Random(seed).shuffle(idxs)
        cut = int(len(idxs) * frac_train)
        self.train_pairs, self.val_pairs = idxs[:cut], idxs[cut:]

    def example(self, idx: int):
        a = self.elems1[idx // len(self.elems2)]
        b = self.elems2[idx % len(self.elems2)]
        c = (a * pow(b, self.p - 2, self.p)) % self.p
        x = [self.vocab2idx[a], self.vocab2idx["o"], self.vocab2idx[b], self.vocab2idx["="]]
        return x, c

    def tensors(self, pair_idxs):
        xs, ys = zip(*(self.example(i) for i in pair_idxs))
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


# ------------------------------------------------------------- model (Sea-Snell port)
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, hidden_dim, attn_dim, dropout=0.1):
        super().__init__()
        self.heads, self.attn_dim = heads, attn_dim
        self.dropout = nn.Dropout(p=dropout)
        self.key_proj = nn.Linear(hidden_dim, attn_dim * heads)
        self.val_proj = nn.Linear(hidden_dim, attn_dim * heads)
        self.query_proj = nn.Linear(hidden_dim, attn_dim * heads)
        self.output_proj = nn.Linear(attn_dim * heads, hidden_dim)

    def forward(self, queries, keys, values, mask):
        batch, time, _ = queries.shape
        key_heads = self.key_proj(keys).reshape(batch, time, self.heads, self.attn_dim)
        val_heads = self.val_proj(values).reshape(batch, time, self.heads, self.attn_dim)
        # NOTE: Sea-Snell computes queries from the `values` argument; the call sites pass
        # the same tensor for all three, so this is equivalent -- ported verbatim.
        query_heads = self.query_proj(values).reshape(batch, time, self.heads, self.attn_dim)
        attn = F.softmax((torch.einsum("bqhd,bkhd->hbqk", query_heads, key_heads)
                          / math.sqrt(self.attn_dim)).masked_fill(mask, float("-inf")), dim=-1)
        attn = self.dropout(attn.transpose(0, 1).contiguous())
        combined = torch.einsum("bkhd,bhqk->bqhd", val_heads, attn).reshape(
            batch, time, self.attn_dim * self.heads)
        return self.output_proj(combined)


class TransformerBlock(nn.Module):
    def __init__(self, heads, hidden_dim, attn_dim, intermediate_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(heads, hidden_dim, attn_dim, dropout=dropout)
        self.ff1 = nn.Linear(hidden_dim, intermediate_dim)
        self.ff2 = nn.Linear(intermediate_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, attn_mask):        # pre-norm variant (config default)
        x_norm1 = self.layer_norm1(x)
        x = self.dropout1(self.attn(x_norm1, x_norm1, x_norm1, attn_mask)) + x
        x_norm2 = self.layer_norm2(x)
        mlp_out = self.ff2(self.dropout2(F.gelu(self.ff1(x_norm2))))
        x = self.dropout3(mlp_out) + x
        return x


class GrokkTransformer(nn.Module):
    def __init__(self, vocab_size, output_size, max_length=MAX_LEN, heads=4, hidden_dim=128,
                 attn_dim=32, intermediate_dim=512, num_blocks=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.blocks = nn.ModuleList([
            TransformerBlock(heads, hidden_dim, attn_dim, intermediate_dim, dropout=dropout)
            for _ in range(num_blocks)])
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.positions = nn.Embedding(max_length, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, return_block_acts=False):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).expand(x.shape[0], T, T)
        h = self.dropout(self.embeddings(x) * math.sqrt(self.hidden_dim)
                         + self.positions.weight[:T, :])
        acts = []
        for blk in self.blocks:
            h = blk(h, mask)
            acts.append(h)
        logits = self.output(self.norm(h))
        if return_block_acts:
            # acts[-1][:, -1, :] is the representation "after the second transformer
            # block's MLP": the residual stream leaving block 2, before the final
            # LayerNorm, at the answer position.
            return logits, acts
        return logits


class HFGrokkModel(nn.Module):
    """GPT-2 backbone from the huggingface transformers library (per Brown et al.
    Appendix A) with a linear classification head over the 97 output classes, read at
    the last position. block2_out (pre-ln_f) is captured by a forward hook."""

    def __init__(self, vocab_size, output_size, n_positions=MAX_LEN):
        super().__init__()
        from transformers import GPT2Config, GPT2Model
        cfg = GPT2Config(vocab_size=vocab_size, n_positions=n_positions, n_embd=128,
                         n_layer=2, n_head=4, bos_token_id=None, eos_token_id=None,
                         use_cache=False)
        self.backbone = GPT2Model(cfg)
        self.head = nn.Linear(128, output_size)
        self._block2 = None
        self.backbone.h[-1].register_forward_hook(
            lambda mod, inp, out: setattr(
                self, "_block2", out[0] if isinstance(out, tuple) else out))

    def forward(self, x, return_block_acts=False):
        h = self.backbone(x).last_hidden_state          # ln_f applied
        logits = self.head(h)
        if return_block_acts:
            # self._block2 is the residual stream after block 2's MLP, before ln_f
            return logits, [self._block2]
        return logits


def build_model(impl, vocab_size, output_size):
    if impl == "hf":
        return HFGrokkModel(vocab_size, output_size)
    return GrokkTransformer(vocab_size, output_size)


def build_optimizer(optim, model, wd):
    if optim == "adam":     # Brown et al.: "The ADAM optimizer is used" (coupled decay)
        return torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd, betas=BETAS)
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd, betas=BETAS)


# ------------------------------------------------------------------------------- train
def evaluate(model, X, Y, bs=2048):
    model.eval()
    correct, loss_sum = 0, 0.0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            logits = model(X[i:i + bs])[:, -1, :]
            loss_sum += F.cross_entropy(logits, Y[i:i + bs], reduction="sum").item()
            correct += (logits.argmax(-1) == Y[i:i + bs]).sum().item()
    model.train()
    return correct / len(X), loss_sum / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--wd", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", type=str, default="pilot")
    ap.add_argument("--impl", choices=["hf", "port"], default="hf")
    ap.add_argument("--optim", choices=["adam", "adamw"], default="adam")
    args = ap.parse_args()

    torch.set_num_threads(3)
    name = f"brown_div97__frac-{args.frac}__wd-{args.wd}__seed-{args.seed}__{args.tag}"
    out = REPO / "runs_brown" / name
    (out / "ckpt").mkdir(parents=True, exist_ok=True)

    ds = ModDivisionDataset(P, args.frac, seed=args.seed)
    torch.manual_seed(args.seed)
    model = build_model(args.impl, ds.n_vocab, ds.n_out)
    opt = build_optimizer(args.optim, model, args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: min(s / WARMUP, 1))
    batch_rng = random.Random(args.seed + 10_000)

    Xtr, Ytr = ds.tensors(ds.train_pairs)
    Xva, Yva = ds.tensors(ds.val_pairs)
    meta = {"p": P, "frac_train": args.frac, "weight_decay": args.wd, "seed": args.seed,
            "tag": args.tag, "impl": args.impl, "optim": args.optim, "steps": STEPS,
            "bsize": BSIZE, "lr": LR, "betas": BETAS, "warmup": WARMUP,
            "n_train": len(Xtr), "n_val": len(Xva), "ckpt_steps": CKPT_STEPS,
            "eval_every": EVAL_EVERY, "torch": torch.__version__}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    log_path = out / "curves.csv"
    log_path.write_text("step,train_acc,val_acc,train_loss,val_loss\n", encoding="utf-8")
    t0 = time.time()
    model.train()
    for step in range(STEPS + 1):
        if step in CKPT_STEPS:
            torch.save(model.state_dict(), out / "ckpt" / f"step{step:06d}.pt")
        if step % EVAL_EVERY == 0:
            tr_a, tr_l = evaluate(model, Xtr, Ytr)
            va_a, va_l = evaluate(model, Xva, Yva)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{step},{tr_a:.6f},{va_a:.6f},{tr_l:.6f},{va_l:.6f}\n")
            if step % 2000 == 0:
                print(f"[{name}] step {step}  train {tr_a:.3f}  val {va_a:.3f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
        if step == STEPS:
            break
        idx = [batch_rng.choice(ds.train_pairs) for _ in range(BSIZE)]
        xb, yb = ds.tensors(idx)
        loss = F.cross_entropy(model(xb)[:, -1, :], yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    print(f"[{name}] DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
