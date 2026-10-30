import sys; sys.path.insert(0,".")
import torch
from models import Transformer
torch.manual_seed(0)
a=torch.tensor([[3,7]]); b=torch.tensor([[7,3]])
for use_pos in (True, False):
    t=Transformer(d_model=128,num_heads=4,num_layers=1,vocab_size=113,seq_len=2,use_positional=use_pos).eval()
    with torch.no_grad():
        ra, rb = t(a)[:,-1], t(b)[:,-1]          # last-token readout, as the training loop uses
        full_a, full_b = t(a), t(b)
        # is the model permutation-EQUIVARIANT (i.e. truly position-blind)?
        equivar = (full_a - full_b.flip(1)).abs().max().item()
        readout_diff = (ra-rb).abs().max().item()
    print(f"use_positional={str(use_pos):<5}  last-token readout |f(a,b)-f(b,a)| = {readout_diff:.4f}   "
          f"permutation-equivariance residual = {equivar:.2e}")
print("""
Interpretation: without positional embedding the model is permutation-EQUIVARIANT
(residual ~0): swapping the inputs swaps the outputs. But the training loop reads
out at the LAST token only, and out[-1] is a function of (x_last, multiset{x}).
Given the bag {a,b} and knowing which one is last, the ordered pair is recoverable
-- so sub/div are NOT unlearnable after all. Positional embedding is still the
right thing to add (it makes the ordering explicit rather than implicit), but my
earlier 'unlearnable' claim was wrong.""")
