"""Smoke-test the code fixes T1,T3,T6,T7 without training anything."""
import sys; sys.path.insert(0,".")
import torch, numpy as np
from models import MLP, Transformer
from binary_operations import divide_mod
from logger import resolve_probe_layers, probe_display_name, MetricsLogger

# T3: Fermat modular inverse
p=113; ok=all((divide_mod(x,y,p)*y)%p == x%p for x in range(1,p) for y in range(1,p))
print("T3 modular inverse correct for all x,y in [1,p):", ok)

# T1: positional encoding makes the transformer order-sensitive
t = Transformer(d_model=128,num_heads=4,num_layers=1,vocab_size=113,seq_len=2)
a=torch.tensor([[3,7]]); b=torch.tensor([[7,3]])
t.eval()
with torch.no_grad(): d=(t(a)-t(b)).abs().max().item()
t_nopos = Transformer(d_model=128,num_heads=4,num_layers=1,vocab_size=113,seq_len=2,use_positional=False)
t_nopos.eval()
with torch.no_grad(): d0=(t_nopos(a)-t_nopos(b)).abs().max().item()
print(f"T1 order-sensitivity  with pos-emb: {d:.4f}   without (old behaviour): {d0:.2e}")

# T6/T7: probe resolution + weight rows for BOTH architectures
mlp = MLP(input_size=226,hidden_sizes=[200,200],output_size=113,bias=False)
for name,m in (("mlp",mlp),("transformer",t)):
    probes, pre = resolve_probe_layers(m)
    disp=[probe_display_name(x) for x in probes]
    nW=sum(1 for n,q in m.named_parameters() if n.endswith(".weight") and q.ndim==2)
    print(f"T6/T7 {name:<12} probes={disp} pre_logits<-{pre}  2D weight matrices={nW}")
