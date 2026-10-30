import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from analysis.runlib import load_runs, split_runs
runs,_=load_runs(); disc,_=split_runs(runs)
def npts(layer,metric,lag,pre):
    ns=[]
    for r in disc.values():
        gap=r.gen_gap; dim=r.dim_series(layer,metric)
        d=dim.copy(); d.index=d.index-lag
        idx=gap.index.intersection(d.index)
        if pre: idx=idx[idx<r.t_grok]
        ns.append(len(idx))
    return np.median(ns)
print("median # paired points behind each reported correlation:\n")
print(f"{'signal':<42}{'lag':>8}{'window':>11}{'pts':>6}")
for layer,metric in [("layers.1","mst_dim"),("layers.2","mst_alpha"),("layers.0","pca_participation_ratio")]:
    for lag,pre in [(0,False),(0,True),(-1000,True),(-5000,True),(-11000,True)]:
        print(f"{layer+'|'+metric:<42}{lag:>8}{('pre-onset' if pre else 'full'):>11}{npts(layer,metric,lag,pre):>6.0f}")
