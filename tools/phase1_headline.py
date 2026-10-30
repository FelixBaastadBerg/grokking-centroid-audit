"""PHASE 1, checks 1 & 2: nested-CV collapse of the R^2=0.98 claim, on DISCOVERY."""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd
from analysis.runlib import load_runs, split_runs, RESULTS
from analysis import predictor as P

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
runs,_ = load_runs(); disc, conf = split_runs(runs)
print(f"discovery={len(disc)}  confirmation={len(conf)}  (confirmation stays LOCKED)\n")

CANDS = [("layers.1","mst_dim"), ("layers.2","mst_dim"), ("layers.2","mst_alpha"),
         ("layers.1","mst_alpha"), ("all_layers_concat","mst_dim")]
out=[]
for layer, metric in CANDS:
    df = P.build_tq_table(disc, P.signal_dim(layer, metric))
    if len(df) < 10:
        print(f"{layer}|{metric}: only {len(df)} runs fit -- skipped"); continue
    res = P.eval_all(df); res.insert(0,"signal",f"{layer}|{metric}"); res.insert(1,"n_fit",len(df))
    out.append(res)
    print(f"=== {layer} | {metric}   ({len(df)}/{len(disc)} runs fit) ===")
    print(res[["method","n","mae","rmse","r2_fit","r2_oos","norm_mae"]].to_string(index=False)); print()
allres = pd.concat(out, ignore_index=True)
allres.to_csv(RESULTS/"phase1_headline_discovery.csv", index=False)
print("wrote results/phase1_headline_discovery.csv")
