import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd
from analysis.runlib import load_runs, split_runs, RESULTS
from analysis import predictor as P
from analysis.causal import build_tq_table_truncated, eval_causal
pd.set_option("display.width",240); pd.set_option("display.max_columns",60)
import warnings; warnings.filterwarnings("ignore")

runs,_ = load_runs(); disc,_ = split_runs(runs)
CUTOFFS = [5000,8000,10000,12000,15000,20000,25000,30000,40000,80000]
SIGS = [("layers.1|mst_dim", P.signal_dim("layers.1","mst_dim")),
        ("layers.2|mst_alpha", P.signal_dim("layers.2","mst_alpha")),
        ("layers.0|pca_participation_ratio", P.signal_dim("layers.0","pca_participation_ratio"))]
rows=[]
for name, sig in SIGS:
    for E in CUTOFFS:
        df = build_tq_table_truncated(disc, sig, E)
        if len(df) < 8: continue
        r = eval_causal(df)
        rows.append({"signal":name,"cutoff":E,"n":r.get("n"),"n_forecast":r.get("n_forecast"),
                     "mae":r.get("mae"),"r2_oos":r.get("r2_oos"),"base_mae":r.get("base_mae"),
                     "fc_mae":r.get("fc_mae"),"fc_r2_oos":r.get("fc_r2_oos"),"fc_base_mae":r.get("fc_base_mae")})
t=pd.DataFrame(rows)
t["skill_vs_base"]=1-t["mae"]/t["base_mae"]; t["fc_skill_vs_base"]=1-t["fc_mae"]/t["fc_base_mae"]
print("=== CAUSAL prefix evaluation (DISCOVERY). fc_* = runs whose T_grok is still in the FUTURE at cutoff ===")
print(t.round(3).to_string(index=False))
t.to_csv(RESULTS/"phase1_causal_discovery.csv",index=False)
