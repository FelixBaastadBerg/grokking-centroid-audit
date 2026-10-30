"""PHASE 2: broad search over EVERY logged signal for genuine causal skill. DISCOVERY only."""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings, itertools
warnings.filterwarnings("ignore")
from analysis.runlib import load_runs, split_runs, RESULTS
from analysis import predictor as P
from analysis.causal import build_tq_table_truncated, eval_causal

runs,_ = load_runs(); disc,_ = split_runs(runs)
r0 = next(iter(disc.values()))
combos = sorted({(l,m) for l,m in zip(r0.dims.layer, r0.dims.metric_name)})
combos = [(l,m) for l,m in combos if not m.endswith("_r2")]
print(f"{len(combos)} (layer,metric) signals x {len(disc)} discovery runs", flush=True)

SIGS = [(f"{l}|{m}", P.signal_dim(l,m)) for l,m in combos]
SIGS += [("train|loss", P.signal_metrics("loss","train")),
         ("train|accuracy", P.signal_metrics("accuracy","train")),
         ("train|softmax_collapse", P.signal_metrics("softmax_collapse","train")),
         ("train|zero_terms", P.signal_metrics("zero_terms","train")),
         ("general|compression_lzma_bits", P.signal_metrics("compression_lzma_bits","general")),
         ("general|compression_lzma_bits_q8", P.signal_metrics("compression_lzma_bits_q8","general"))]
SIGS += [(f"weights_l2|{l}", P.signal_weights_l2(l)) for l in ("layers.0.weight","layers.1.weight","layers.2.weight")]

CUTOFFS=[10000,15000,20000,25000,30000]
rows=[]
for i,(name,sig) in enumerate(SIGS):
    # retrospective (full-trajectory) skill, for contrast
    dfull = P.build_tq_table(disc, sig)
    retro = P.eval_nested(dfull)[0] if len(dfull)>=10 else {}
    for E in CUTOFFS:
        df = build_tq_table_truncated(disc, sig, E)
        if len(df) < 10: continue
        r = eval_causal(df)
        rows.append({"signal":name,"cutoff":E,"n":r.get("n"),"n_fc":r.get("n_forecast"),
                     "fc_mae":r.get("fc_mae"),"fc_base_mae":r.get("fc_base_mae"),
                     "fc_r2_oos":r.get("fc_r2_oos"),"mae":r.get("mae"),"r2_oos":r.get("r2_oos"),
                     "retro_r2_oos":retro.get("r2_oos"),"retro_mae":retro.get("mae")})
    if (i+1)%10==0: print(f"  ...{i+1}/{len(SIGS)}", flush=True)

t=pd.DataFrame(rows)
t["fc_skill"]=1-t["fc_mae"]/t["fc_base_mae"]
t.to_csv(RESULTS/"phase2_sweep_discovery.csv",index=False)
agg=(t[t.n_fc>=8].groupby("signal")
       .agg(mean_fc_skill=("fc_skill","mean"), max_fc_skill=("fc_skill","max"),
            mean_fc_r2=("fc_r2_oos","mean"), retro_r2=("retro_r2_oos","first"), n_cut=("cutoff","size"))
       .sort_values("mean_fc_skill",ascending=False))
print("\n=== TOP 20 by mean FORECAST skill vs granule-free baseline (DISCOVERY) ===")
print(agg.head(20).round(3).to_string())
print("\n=== BOTTOM 5 ==="); print(agg.tail(5).round(3).to_string())
agg.to_csv(RESULTS/"phase2_ranking_discovery.csv")
print("\nretrospective vs causal, Spearman:",
      round(float(agg[["mean_fc_skill","retro_r2"]].corr(method="spearman").iloc[0,1]),3))
