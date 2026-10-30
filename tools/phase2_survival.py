"""
The naive LOO-mean baseline ignores CENSORING: if a run has not grokked by epoch E,
you already know T_grok > E for free. A fair baseline must use that. Here we compare
the top causal candidates against E[T_grok | T_grok > E] estimated from training runs.
DISCOVERY only.
"""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from analysis.runlib import load_runs, split_runs, RESULTS
from analysis import predictor as P
from analysis.causal import build_tq_table_truncated
from analysis.predictor import _pick_q, Q_GRID

runs,_=load_runs(); disc,_=split_runs(runs)
CANDS=["layers.2|corr_dim","layers.2|pca_participation_ratio","layers.0|corr_dim",
       "layers.2|mst_dim","layers.1|mst_dim","layers.2|mst_alpha","layers.0|pca_participation_ratio"]
CUTOFFS=[10000,15000,20000,25000,30000]
rows=[]
for name in CANDS:
    layer,metric=name.split("|"); sig=P.signal_dim(layer,metric)
    for E in CUTOFFS:
        df=build_tq_table_truncated(disc,sig,E)
        if len(df)<10: continue
        y=df["target"].to_numpy(float); n=len(y); fc=df["is_forecast"].to_numpy(bool)
        if fc.sum()<5: continue
        pred=np.full(n,np.nan); base_naive=np.full(n,np.nan); base_surv=np.full(n,np.nan)
        for i in range(n):
            tr=np.arange(n)!=i
            q=_pick_q(df.iloc[tr],Q_GRID)
            x=df[f"tq_{q:.2f}"].to_numpy(float) if np.isfinite(q) else None
            if x is not None and np.isfinite(x[i]):
                d=(x[tr]-y[tr]); d=d[np.isfinite(d)]
                if len(d): pred[i]=x[i]-np.mean(d)
            base_naive[i]=np.mean(y[tr])
            surv=y[tr][y[tr]>E]                       # training runs that also survived to E
            base_surv[i]=np.mean(surv) if len(surv) else np.mean(y[tr])
        m=fc & np.isfinite(pred)
        if m.sum()<5: continue
        mae=np.mean(np.abs(pred[m]-y[m])); bn=np.mean(np.abs(base_naive[m]-y[m])); bs=np.mean(np.abs(base_surv[m]-y[m]))
        rows.append({"signal":name,"cutoff":E,"n_fc":int(m.sum()),"mae":mae,
                     "base_naive":bn,"base_survival":bs,
                     "skill_vs_naive":1-mae/bn,"skill_vs_survival":1-mae/bs})
t=pd.DataFrame(rows)
t.to_csv(RESULTS/"phase2_survival_discovery.csv",index=False)
pd.set_option("display.width",200)
print("=== forecast skill against a NAIVE vs a SURVIVAL-AWARE baseline (DISCOVERY) ===")
print(t.round(3).to_string(index=False))
print("\n=== per-signal means ===")
g=t.groupby("signal")[["skill_vs_naive","skill_vs_survival"]].mean().sort_values("skill_vs_survival",ascending=False)
print(g.round(3).to_string())
