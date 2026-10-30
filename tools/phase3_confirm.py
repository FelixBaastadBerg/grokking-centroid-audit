"""
CONFIRMATION -- run EXACTLY ONCE on the locked 20-run confirmation set.
Candidates were chosen on DISCOVERY only. These are the numbers we report.
"""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from analysis.runlib import load_runs, split_runs, RESULTS
from analysis import predictor as P
from analysis.causal import build_tq_table_truncated
from analysis.predictor import _pick_q, Q_GRID

runs,_=load_runs(); disc, conf = split_runs(runs)
print(f"CONFIRMATION SET: {len(conf)} runs, granules={json.load(open(RESULTS/'split.json'))['confirmation_granules']}\n")

# ---------- H1 (retrospective): does the paper-style predictor reproduce? ----------
print("="*100)
print("H1  RETROSPECTIVE skill (full trajectory, nested CV) -- the paper's protocol")
print("="*100)
rows=[]
for name in ["layers.1|mst_dim","layers.2|mst_alpha","layers.2|corr_dim"]:
    l,m=name.split("|")
    df=P.build_tq_table(conf,P.signal_dim(l,m))
    if len(df)<8: continue
    r,_,_=P.eval_nested(df); r["signal"]=name; rows.append(r)
for fn,lab in [(P.eval_global_mean,"BASELINE global mean"),(P.eval_hparam_only,"BASELINE hyperparams only")]:
    df=P.build_tq_table(conf,P.signal_dim("layers.1","mst_dim"))
    r,_,_=fn(df); r["signal"]=lab; rows.append(r)
t1=pd.DataFrame(rows)[["signal","n","mae","rmse","r2_fit","r2_oos"]]
print(t1.round(3).to_string(index=False))

# ---------- H2 (causal): the claim that actually matters ----------
print("\n"+"="*100)
print("H2  CAUSAL forecast skill vs SURVIVAL-AWARE baseline -- runs not yet grokked at cutoff E")
print("="*100)
rows=[]
for name in ["layers.2|corr_dim","layers.1|mst_dim","layers.2|mst_alpha"]:
    l,m=name.split("|"); sig=P.signal_dim(l,m)
    for E in [10000,15000,20000,25000]:
        df=build_tq_table_truncated(conf,sig,E)
        if len(df)<8: continue
        y=df["target"].to_numpy(float); n=len(y); fc=df["is_forecast"].to_numpy(bool)
        if fc.sum()<4: continue
        pred=np.full(n,np.nan); bsurv=np.full(n,np.nan); bnaive=np.full(n,np.nan)
        for i in range(n):
            tr=np.arange(n)!=i
            q=_pick_q(df.iloc[tr],Q_GRID)
            if np.isfinite(q):
                x=df[f"tq_{q:.2f}"].to_numpy(float)
                if np.isfinite(x[i]):
                    d=(x[tr]-y[tr]); d=d[np.isfinite(d)]
                    if len(d): pred[i]=x[i]-np.mean(d)
            bnaive[i]=np.mean(y[tr]); s=y[tr][y[tr]>E]
            bsurv[i]=np.mean(s) if len(s) else np.mean(y[tr])
        msk=fc&np.isfinite(pred)
        if msk.sum()<4: continue
        mae=np.mean(np.abs(pred[msk]-y[msk]))
        rows.append({"signal":name,"cutoff":E,"n_fc":int(msk.sum()),"mae":round(mae,1),
                     "base_survival":round(np.mean(np.abs(bsurv[msk]-y[msk])),1),
                     "base_naive":round(np.mean(np.abs(bnaive[msk]-y[msk])),1),
                     "skill_vs_survival":round(1-mae/np.mean(np.abs(bsurv[msk]-y[msk])),3),
                     "skill_vs_naive":round(1-mae/np.mean(np.abs(bnaive[msk]-y[msk])),3)})
t2=pd.DataFrame(rows)
print(t2.to_string(index=False))
print("\nper-signal mean skill vs survival-aware baseline:")
print(t2.groupby("signal")["skill_vs_survival"].mean().round(3).to_string())
t1.to_csv(RESULTS/"CONFIRMATION_retrospective.csv",index=False)
t2.to_csv(RESULTS/"CONFIRMATION_causal.csv",index=False)
print("\n>>> CONFIRMATION COMPLETE. These numbers are final and are not to be re-tuned. <<<")
