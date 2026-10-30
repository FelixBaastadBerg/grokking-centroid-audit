"""Run-level bootstrap CI on the CONFIRMATION causal numbers (reporting, not re-tuning)."""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from analysis.runlib import load_runs, split_runs, RESULTS
from analysis import predictor as P
from analysis.causal import build_tq_table_truncated
from analysis.predictor import _pick_q, Q_GRID
runs,_=load_runs(); _, conf = split_runs(runs)
rng=np.random.default_rng(0)
print("Pooled over cutoffs; bootstrap resamples RUNS (not epochs).\n")
for name in ["layers.2|corr_dim","layers.2|mst_alpha","layers.1|mst_dim"]:
    l,m=name.split("|"); sig=P.signal_dim(l,m)
    ae_pred=[]; ae_base=[]
    for E in [10000,15000,20000,25000]:
        df=build_tq_table_truncated(conf,sig,E)
        if len(df)<8: continue
        y=df["target"].to_numpy(float); n=len(y); fc=df["is_forecast"].to_numpy(bool)
        pred=np.full(n,np.nan); bs=np.full(n,np.nan)
        for i in range(n):
            tr=np.arange(n)!=i; q=_pick_q(df.iloc[tr],Q_GRID)
            if np.isfinite(q):
                x=df[f"tq_{q:.2f}"].to_numpy(float)
                if np.isfinite(x[i]):
                    d=(x[tr]-y[tr]); d=d[np.isfinite(d)]
                    if len(d): pred[i]=x[i]-np.mean(d)
            s=y[tr][y[tr]>E]; bs[i]=np.mean(s) if len(s) else np.mean(y[tr])
        msk=fc&np.isfinite(pred)
        ae_pred += list(np.abs(pred[msk]-y[msk])); ae_base += list(np.abs(bs[msk]-y[msk]))
    ae_pred=np.array(ae_pred); ae_base=np.array(ae_base)
    skill=1-ae_pred.mean()/ae_base.mean()
    bo=[]
    for _ in range(5000):
        idx=rng.integers(0,len(ae_pred),len(ae_pred))
        bo.append(1-ae_pred[idx].mean()/ae_base[idx].mean())
    lo,hi=np.percentile(bo,[2.5,97.5])
    print(f"{name:<24} n={len(ae_pred):>3}  skill vs survival baseline = {skill:+.3f}  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]   {'DISTINGUISHABLE' if lo>0 else 'NOT distinguishable from 0'}")
