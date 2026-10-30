import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from analysis.runlib import load_runs, split_runs, RESULTS
runs,_=load_runs(); disc,_=split_runs(runs)
LAGS=list(range(-15000,15001,1000))
def scan(layer,metric,pre=False):
    out=[]
    for lag in LAGS:
        vals=[]
        for r in disc.values():
            gap=r.gen_gap; dim=r.dim_series(layer,metric)
            if len(gap)<5 or len(dim)<5: continue
            d=dim.copy(); d.index=d.index-lag
            idx=gap.index.intersection(d.index)
            if pre: idx=idx[idx<r.t_grok]
            if len(idx)<6: continue
            x=gap.loc[idx].to_numpy(float); y=d.loc[idx].to_numpy(float)
            ok=np.isfinite(x)&np.isfinite(y); x,y=x[ok],y[ok]
            if len(x)<6 or np.std(x)<1e-12 or np.std(y)<1e-12: continue
            vals.append(np.corrcoef(x,y)[0,1])
        if vals: out.append({"lag":lag,"mean_r":float(np.mean(vals)),"n":len(vals)})
    return pd.DataFrame(out)
print("CONVENTION: lag>0 means dim is sampled LATER than gen_gap (Table-1 caption).")
print("            For the dimension to LEAD (be predictive), the peak must be at lag<0.\n")
for layer,metric in [("layers.1","mst_dim"),("layers.2","mst_alpha"),("layers.0","pca_participation_ratio")]:
    for pre in (False,True):
        d=scan(layer,metric,pre)
        if d.empty: continue
        i=d.mean_r.abs().idxmax()
        print(f"{layer:<18}{metric:<24}{'pre-onset' if pre else 'full     '}  "
              f"argmax lag={d.loc[i,'lag']:>7}  r={d.loc[i,'mean_r']:.3f}   "
              f"| r(lag=0)={d.loc[d.lag==0,'mean_r'].iloc[0]:.3f}  r(lag=+1000)={d.loc[d.lag==1000,'mean_r'].iloc[0]:.3f}  r(lag=-1000)={d.loc[d.lag==-1000,'mean_r'].iloc[0]:.3f}")
        d.to_csv(RESULTS/f"lagscan_{layer.replace('.','_')}_{metric}_{'pre' if pre else 'full'}.csv",index=False)
