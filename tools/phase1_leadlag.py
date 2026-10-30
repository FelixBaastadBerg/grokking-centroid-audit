"""PHASE 1: reproduce C1 (lead-lag Pearson ~0.985) and test it pre-onset only."""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from analysis.runlib import load_runs, split_runs, RESULTS
pd.set_option("display.width",200)

runs,_ = load_runs(); disc,_ = split_runs(runs)

def per_run_corr(r, layer, metric, lag, pre_onset=False, kind="pearson"):
    gap = r.gen_gap; dim = r.dim_series(layer, metric)
    if len(gap)<5 or len(dim)<5: return np.nan, 0
    # correlate gen_gap(t) with dim(t+lag)  (paper's convention)
    d_sh = dim.copy(); d_sh.index = d_sh.index - lag
    idx = gap.index.intersection(d_sh.index)
    if pre_onset:
        idx = idx[idx < r.t_grok]
    if len(idx) < 6: return np.nan, len(idx)
    x = gap.loc[idx].to_numpy(float); y = d_sh.loc[idx].to_numpy(float)
    ok = np.isfinite(x)&np.isfinite(y); x,y = x[ok],y[ok]
    if len(x)<6 or np.std(x)<1e-12 or np.std(y)<1e-12: return np.nan, len(x)
    s = pd.Series(x).corr(pd.Series(y), method=kind)
    return float(s), len(x)

CANDS=[("layers.1","mst_dim"),("layers.2","mst_dim"),("layers.2","mst_alpha"),
       ("layers.1","corr_dim"),("layers.0","pca_participation_ratio"),
       ("all_layers_concat","E_alpha_1_row_l2")]
rows=[]
for layer,metric in CANDS:
    for lag in (0,1000,5000,10000):
        for pre in (False,True):
            vals=[]; ns=[]
            for r in disc.values():
                c,n = per_run_corr(r,layer,metric,lag,pre_onset=pre)
                if np.isfinite(c): vals.append(c); ns.append(n)
            if not vals: continue
            v=np.array(vals)
            # run-level bootstrap CI (resample RUNS, not epochs -- epochs are autocorrelated)
            rng=np.random.default_rng(0)
            bs=[np.mean(rng.choice(v,size=len(v),replace=True)) for _ in range(2000)]
            rows.append({"layer":layer,"metric":metric,"lag":lag,
                         "window":"pre-onset" if pre else "full",
                         "n_runs":len(v),"median_pts":int(np.median(ns)),
                         "mean_r":float(v.mean()),
                         "ci_lo":float(np.percentile(bs,2.5)),"ci_hi":float(np.percentile(bs,97.5))})
t=pd.DataFrame(rows)
print("=== C1 reproduction: per-run Pearson( gen_gap(t), dim(t+lag) ), DISCOVERY ===")
print("    run-level bootstrap CIs (epochs are autocorrelated; iid CIs are invalid)\n")
print(t.round(3).to_string(index=False))
t.to_csv(RESULTS/"phase1_leadlag_discovery.csv",index=False)
piv=t.pivot_table(index=["layer","metric","lag"],columns="window",values="mean_r").round(3)
piv["drop"]=(piv["full"]-piv["pre-onset"]).round(3)
print("\n=== full-window vs pre-onset-only mean r ===")
print(piv.to_string())
