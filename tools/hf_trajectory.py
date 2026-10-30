"""H-F: PH-dimension of the training trajectory in weight space, as a function of training time."""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings, time; warnings.filterwarnings("ignore")
from analysis.topo_runs import load_runs
from analysis.traj import trajectory_matrix, ph_dim, sliding_ph_dim
from pathlib import Path
OUT=Path("RESULTS"); OUT.mkdir(exist_ok=True)
runs=load_runs(); print(f"{len(runs)} runs",flush=True)
glob_rows=[]; slide=[]
t0=time.time()
for r in runs:
    M=trajectory_matrix(r, project_dim=256, seed=0)     # JL projection: distances preserved
    eps=r.epochs
    d_all,r2_all = ph_dim(M, alpha=1.0, n_repeats=6, seed=0)
    tg, tm = r.t_grok, r.t_memorize
    # split trajectory into memorization phase vs post-onset phase
    pre = M[(eps>=tm)&(eps<tg)] if np.isfinite(tg) and np.isfinite(tm) and (eps<tg).sum()>20 else None
    post= M[eps>=tg] if np.isfinite(tg) else None
    d_pre = ph_dim(pre,n_repeats=6,seed=0)[0] if pre is not None and len(pre)>=32 else np.nan
    d_post= ph_dim(post,n_repeats=6,seed=0)[0] if post is not None and len(post)>=32 else np.nan
    glob_rows.append({"run_id":r.path.name,"task":r.meta["task"],"p":r.p,"seed":r.meta["seed"],
        "train_fraction":r.meta["train_fraction"],"groks":r.groks,"t_grok":tg,"t_memorize":tm,
        "final_test":float(r.curves.test_acc.iloc[-1]),
        "phdim_all":d_all,"phdim_r2":r2_all,"phdim_pre":d_pre,"phdim_post":d_post,
        "gen_gap":float(r.curves.train_acc.iloc[-1]-r.curves.test_acc.iloc[-1])})
    s=sliding_ph_dim(M,eps,window=40,step=5,seed=0)
    for k in ("run_id","task","seed","train_fraction"): s[k]=r.path.name if k=="run_id" else r.meta.get(k)
    s["t_grok"]=tg; s["groks"]=r.groks; slide.append(s)
    print(f"  {r.path.name[:56]:<58} phdim={d_all:.2f} (r2={r2_all:.3f}) pre={d_pre:.2f} post={d_post:.2f} ({time.time()-t0:.0f}s)",flush=True)
G=pd.DataFrame(glob_rows); G.to_csv(OUT/"hf_trajectory_global.csv",index=False)
S=pd.concat(slide,ignore_index=True); S.to_csv(OUT/"hf_trajectory_sliding.csv",index=False)
pd.set_option("display.width",240)
print("\n=== H-F: trajectory PH-dimension per run ===")
print(G[["task","p","seed","train_fraction","groks","final_test","t_memorize","t_grok",
         "phdim_all","phdim_r2","phdim_pre","phdim_post"]].round(3).to_string(index=False))
print("\n=== by task (grokking runs) ===")
print(G[G.groks].groupby("task")[["phdim_all","phdim_pre","phdim_post","gen_gap"]].mean().round(3).to_string())
print("\n=== sliding PH-dim: mean before vs after onset ===")
S["phase"]=np.where(S.mid_epoch<S.t_grok,"pre-onset","post-onset")
print(S.groupby(["task","phase"])[["ph_dim","ph_dim_r2","step_norm"]].mean().round(3).to_string())
