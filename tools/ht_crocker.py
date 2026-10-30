"""CROCKER of the real weight-space training trajectory, with per-run time-shuffle null."""
import sys, os, warnings; sys.path.insert(0,'.')
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from analysis.topo_runs import TopoRun
from analysis.traj import trajectory_matrix
from analysis.crocker import crocker_trajectory, loop_summary

rows=[]
runs=[d for d in sorted(Path('runs_topo').glob('*')) if (d/'trajectory.npz').exists()]
for d in runs:
    r=TopoRun(d)
    if r.curves.test_acc.max()<0.5: continue
    tg=r.t_grok; tm=r.t_memorize
    if not np.isfinite(tg): continue
    M=trajectory_matrix(r,project_dim=256,seed=0); ep=r.epochs
    for w in (60,120):
        if len(M)<w+10: continue
        mids,grid,B0,B1=crocker_trajectory(M,ep,window=w,stride=6,n_scales=40)
        s=loop_summary(mids,grid,B1); la=np.asarray(s['loop_area'])
        # null: shuffle the time order, keeping the same point cloud
        rng=np.random.default_rng(0); nulls=[]
        for _ in range(3):
            Ms=M[rng.permutation(len(M))]
            _,g2,_,B1s=crocker_trajectory(Ms,ep,window=w,stride=6,n_scales=40)
            nulls.append(loop_summary(mids,g2,B1s)['loop_area'].max())
        rows.append({'run':d.name,'arch':r.meta.get('arch','mlp'),'task':r.meta['task'],'p':r.p,
                     'tf':r.meta['train_fraction'],'seed':r.meta['seed'],'window':w,
                     't_grok':tg,'t_mem':tm,'plateau':tg-tm,
                     'loop_max':float(la.max()),'loop_frac':float((la>0).mean()),
                     'null_loop_max':float(np.mean(nulls)),
                     'peak_epoch':float(mids[int(np.argmax(la))]) if la.max()>0 else np.nan})
    print(f"  {d.name[:58]}",flush=True)
D=pd.DataFrame(rows); D.to_csv('RESULTS/ht_crocker.csv',index=False)
pd.set_option('display.width',240)
print(f"\n=== trajectory CROCKER on {D.run.nunique()} grokking runs ===")
print(D.groupby(['arch','task','window']).agg(n=('run','size'),
  loop_max=('loop_max','mean'),null=('null_loop_max','mean'),
  loop_frac=('loop_frac','mean')).round(3).to_string())
print("\n=== is there loop content above the time-shuffle null? ===")
D['excess']=D.loop_max-D.null_loop_max
print(f"  runs with loop_max > null: {(D.excess>0).sum()}/{len(D)}")
print(f"  median loop_max {D.loop_max.median():.2f}  median null {D.null_loop_max.median():.2f}")
sub=D[(D.loop_max>0)&D.peak_epoch.notna()]
if len(sub):
    sub=sub.assign(phase=(sub.peak_epoch-sub.t_mem)/sub.plateau)
    print(f"\n  where does peak loop content sit relative to the plateau? (0=memorised, 1=grokked)")
    print(f"    median phase {sub.phase.median():+.2f}   IQR {sub.phase.quantile(.25):+.2f}..{sub.phase.quantile(.75):+.2f}  n={len(sub)}")
