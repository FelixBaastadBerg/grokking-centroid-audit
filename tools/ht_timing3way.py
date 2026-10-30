"""
THREE-WAY TIMING: our symmetry order parameter vs Monod's H1 statistic vs the
trajectory, measured on IDENTICAL runs. Nobody has done this comparison -- the two
published timings (lead vs lag) come from different papers on different runs.

Monod et al. (2605.06352) compute Rips H1 max/total persistence on the token-embedding
rows. We compute exactly that per checkpoint (amb_H1_max), so their statistic and ours
sit side by side on the same networks.

Timing measure = PHASE: where in the memorisation plateau the signal reaches 50% of
its own rise. 0 = the instant memorisation completes, 1 = when accuracy moves.
Boundary markers sit near 0 (vacuous); a signal that marks a real interior event sits
strictly inside.
"""
import sys, os, warnings; sys.path.insert(0,'.')
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from analysis.topo_runs import TopoRun

def rise_epoch(e, y, frac=0.5, tail=5):
    y=np.asarray(y,float); ok=np.isfinite(y)
    if ok.sum()<8: return np.nan
    e,y=np.asarray(e,float)[ok],y[ok]
    base=np.median(y[:tail]); top=np.median(y[-tail:]); d=top-base
    if not np.isfinite(d) or abs(d)<1e-12: return np.nan
    thr=base+frac*d
    hit=np.where(y>=thr)[0] if d>0 else np.where(y<=thr)[0]
    return float(e[hit[0]]) if hit.size else np.nan

E=pd.read_csv('RESULTS/temporal_equiv_series.csv')
rows=[]
for d in sorted(Path('results_runs').glob('*')):
    pc=d/'per_checkpoint.csv'
    rd=Path('runs_topo')/d.name
    if not pc.exists() or not (rd/'trajectory.npz').exists(): continue
    r=TopoRun(rd)
    c=r.curves.sort_values('epoch')
    if c.test_acc.max()<0.5: continue
    hit=c[c.train_acc>=0.99]
    tm=float(hit.epoch.iloc[0]) if len(hit) else np.nan
    tt=rise_epoch(c.epoch,c.test_acc)
    if not np.isfinite(tm) or not np.isfinite(tt) or tt<=tm: continue
    P=pd.read_csv(pc).sort_values('epoch')
    sub=E[E.run==d.name].sort_values('epoch')
    sig={}
    if 'amb_H1_max' in P: sig['Monod H1 (rep.)']=(P.epoch.values,P.amb_H1_max.values)
    if len(sub): sig['equiv_R2 (rep.)']=(sub.epoch.values,sub.equiv_R2.values)
    sig['test accuracy']=(c.epoch.values,c.test_acc.values)
    sig['train loss']=(c.epoch.values,c.train_loss.values)
    plate=tt-tm
    for nm,(e,y) in sig.items():
        t50=rise_epoch(e,y)
        rows.append({'run':d.name,'arch':r.meta.get('arch','mlp'),'task':r.meta['task'],'p':r.p,
                     'tf':r.meta['train_fraction'],'seed':r.meta['seed'],'signal':nm,
                     't_mem':tm,'t_test50':tt,'plateau':plate,'t50':t50,
                     'phase':(t50-tm)/plate if np.isfinite(t50) else np.nan,
                     'lead':tt-t50 if np.isfinite(t50) else np.nan})
D=pd.DataFrame(rows); D.to_csv('RESULTS/timing_3way.csv',index=False)
pd.set_option('display.width',240)
long=D[D.plateau>=5000]
print(f"=== THREE-WAY TIMING on identical runs (plateau >= 5000 epochs, n={long.run.nunique()}) ===\n")
g=long.groupby('signal').agg(n=('run','size'),phase_med=('phase','median'),
   q1=('phase',lambda s:s.quantile(.25)),q3=('phase',lambda s:s.quantile(.75)),
   lead_med=('lead','median'),lead_pos=('lead',lambda s:(s>0).mean())).round(3)
order=['train loss','equiv_R2 (rep.)','Monod H1 (rep.)','test accuracy']
print(g.loc[[o for o in order if o in g.index]].to_string())
print("\n=== by architecture (Monod report LAG on transformers, ~0 on MLPs) ===")
for arch,sa in long.groupby('arch'):
    print(f"  {arch}:")
    ga=sa.groupby('signal')['phase'].agg(['size','median']).round(3)
    for o in order:
        if o in ga.index: print(f"     {o:<20} n={ga.loc[o,'size']:>3.0f}  phase {ga.loc[o,'median']:+.3f}")
