"""Circular coordinates on real trained embeddings, with a per-run null."""
import sys, os, warnings; sys.path.insert(0,'.')
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from analysis.topo_runs import TopoRun
from analysis.circcoords import circular_coordinate, winding_fit
from analysis.tda import prepare_cloud, persistence_diagrams, diagram_summaries

def coherence(E, u, n_null=12, seed=0):
    th,info = circular_coordinate(E, coeff=17, eps_frac=0.5)
    if th is None: return None
    w = winding_fit(th,u)
    rng=np.random.default_rng(seed); nulls=[]
    for _ in range(n_null):
        Ep = E[rng.permutation(E.shape[0])]
        t2,_ = circular_coordinate(Ep, coeff=17, eps_frac=0.5)
        if t2 is not None: nulls.append(winding_fit(t2,u)['circ_resid'])
    nulls=np.array(nulls,float)
    out={'winding':w['winding'],'circ_resid':w['circ_resid'],'lifetime':info['lifetime'],
         'null_mean':float(nulls.mean()) if nulls.size else np.nan,
         'null_sd':float(nulls.std()) if nulls.size else np.nan}
    out['z_below_null']=(out['null_mean']-out['circ_resid'])/out['null_sd'] if nulls.size and nulls.std()>0 else np.nan
    return out

rows=[]
for d in sorted(Path('runs_topo').glob('*')):
    if not (d/'trajectory.npz').exists(): continue
    r=TopoRun(d)
    if r.meta.get('task')!='add': continue          # circular coords need a cyclic ground truth
    p=r.p; u=np.arange(p)/p
    E=r.embeddings('a',-1)
    c=coherence(E,u,seed=0)
    if c is None: continue
    dg=persistence_diagrams(prepare_cloud(E,max_points=400,scale='diameter'),maxdim=1)
    s=diagram_summaries(dg[1],'H1')
    rows.append({'arch':r.meta.get('arch','mlp'),'p':p,'tf':r.meta['train_fraction'],'seed':r.meta['seed'],
                 'test':float(r.curves.test_acc.iloc[-1]),'ambient_b1':s['H1_n_above_0.15'],
                 'ambient_H1max':s['H1_max_pers'],**c})
    print(f"  {d.name[:56]}",flush=True)
D=pd.DataFrame(rows); D.to_csv('RESULTS/ht_circcoords.csv',index=False)
pd.set_option('display.width',240)
print(f"\n=== circular coordinates on {len(D)} cyclic-task runs ===")
print(D.groupby(['arch','p','tf']).agg(n=('seed','size'),test=('test','mean'),
  amb_b1=('ambient_b1','mean'),circ_resid=('circ_resid','mean'),
  null=('null_mean','mean'),z=('z_below_null','mean')).round(3).to_string())
print("\n=== does circular coherence detect structure where ambient b1 = 0? ===")
z=D[D.ambient_b1==0]
print(f"  runs with ambient b1 = 0: {len(z)} of {len(D)}")
if len(z): print(f"  of those, z_below_null > 2 (coherent winding despite b1=0): {(z.z_below_null>2).sum()}")
from scipy.stats import spearmanr
ok=D.dropna(subset=['z_below_null'])
if len(ok)>5:
    rho,pv=spearmanr(ok.test,ok.z_below_null); print(f"\n  spearman(test_acc, circular coherence z) = {rho:+.3f} (p={pv:.1e}, n={len(ok)})")
