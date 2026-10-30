"""
SW1PerS (Perea-Harer): sliding-window / Takens embedding of a SCALAR training signal,
then max H1 persistence -> a periodicity score. A periodic signal's delay embedding is
a loop; noise and monotone drift are not.

Score is normalised so a perfect circle -> 1 and nothing -> 0.
"""
import sys, os, warnings; sys.path.insert(0,'.')
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

def sw_embed(x, d=6, tau=3):
    N=len(x)-(d-1)*tau
    if N<12: return None
    return np.stack([x[i:i+N] for i in range(0,(d-1)*tau+1,tau)],axis=1)

def periodicity(x, d=6, tau=3, max_pts=400, seed=0):
    Z=sw_embed(np.asarray(x,float),d,tau)
    if Z is None: return np.nan
    Z=Z-Z.mean(0,keepdims=True)
    n=np.linalg.norm(Z,axis=1,keepdims=True); n[n==0]=1
    Z=Z/n                                        # project to sphere: SW1PerS convention
    if len(Z)>max_pts:
        Z=Z[np.random.default_rng(seed).choice(len(Z),max_pts,replace=False)]
    dg=ripser(Z,maxdim=1)['dgms'][1]
    if len(dg)==0: return 0.0
    life=dg[:,1]-dg[:,0]
    return float(np.max(life)/np.sqrt(3.0))      # sqrt(3) = max possible for this construction

if __name__=='__main__':
    print('=== VALIDATION: SW1PerS on signals of known periodicity ===')
    t=np.linspace(0,1,600); rng=np.random.default_rng(0)
    cases={'pure sine (10 cycles)':np.sin(2*np.pi*10*t),
           'sine + noise':np.sin(2*np.pi*10*t)+0.3*rng.normal(size=600),
           'quasi-periodic (2 freqs)':np.sin(2*np.pi*10*t)+np.sin(2*np.pi*10*np.sqrt(2)*t),
           'damped oscillation':np.exp(-3*t)*np.sin(2*np.pi*10*t),
           'monotone ramp':t,
           'white noise':rng.normal(size=600),
           'sigmoid step':1/(1+np.exp(-30*(t-0.5)))}
    for k,v in cases.items():
        print(f'  {k:<26} periodicity = {periodicity(v):.3f}')
    print()
    print('=== REAL RUNS: periodicity of training signals inside the plateau ===')
    from analysis.topo_runs import TopoRun
    rows=[]
    for d in sorted(Path('runs_topo').glob('*')):
        if not (d/'curves.csv').exists(): continue
        r=TopoRun(d); c=r.curves.sort_values('epoch')
        if c.test_acc.max()<0.5: continue
        hit=c[c.train_acc>=0.99]
        if not len(hit): continue
        tm=float(hit.epoch.iloc[0]); tg=r.t_grok
        if not np.isfinite(tg) or tg-tm<5000: continue
        win=c[(c.epoch>=tm)&(c.epoch<=tg)]           # the plateau only
        if len(win)<40: continue
        for col in ('train_loss','test_loss','w0_norm'):
            if col not in win: continue
            y=win[col].to_numpy(float)
            yd=np.diff(y)                            # detrend: oscillation, not drift
            rows.append({'run':d.name,'arch':r.meta.get('arch','mlp'),'signal':col,
                         'n':len(yd),'periodicity':periodicity(yd)})
    D=pd.DataFrame(rows)
    if len(D):
        D.to_csv('RESULTS/sw1pers.csv',index=False)
        pd.set_option('display.width',200)
        print(D.groupby('signal').agg(n=('run','size'),mean=('periodicity','mean'),
              sd=('periodicity','std'),mx=('periodicity','max')).round(3).to_string())
        print(f"\n  reference: white noise {periodicity(rng.normal(size=600)):.3f}, pure sine {periodicity(np.sin(2*np.pi*10*t)):.3f}")
