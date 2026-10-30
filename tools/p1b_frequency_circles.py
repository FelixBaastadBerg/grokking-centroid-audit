"""
PRIORITY 1b -- the refinement the probe sweep points to.

Nanda's finding is NOT "the embedding is a circle". It is that the embedding is a
SUPERPOSITION of a few Fourier frequencies. If emb(n) = sum_k [a_k cos(2*pi*k*n/p),
b_k sin(2*pi*k*n/p)] over ~5 key frequencies, the p points lie on a curve winding
around a 5-torus. Consecutive points are then far apart in the high-frequency
directions, so the AMBIENT point cloud has no short-scale 1-cycle -- exactly the
b1 = 0 we measured -- even though the representation is perfectly structured.

So the right test is per-frequency: project the embedding onto the 2-D subspace
spanned by Re F[k] and Im F[k] and ask whether THAT is a circle. Validated the same
way as everything else: a synthetic multi-frequency embedding is pushed through the
identical pipeline as a positive control.
"""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from analysis.tda import prepare_cloud, persistence_diagrams, diagram_summaries
from analysis.topo_runs import load_runs

def freq_decompose(E):
    """E:(p,h) -> per-frequency power and 2-D projections."""
    p=E.shape[0]; Ec=E-E.mean(0,keepdims=True)
    F=np.fft.rfft(Ec,axis=0)                    # (n_freq, h)
    power=(np.abs(F)**2).sum(1); power[0]=0.0
    out={}
    for k in range(1,len(power)):
        u=np.real(F[k]); v=np.imag(F[k])
        nu,nv=np.linalg.norm(u),np.linalg.norm(v)
        if nu<1e-12 or nv<1e-12: continue
        P=np.c_[Ec@(u/nu), Ec@(v/nv)]           # (p,2) projection onto this frequency's plane
        out[k]=(power[k]/power.sum(), P)
    return power/power.sum(), out

def circ_feats(P,seed=0):
    Xp=prepare_cloud(P,max_points=400,seed=seed,scale="diameter")
    if Xp.shape[0]<8: return {}
    d=persistence_diagrams(Xp,maxdim=1); s=diagram_summaries(d[1],"H1")
    r=np.linalg.norm(P-P.mean(0),axis=1)
    return {"H1_max":s["H1_max_pers"],"b1":s["H1_n_above_0.15"],
            "radius_cv":float(r.std()/r.mean()) if r.mean()>0 else np.nan}

# ---- positive control: synthetic k-frequency embedding ----
print("=== POSITIVE CONTROL: synthetic embeddings built from k Fourier modes ===")
rng=np.random.default_rng(0)
for p,nk in [(59,1),(59,3),(59,5),(23,1),(23,3),(23,5)]:
    n=np.arange(p); E=np.zeros((p,200))
    ks=rng.choice(np.arange(1,p//2),size=nk,replace=False)
    for k in ks:
        A=rng.normal(size=(2,200))
        E+=np.c_[np.cos(2*np.pi*k*n/p),np.sin(2*np.pi*k*n/p)]@A
    amb=circ_feats(E)
    _,fr=freq_decompose(E)
    best=max(fr.items(),key=lambda kv:kv[1][0])
    pf=circ_feats(best[1][1])
    print(f"  p={p} n_freq={nk}: AMBIENT H1_max={amb['H1_max']:.3f} b1={amb['b1']:.0f} | "
          f"TOP-FREQ-PLANE H1_max={pf['H1_max']:.3f} b1={pf['b1']:.0f} radius_cv={pf['radius_cv']:.3f}")

# ---- real runs ----
rows=[]
for r in load_runs():
    if r.meta["task"]=="perm": continue          # Fourier basis is only defined for cyclic groups
    E=r.embeddings("a",-1)
    q,fr=freq_decompose(E)
    order=sorted(fr.items(),key=lambda kv:-kv[1][0])[:5]
    amb=circ_feats(E)
    for rank,(k,(pw,P)) in enumerate(order):
        f=circ_feats(P)
        rows.append({"run":r.path.name,"task":r.meta["task"],"p":r.p,"tf":r.meta["train_fraction"],
                     "final_test":float(r.curves.test_acc.iloc[-1]),"freq_rank":rank,"freq":k,
                     "power_frac":pw,"amb_H1":amb["H1_max"],**f})
D=pd.DataFrame(rows); D.to_csv("RESULTS/p1b_frequency_circles.csv",index=False)
pd.set_option("display.width",250)
print("\n=== REAL RUNS: is each dominant frequency's 2-D projection a circle? ===")
print(D.round(3).to_string(index=False))
print("\n=== per-run summary: top-5 frequency power, and circle quality of the top plane ===")
g=D.groupby(["task","p","tf","final_test"]).agg(
    top5_power=("power_frac","sum"), top_freq_power=("power_frac","max"),
    best_plane_H1=("H1_max","max"), best_plane_b1=("b1","max"),
    ambient_H1=("amb_H1","first")).round(3)
print(g.to_string())
