"""
DECISIVE CONTROL for the per-frequency circle result.

Projecting a cloud onto the 2-D plane spanned by Re F[k], Im F[k] -- a basis derived
FROM THE CLOUD ITSELF -- could manufacture a circle from nothing. If an isotropic
random embedding also yields b1 = 1 in its own top-frequency plane, the measurement
is vacuous and the result must be withdrawn.

Nulls tested, all pushed through the identical pipeline:
  random_gauss   iid Gaussian embedding, no structure whatsoever
  random_perm    the REAL embedding with its rows randomly permuted (destroys the
                 map n -> emb(n) while preserving the exact multiset of vectors)
  synth_kfreq    synthetic k-frequency embedding (positive control)
  real           the actual trained embeddings
"""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from analysis.tda import prepare_cloud, persistence_diagrams, diagram_summaries
from analysis.topo_runs import load_runs
sys.path.insert(0,"tools")
from p1b_frequency_circles import freq_decompose, circ_feats

def top_plane(E, rank=0):
    q,fr=freq_decompose(E)
    if not fr: return None,None
    order=sorted(fr.items(),key=lambda kv:-kv[1][0])
    if rank>=len(order): return None,None
    k,(pw,P)=order[rank]
    return pw,P

rows=[]
rng=np.random.default_rng(0)
for r in load_runs():
    if r.meta["task"]=="perm": continue
    p,h=r.p,r.traj["W0"].shape[1]
    E=r.embeddings("a",-1)
    variants={}
    variants["real"]=E
    variants["random_perm"]=E[rng.permutation(p)]
    variants["random_gauss"]=rng.normal(size=(p,h))*E.std()
    n=np.arange(p); S=np.zeros((p,h))
    for k in rng.choice(np.arange(1,p//2),size=5,replace=False):
        S+=np.c_[np.cos(2*np.pi*k*n/p),np.sin(2*np.pi*k*n/p)]@rng.normal(size=(2,h))
    variants["synth_5freq"]=S
    n1=np.arange(p); S1=np.c_[np.cos(2*np.pi*3*n1/p),np.sin(2*np.pi*3*n1/p)]@rng.normal(size=(2,h))
    variants["synth_1freq"]=S1
    for name,X in variants.items():
        amb=circ_feats(X); pw,P=top_plane(X)
        pf=circ_feats(P) if P is not None else {}
        rows.append({"run":r.path.name[:40],"task":r.meta["task"],"p":p,"tf":r.meta["train_fraction"],
                     "final_test":round(float(r.curves.test_acc.iloc[-1]),3),"variant":name,
                     "top_power":pw,"amb_H1":amb["H1_max"],"amb_b1":amb["b1"],
                     "plane_H1":pf.get("H1_max"),"plane_b1":pf.get("b1"),
                     "plane_radius_cv":pf.get("radius_cv")})
D=pd.DataFrame(rows); D.to_csv("RESULTS/p1c_null_control.csv",index=False)
pd.set_option("display.width",250)
print("=== NULL CONTROL: does projecting onto a self-derived Fourier plane manufacture a circle? ===\n")
g=D.groupby("variant").agg(n=("variant","size"),top_power=("top_power","mean"),
    amb_H1=("amb_H1","mean"),amb_b1=("amb_b1","mean"),
    plane_H1=("plane_H1","mean"),plane_b1=("plane_b1","mean"),
    plane_radius_cv=("plane_radius_cv","mean")).round(3)
print(g.loc[[v for v in ["synth_1freq","synth_5freq","real","random_perm","random_gauss"] if v in g.index]].to_string())
print("\n=== real runs only, per run ===")
print(D[D.variant=="real"][["task","p","tf","final_test","top_power","amb_H1","plane_H1","plane_b1","plane_radius_cv"]].round(3).to_string(index=False))
print("\n=== VERDICT ===")
real=D[D.variant=="real"]; rg=D[D.variant=="random_gauss"]; rp=D[D.variant=="random_perm"]
print(f"real plane_H1      = {real.plane_H1.mean():.3f} +/- {real.plane_H1.std():.3f}")
print(f"random_gauss       = {rg.plane_H1.mean():.3f} +/- {rg.plane_H1.std():.3f}")
print(f"random_perm        = {rp.plane_H1.mean():.3f} +/- {rp.plane_H1.std():.3f}")
print(f"top-frequency power: real={real.top_power.mean():.3f} gauss={rg.top_power.mean():.3f} perm={rp.top_power.mean():.3f}")
