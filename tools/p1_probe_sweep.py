"""
PRIORITY 1 -- probe placement. Where does the circle live, if anywhere?

The Nanda circle is a PER-NUMBER object: residue n -> a point, giving a p-point
cloud. There are several defensible ways to read that off an MLP with one-hot
inputs, and 'no loop' is only meaningful if it holds for all of them. We test:

  emb_a        W0[:, n]                  raw first-layer column for integer n (a-slot)
  emb_b        W0[:, p+n]                same for the b-slot
  emb_ab       concat of the two
  relu_a       relu(W0[:, n])            first-layer RESPONSE to the single integer n
  fwd_a0       relu(W0 @ [e_n ; e_0])    full forward response, b pinned to 0
  marg_a       mean_b relu(W0 @ [e_n;e_b])  per-number representation marginalised over b
  layer2_a     second-layer response, marginalised over b
  unemb        W2[n, :]                  output/unembedding row for class n

Each cloud is p points. All go through the SAME validated pipeline (raw, diameter
normalisation, H1_max_pers + H1_n_above_0.15), and every run is compared against a
synthetic p-point circle as a positive control.
"""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from analysis.tda import prepare_cloud, persistence_diagrams, diagram_summaries
from analysis.topo_runs import load_runs, fourier_concentration, circularity_score

def probes(run, idx=-1):
    W0=run.traj["W0"][idx]; W1=run.traj["W1"][idx]; W2=run.traj["W2"][idx]
    p=run.p; A=W0[:,:p]; B=W0[:,p:2*p]
    relu=lambda z: np.maximum(z,0.0)
    out={}
    out["emb_a"]=A.T; out["emb_b"]=B.T
    out["emb_ab"]=np.concatenate([A.T,B.T],axis=1)
    out["relu_a"]=relu(A).T
    out["fwd_a0"]=relu(A+B[:,[0]]).T
    # marginalise over b: mean_b relu(A[:,n]+B[:,b])
    M=np.stack([relu(A+B[:,[j]]).mean(axis=1) for j in range(p)],axis=0)  # wrong axis guard below
    M=np.stack([relu(A[:,n][:,None]+B).mean(axis=1) for n in range(p)],axis=0)
    out["marg_a"]=M
    out["layer2_a"]=np.stack([relu(W1@relu(A[:,n][:,None]+B)).mean(axis=1) for n in range(p)],axis=0)
    out["unemb"]=W2
    return out

def feats(X,seed=0):
    Xp=prepare_cloud(X,max_points=400,seed=seed,scale="diameter")
    if Xp.shape[0]<8: return {}
    d=persistence_diagrams(Xp,maxdim=1)
    s=diagram_summaries(d[1],"H1")
    return {"H1_max":s["H1_max_pers"],"b1":s["H1_n_above_0.15"],"gap":s["H1_gap_ratio"]}

rows=[]
for r in load_runs():
    P=probes(r,-1)
    base={"run":r.path.name,"task":r.meta["task"],"p":r.p,"tf":r.meta["train_fraction"],
          "seed":r.meta["seed"],"final_test":float(r.curves.test_acc.iloc[-1])}
    for name,X in P.items():
        f=feats(X)
        if not f: continue
        fc=fourier_concentration(X) if X.shape[0]==r.p else {}
        rows.append({**base,"probe":name,**f,
                     "fourier_top5":fc.get("fourier_topk_frac",np.nan),
                     "fourier_neff":fc.get("fourier_n_eff",np.nan)})
    # positive control: synthetic circle at this p, same pipeline
    t=np.linspace(0,2*np.pi,r.p,endpoint=False); C=np.c_[np.cos(t),np.sin(t)]
    Z=np.zeros((r.p,200)); Z[:,:2]=C
    rows.append({**base,"probe":"SYNTH_circle",**feats(Z),
                 "fourier_top5":fourier_concentration(Z)["fourier_topk_frac"],
                 "fourier_neff":fourier_concentration(Z)["fourier_n_eff"]})
D=pd.DataFrame(rows); D.to_csv("RESULTS/p1_probe_sweep.csv",index=False)
pd.set_option("display.width",250)
print("=== H1 loop strength by PROBE PLACEMENT (final checkpoint) ===")
piv=D.pivot_table(index=["task","p","tf","final_test"],columns="probe",values="H1_max").round(3)
order=["SYNTH_circle","emb_a","emb_b","emb_ab","relu_a","fwd_a0","marg_a","layer2_a","unemb"]
print(piv[[c for c in order if c in piv.columns]].to_string())
print("\n=== b1 (calibrated count at lifetime>=0.15) ===")
piv2=D.pivot_table(index=["task","p","tf","final_test"],columns="probe",values="b1").round(2)
print(piv2[[c for c in order if c in piv2.columns]].to_string())
print("\n=== best non-synthetic probe per run ===")
ns=D[D.probe!="SYNTH_circle"]
best=ns.loc[ns.groupby(["run"])["H1_max"].idxmax()][["task","p","tf","final_test","probe","H1_max","b1","fourier_neff"]]
print(best.sort_values(["task","tf"]).round(3).to_string(index=False))
