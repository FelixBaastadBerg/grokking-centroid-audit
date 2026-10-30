"""
H-E: validate the PH pipeline on shapes of KNOWN topology before trusting any feature.
  circle b1=1 | torus b1=2,b2=1 | sphere b2=1 | plane trivial | 2 circles b0=2,b1=2 | fig-8 b1=2
Also quantifies what the OLD row_l2 normalisation does to the same shapes.
Cost control: H1 on 300 pts (cheap), H2 on 120 pts (ripser maxdim=2 is ~O(n^3)).
"""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings, time; warnings.filterwarnings("ignore")
from analysis.tda import prepare_cloud, persistence_diagrams, diagram_summaries
from logger import normalize_point_cloud

def embed(X,amb,rng):
    n,d=X.shape
    if amb<=d: return X
    Z=np.zeros((n,amb)); Z[:,:d]=X
    Q,_=np.linalg.qr(rng.normal(size=(amb,amb))); return Z@Q

def shape(name,n,rng,noise=0.0):
    if name=="circle":
        t=rng.uniform(0,2*np.pi,n); X=np.c_[np.cos(t),np.sin(t)]; b=(1,1,0)
    elif name=="torus":
        a=rng.uniform(0,2*np.pi,n); c=rng.uniform(0,2*np.pi,n); R,r=2.0,0.9
        X=np.c_[(R+r*np.cos(c))*np.cos(a),(R+r*np.cos(c))*np.sin(a),r*np.sin(c)]; b=(1,2,1)
    elif name=="sphere":
        V=rng.normal(size=(n,3)); X=V/np.linalg.norm(V,axis=1,keepdims=True); b=(1,0,1)
    elif name=="plane":
        X=rng.uniform(0,1,size=(n,2)); b=(1,0,0)
    elif name=="two_circles":
        m=n//2; t=rng.uniform(0,2*np.pi,m); t2=rng.uniform(0,2*np.pi,n-m)
        X=np.vstack([np.c_[np.cos(t),np.sin(t)],np.c_[np.cos(t2)+6,np.sin(t2)]]); b=(2,2,0)
    elif name=="figure8":
        m=n//2; t=rng.uniform(0,2*np.pi,m); t2=rng.uniform(0,2*np.pi,n-m)
        X=np.vstack([np.c_[np.cos(t)-1,np.sin(t)],np.c_[np.cos(t2)+1,np.sin(t2)]]); b=(1,2,0)
    if noise: X=X+rng.normal(scale=noise,size=X.shape)
    return X,b

def main():
    SHAPES=["circle","torus","sphere","plane","two_circles","figure8"]
    rows=[]; t0=time.time()
    for name in SHAPES:
        for amb in (3,200):
            for noise in (0.0,0.05):
                for seed in (0,1,2):
                    rng=np.random.default_rng(seed)
                    X,bt=shape(name,600,rng,noise)
                    if amb>X.shape[1]: X=embed(X,amb,rng)
                    for mode in ("raw","row_l2_OLDBUG"):
                        Y=normalize_point_cloud(X,mode="row_l2") if mode!="raw" else X
                        r={"shape":name,"ambient":amb,"noise":noise,"seed":seed,"mode":mode,
                           "b0_true":bt[0],"b1_true":bt[1],"b2_true":bt[2]}
                        X1=prepare_cloud(Y,max_points=300,seed=seed,scale="diameter")
                        d1=persistence_diagrams(X1,maxdim=1)
                        r.update(diagram_summaries(d1[0],"H0")); r.update(diagram_summaries(d1[1],"H1"))
                        X2=prepare_cloud(Y,max_points=120,seed=seed,scale="diameter")
                        d2=persistence_diagrams(X2,maxdim=2)
                        r.update(diagram_summaries(d2[2],"H2"))
                        rows.append(r)
        print(f"  {name} done ({time.time()-t0:.0f}s)",flush=True)
    df=pd.DataFrame(rows); df.to_csv("RESULTS/he_ph_validation.csv",index=False)
    pd.set_option("display.width",250)
    print("\n=== RAW clouds (corrected pipeline): detected significant features vs TRUE Betti ===")
    print(df[df["mode"]=="raw"].groupby(["shape","ambient","noise"])[
      ["b0_true","H0_n_significant","b1_true","H1_n_significant","b2_true","H2_n_significant"]].mean().round(2).to_string())
    print("\n=== H1/H2 max persistence, raw, ambient=200, noiseless ===")
    print(df[(df["mode"]=="raw")&(df.ambient==200)&(df.noise==0)].groupby("shape")[
      ["H1_max_pers","H1_total_pers","H1_gap_ratio","H2_max_pers"]].mean().round(3).to_string())
    print("\n=== row_l2 (OLD BUG) vs raw: H1 max persistence, noiseless ===")
    piv=df[df.noise==0].pivot_table(index=["shape","ambient"],columns="mode",values="H1_max_pers").round(3)
    piv["ratio_bug_over_raw"]=(piv["row_l2_OLDBUG"]/piv["raw"]).round(3)
    print(piv.to_string())


if __name__ == '__main__':
    main()
