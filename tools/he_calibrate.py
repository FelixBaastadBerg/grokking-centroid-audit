"""Calibrate an absolute lifetime threshold for Betti counting (H1), ambient=200."""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from analysis.tda import prepare_cloud, persistence_diagrams, diagram_summaries
sys.path.insert(0,"tools"); from he_validate_ph import shape, embed
rows=[]
for name in ["circle","figure8","two_circles","torus","sphere","plane"]:
    for noise in (0.0,0.05,0.10):
        for seed in (0,1,2):
            rng=np.random.default_rng(seed)
            X,bt=shape(name,600,rng,noise); X=embed(X,200,rng)
            Xp=prepare_cloud(X,max_points=300,seed=seed,scale="diameter")
            d=persistence_diagrams(Xp,maxdim=1)
            r={"shape":name,"noise":noise,"b1_true":bt[1]}
            r.update(diagram_summaries(d[1],"H1")); rows.append(r)
df=pd.DataFrame(rows); pd.set_option("display.width",220)
cols=[c for c in df.columns if "n_above" in c]
print("=== H1 Betti counts at absolute lifetime thresholds (diameter-normalised clouds) ===")
print(df.groupby(["shape","noise"])[["b1_true"]+cols+["H1_max_pers"]].mean().round(2).to_string())
print("\n=== accuracy of each threshold as a b1 estimator (exact match rate) ===")
for c in cols:
    acc=(df[c]==df["b1_true"]).mean(); print(f"  {c}: {acc:.0%} exact")
df.to_csv("RESULTS/he_betti_calibration.csv",index=False)
