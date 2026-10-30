import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, json
pd.set_option("display.width",240); pd.set_option("display.max_columns",60)
df = pd.read_csv("results/phase0_estimator_audit.csv")
EST = ["mst_dim","mst_alpha","mst_dim_from_alpha","ph0_dim_scaling","corr_dim","twonn_id","pca_participation_ratio"]
print("=== recovery of TRUE intrinsic dimension (mean over M,seed), norm_mode=none ===")
a = df[df.norm_mode=="none"].groupby(["shape","true_d"])[EST].mean().round(2)
print(a.to_string())
print("\n=== same, norm_mode=row_l2  (what logger.py ACTUALLY does) ===")
b = df[df.norm_mode=="row_l2"].groupby(["shape","true_d"])[EST].mean().round(2)
print(b.to_string())
print("\n=== non-finite rate per estimator ===")
for nm in ("none","row_l2"):
    s = df[df.norm_mode==nm][EST].apply(lambda c: (~np.isfinite(c)).mean())
    print(f"  {nm:<7}", {k: f"{v:.0%}" for k,v in s.items()})
print("\n=== mst_alpha range (alpha->1 is the blow-up regime; mst_dim=1/(1-alpha)) ===")
print(df.groupby("norm_mode")["mst_alpha"].describe()[["min","25%","50%","75%","max"]].round(4).to_string())
print("\n=== correlation of each estimator with true_d (Spearman, across shapes) ===")
for nm in ("none","row_l2"):
    d=df[df.norm_mode==nm]
    print(f"  {nm:<7}", {e: round(float(d[[e,'true_d']].replace([np.inf,-np.inf],np.nan).dropna().corr(method='spearman').iloc[0,1]),3) for e in EST})
print("\n=== MST vs ripser PH0 consistency ===")
print(json.dumps(json.load(open("results/phase0_mst_vs_ph0.json")), indent=1))
print("\n=== weight metrics vs known rank ===")
print(pd.read_csv("results/phase0_weight_metrics.csv").round(2).to_string(index=False))
