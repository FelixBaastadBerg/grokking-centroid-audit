"""PHASE 1: negative controls + run-level permutation, DISCOVERY only."""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd
from analysis.runlib import load_runs, split_runs, RESULTS
from analysis import predictor as P
pd.set_option("display.width",220); pd.set_option("display.max_columns",50)

runs,_ = load_runs(); disc, conf = split_runs(runs)
real = P.signal_dim("layers.1","mst_dim")

SIGNALS = [
 ("REAL  layers.1|mst_dim",              real),
 ("REAL  layers.2|mst_alpha",            P.signal_dim("layers.2","mst_alpha")),
 ("REAL  layers.0|pca_participation_ratio", P.signal_dim("layers.0","pca_participation_ratio")),
 ("CTRL  fixed sigmoid (same for all runs)", P.control_fixed_sigmoid()),
 ("CTRL  phase-shuffled mst_dim",        P.control_phase_shuffled(real)),
 ("CTRL  time-reversed mst_dim",         P.control_time_reversed(real)),
 ("ALT   train loss",                    P.signal_metrics("loss","train")),
 ("ALT   train accuracy",                P.signal_metrics("accuracy","train")),
 ("ALT   weights_l2 layers.0.weight",    P.signal_weights_l2("layers.0.weight")),
 ("ALT   weights_l2 layers.1.weight",    P.signal_weights_l2("layers.1.weight")),
 ("ALT   weights_l2 layers.2.weight",    P.signal_weights_l2("layers.2.weight")),
 ("ALT   compression_lzma_bits_q8",      P.signal_metrics("compression_lzma_bits_q8","general")),
 ("ALT   softmax_collapse",              P.signal_metrics("softmax_collapse","train")),
]
rows=[]
for label, sig in SIGNALS:
    df = P.build_tq_table(disc, sig)
    if len(df) < 10:
        rows.append({"signal":label,"n_fit":len(df),"note":"too few logistic fits"}); continue
    res,_,_ = P.eval_nested(df)
    res2,_,_ = P.eval_leaky(df)
    rows.append({"signal":label,"n_fit":len(df),"q_star":res2.get("q_star"),
                 "mae":res["mae"],"rmse":res["rmse"],"r2_fit":res["r2_fit"],"r2_oos":res["r2_oos"]})
t = pd.DataFrame(rows).sort_values("r2_oos",ascending=False,na_position="last")
print("=== NESTED-CV predictor skill by signal (DISCOVERY, n=51) ===")
print(t.to_string(index=False))
t.to_csv(RESULTS/"phase1_controls_discovery.csv",index=False)

# ---- run-level permutation on the real signal: must collapse to chance ----
df = P.build_tq_table(disc, real)
rng = np.random.default_rng(0); perm_r2=[]
for _ in range(200):
    d2 = df.copy(); d2["target"] = rng.permutation(d2["target"].to_numpy())
    r,_,_ = P.eval_nested(d2); perm_r2.append(r["r2_oos"])
perm_r2=np.array(perm_r2,float)
real_r2 = P.eval_nested(df)[0]["r2_oos"]
print(f"\n=== run-level permutation (200 shuffles, layers.1|mst_dim) ===")
print(f"real r2_oos = {real_r2:.4f}")
print(f"permuted    mean={np.nanmean(perm_r2):.4f}  p95={np.nanpercentile(perm_r2,95):.4f}  max={np.nanmax(perm_r2):.4f}")
print(f"empirical p = {(np.sum(perm_r2>=real_r2)+1)/(len(perm_r2)+1):.4f}")
