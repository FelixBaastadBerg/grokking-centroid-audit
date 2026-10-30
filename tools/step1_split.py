import sys; sys.path.insert(0,".")
from analysis.runlib import load_runs, make_split, split_runs, run_table, RESULTS
import numpy as np, pandas as pd
runs, dropped = load_runs(verbose=True)
print(f"\nloaded {len(runs)} runs, dropped {len(dropped)}")
split = make_split(runs)
print(f"split: {split['n_discovery']} discovery / {split['n_confirmation']} confirmation "
      f"({split['n_confirm_granules']}/{split['n_granules']} granules)")
tb = run_table(runs)
tb["set"] = tb.run_id.map(split["assignment"])
tb.to_csv(RESULTS/"run_table.csv", index=False)
print("\nT_grok summary (all kept runs):")
print(tb[["t_grok","t_grok_gap","t_grok_thresh50"]].describe().to_string())
print("\nper set:"); print(tb.groupby("set")["t_grok"].agg(["count","mean","std","min","max"]).to_string())
# WITHIN-GRANULE VARIANCE OF T_grok -- Phase 1 check #2
print("\n=== within-granule variance of T_grok ===")
for vary in ["weight_decay","lr","train_fraction"]:
    fixed = [k for k in ["train_fraction","lr","weight_decay"] if k != vary]
    g = tb.groupby(fixed)["t_grok"]
    sizes = g.size(); ok = sizes[sizes>=3].index
    sub = tb.set_index(fixed).loc[ok].reset_index()
    within = sub.groupby(fixed)["t_grok"].std(ddof=1).dropna()
    print(f"vary={vary:<15} granules>=3: {len(ok):>3}  within-granule std: "
          f"median={within.median():8.1f} mean={within.mean():8.1f}   | overall std={tb.t_grok.std():8.1f}")
