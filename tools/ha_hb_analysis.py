"""
H-A / H-B: per-checkpoint topology of the integer-embedding cloud, across tasks.

For every run in runs_topo/, at every checkpoint:
  - PH of the embedding cloud {emb(a)} : H0/H1/H2 summaries (validated statistics only)
  - Fourier concentration of the same embeddings (methodologically independent)
  - circularity score (geometric, also independent)
Then align to T_grok and compare cyclic (Z_p) vs non-abelian (S_4) vs non-grokking controls.
"""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings, time; warnings.filterwarnings("ignore")
from analysis.topo_runs import TopoRun, analyse_run, load_runs
from analysis.runlib import RESULTS
from pathlib import Path

OUT = Path("RESULTS"); OUT.mkdir(exist_ok=True)
runs = load_runs()
print(f"{len(runs)} runs found", flush=True)
frames=[]; t0=time.time()
for r in runs:
    df = analyse_run(r, maxdim=2, stride=1, max_points=400, scale="diameter")
    df["groks"] = r.groks
    frames.append(df)
    print(f"  {r.path.name[:60]:<62} ckpts={len(df)} groks={r.groks} "
          f"final_test={r.curves.test_acc.iloc[-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)
if not frames:
    print("no runs yet"); raise SystemExit(0)
D = pd.concat(frames, ignore_index=True)
D.to_csv(OUT/"ha_hb_per_checkpoint.csv", index=False)
print(f"\nwrote {OUT/'ha_hb_per_checkpoint.csv'}  ({len(D)} rows)")

pd.set_option("display.width",240)
print("\n=== run inventory ===")
inv = D.groupby(["task","p","train_fraction","seed"]).agg(
    groks=("groks","first"), t_grok=("t_grok","first"),
    final_test=("test_acc","last"), n_ckpt=("epoch","size")).reset_index()
print(inv.to_string(index=False))

# --- headline: H1 before vs after grokking, per task family ---
print("\n=== H1 loop strength BEFORE vs AFTER the transition ===")
rows=[]
for (task,tf,seed), g in D.groupby(["task","train_fraction","seed"]):
    g=g.sort_values("epoch"); tg=g["t_grok"].iloc[0]; groks=bool(g["groks"].iloc[0])
    if not np.isfinite(tg): tg = g["epoch"].median()
    pre = g[g.epoch < 0.5*tg]; post = g[g.epoch > 1.5*tg]
    if len(pre)<3 or len(post)<3: continue
    rows.append({"task":task,"train_fraction":tf,"seed":seed,"groks":groks,"t_grok":tg,
        "H1max_pre":pre.H1_max_pers.mean(),"H1max_post":post.H1_max_pers.mean(),
        "b1_pre":pre["H1_n_above_0.15"].mean(),"b1_post":post["H1_n_above_0.15"].mean(),
        "fourier_pre":pre.fourier_topk_frac.mean(),"fourier_post":post.fourier_topk_frac.mean(),
        "circcv_pre":pre.circ_radius_cv.mean(),"circcv_post":post.circ_radius_cv.mean()})
S=pd.DataFrame(rows); S.to_csv(OUT/"ha_hb_pre_post.csv",index=False)
if len(S):
    print(S.round(3).to_string(index=False))
    print("\n=== group means (grokking runs only) ===")
    print(S[S.groks].groupby("task")[["H1max_pre","H1max_post","b1_pre","b1_post",
          "fourier_pre","fourier_post","circcv_pre","circcv_post"]].mean().round(3).to_string())
    if (~S.groks).any():
        print("\n=== group means (NON-grokking controls) ===")
        print(S[~S.groks].groupby("task")[["H1max_pre","H1max_post","b1_pre","b1_post",
              "fourier_pre","fourier_post"]].mean().round(3).to_string())
