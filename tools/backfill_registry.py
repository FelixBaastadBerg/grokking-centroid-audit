"""Register the in-flight sweep: derive each config's expected run dir and set status from disk."""
import sys, re, json; sys.path.insert(0,"."); sys.path.insert(0,"tools")
from pathlib import Path
import jobctl

def run_id_for(cfg: str) -> str:
    g = dict(re.findall(r"--(\S+)\s+(\S+)", cfg))
    task=g.get("task"); p=g.get("p"); sym=g.get("sym_n","5")
    n = int(sym) and (__import__("math").factorial(int(sym)) if task=="perm" else int(p))
    tag=g.get("tag","")
    rid=(f"arch-mlp__task-{task}__p-{n}__seed-{g.get('seed')}__lr-{g.get('lr','0.01')}"
         f"__wd-{g.get('weight_decay')}__tf-{g.get('train_fraction')}")
    return rid + (f"__{tag}" if tag else "")

cfgs=[l.strip() for l in open("logs/batch_configs.txt") if l.strip()]
reg_path=Path("jobs/registry.json")
n_done=n_run=0
for c in cfgs:
    rid=run_id_for(c)
    jid=re.sub(r"[^A-Za-z0-9]+","_",c).strip("_")[:80]
    d=Path("runs_topo")/rid
    complete=(d/"trajectory.npz").exists() and (d/"curves.csv").exists()
    jobctl.register(jid, c, 0, "train")
    if complete: jobctl.mark(jid,"DONE"); n_done+=1
    else:        jobctl.mark(jid,"RUNNING"); n_run+=1
    reg=json.loads(reg_path.read_text()); reg[jid]["run_dir"]=rid
    reg_path.write_text(json.dumps(reg,indent=2))
print(f"backfilled {len(cfgs)} jobs: DONE={n_done} RUNNING={n_run}")
