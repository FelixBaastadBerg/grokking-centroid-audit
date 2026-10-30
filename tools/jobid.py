"""Canonical job id / run-dir name for a config string (single source of truth)."""
import sys, re, math
def parse(cfg):
    return dict(re.findall(r"--(\S+)\s+(\S+)", cfg))
def job_id(cfg): return re.sub(r"[^A-Za-z0-9]+","_",cfg).strip("_")[:80]
def run_dir(cfg):
    g=parse(cfg); task=g.get("task")
    n = math.factorial(int(g.get("sym_n",4))) if task=="perm" else int(g.get("p",113))
    arch=g.get("arch","mlp")
    rid=(f"arch-{arch}__task-{task}__p-{n}__seed-{g.get('seed')}__lr-{g.get('lr','0.01')}"
         f"__wd-{g.get('weight_decay')}__tf-{g.get('train_fraction')}")
    if g.get("batch_size") and g["batch_size"] != "0":
        rid += f"__bs-{g['batch_size']}"
    return rid + (f"__{g['tag']}" if g.get("tag") else "")
if __name__=="__main__":
    a=sys.argv[1:]
    print(run_dir(a[1]) if a and a[0]=="--rundir" else job_id(a[0]))
