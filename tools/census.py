import json, collections
from pathlib import Path
R = Path("data_runs")
metas = []
for d in sorted(R.iterdir()):
    p = d/"run_meta.json"
    if p.exists(): metas.append((d.name, json.loads(p.read_text())))
print("runs:", len(metas))
for k in ("lr","weight_decay","train_fraction","seed","num_epochs","log_frequency","batch_size"):
    c = collections.Counter(m[k] if not isinstance(m[k], list) else tuple(m[k]) for _, m in metas)
    print(f"\n{k}: {len(c)} distinct")
    for v, n in sorted(c.items(), key=lambda x: (x[0] is None, x[0])): print(f"    {v!r:>12}  n={n}")
gr = collections.Counter((m["train_fraction"], m["lr"], m["weight_decay"]) for _, m in metas)
print(f"\n(tf,lr,wd) cells: {len(gr)}; runs with duplicate cells: {sum(v for v in gr.values() if v>1)}")
print("dupes:", [k for k,v in gr.items() if v>1][:10])
