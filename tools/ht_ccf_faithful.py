"""
FIDELITY CHECK on the Monod comparison (2605.06352).

Their timing method, quoted: "cross-correlation function (CCF) between the first order
difference for the test accuracy and [the topological metrics]". Peak at NEGATIVE lag
means test accuracy changes PRECEDE the topological shift (their transformer result,
~1000 steps); peak near zero means simultaneous (their MLP result).

Our earlier comparison used a different statistic entirely (phase within the plateau),
so it was not a faithful reproduction of their protocol. Here we implement THEIR
metric and apply it to both their statistic and ours on identical runs.

Convention (matching theirs): ccf at lag L correlates d(test_acc)[t] with d(metric)[t+L].
Peak at L > 0 => the metric moves AFTER accuracy (lags). L < 0 => metric leads.
"""
import sys, os, warnings; sys.path.insert(0,'.')
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from analysis.topo_runs import TopoRun

def ccf_peak(a, b, max_lag=20):
    """a, b already first-differenced and aligned. Returns (best_lag, best_r)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = len(a); best = (0, 0.0)
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            x, y = a[:n - L], b[L:]
        else:
            x, y = a[-L:], b[:n + L]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 12 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
            continue
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        if abs(r) > abs(best[1]):
            best = (L, r)
    return best

E = pd.read_csv('RESULTS/temporal_equiv_series.csv')
rows = []
for d in sorted(Path('results_runs').glob('*')):
    pc = d / 'per_checkpoint.csv'; rd = Path('runs_topo') / d.name
    if not pc.exists() or not (rd / 'trajectory.npz').exists(): continue
    r = TopoRun(rd)
    if r.curves.test_acc.max() < 0.5: continue
    P = pd.read_csv(pc).sort_values('epoch')
    if 'amb_H1_max' not in P or 'test_acc' not in P: continue
    step = int(np.median(np.diff(P.epoch.values))) or 1
    dacc = np.diff(P.test_acc.values)
    out = {'run': d.name, 'arch': r.meta.get('arch','mlp'), 'task': r.meta['task'], 'p': r.p,
           'tf': r.meta['train_fraction'], 'seed': r.meta['seed'], 'ckpt_step': step}
    L, rr = ccf_peak(dacc, np.diff(P.amb_H1_max.values))
    out['monod_lag_ckpts'] = L; out['monod_lag_steps'] = L * step; out['monod_r'] = round(rr, 3)
    sub = E[E.run == d.name].sort_values('epoch')
    if len(sub) == len(P):
        L2, r2 = ccf_peak(dacc, np.diff(sub.equiv_R2.values))
        out['ours_lag_ckpts'] = L2; out['ours_lag_steps'] = L2 * step; out['ours_r'] = round(r2, 3)
    rows.append(out)

D = pd.DataFrame(rows); D.to_csv('RESULTS/ccf_faithful.csv', index=False)
pd.set_option('display.width', 240)
print(f"=== THEIR metric (CCF of first differences) on our runs, n={len(D)} ===")
print("   lag > 0 => metric moves AFTER accuracy (lags), matching their transformer finding\n")
for arch, sa in D.groupby('arch'):
    print(f"  {arch} (n={len(sa)}):")
    for col, nm in (('monod_lag_steps','H1 max persistence (theirs)'), ('ours_lag_steps','equiv_R2 (ours)')):
        if col not in sa: continue
        s = sa[col].dropna()
        if not len(s): continue
        print(f"     {nm:<28} median lag {s.median():+8.0f} steps   lags(>0) {int((s>0).sum())}/{len(s)}   leads(<0) {int((s<0).sum())}/{len(s)}   at-zero {int((s==0).sum())}")
