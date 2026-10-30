"""
v3 overnight training queue -- the runs T2.1 and T2.2 need, nothing else.

T2.1 (cross-cell recombination) needs >= 3 seeds of the Z_59 addition MLP cell; seeds 0 and 1
exist (tag hb), so seeds 2-4 are trained here with identical hyperparameters. The S_4 MLP cell
(task-perm p-24, 5 seeds) and the Transformer cell (ss777, 3 seeds, full_traj) already exist.

T2.2 (make class-conditional ID measurable) trains the narrow-width arm: hidden 8/16/32 on
Z_23 (m ~ 18 per class) and hidden 16/32 on Z_59 (m ~ 47 per class, the regime where the
conditional estimators are in calibration for d <= width). Whether these grok is itself part of
the result and is reported either way.

Queue discipline: at most MAX_CONCURRENT training processes (each pinned to 2 torch threads),
so the box stays usable. Skips any run whose output directory already holds a trajectory.npz,
so the queue is resumable.
"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGS = REPO / "logs" / "v3_overnight"
LOGS.mkdir(parents=True, exist_ok=True)
MAX_CONCURRENT = 5

Z59 = dict(task="add", p=59, tf=0.8, epochs=60000, hidden=200, tag="hb")
JOBS = []
# T2.1: three more Z_59 addition MLP seeds, hyperparameters identical to the existing hb pair
for seed in (2, 3, 4):
    JOBS.append(dict(**Z59, seed=seed))
# T2.2: narrow widths at p=23 (the canonical cell's operating point, m ~ 18)
for hidden in (8, 16, 32):
    for seed in (0, 1, 2):
        JOBS.append(dict(task="add", p=23, tf=0.8, epochs=50000, hidden=hidden,
                         tag=f"w{hidden}", seed=seed))
# T2.2: narrow widths at p=59 (m ~ 47, conditional estimators in calibration for d <= width)
for hidden in (16, 32):
    for seed in (0, 1, 2):
        JOBS.append(dict(task="add", p=59, tf=0.8, epochs=60000, hidden=hidden,
                         tag=f"w{hidden}", seed=seed))
# wave 2: narrow-width grokking at p=23 is fragile (seeds 1-2 failed at both widths), so more
# seeds are needed to get >= 3 grokked runs per width
for hidden in (16, 32):
    for seed in (3, 4, 5, 6):
        JOBS.append(dict(task="add", p=23, tf=0.8, epochs=50000, hidden=hidden,
                         tag=f"w{hidden}", seed=seed))


def run_id(j):
    return (f"arch-mlp__task-{j['task']}__p-{j['p']}__seed-{j['seed']}"
            f"__lr-0.01__wd-0.03__tf-{j['tf']}__{j['tag']}")


def done(j):
    return (REPO / "runs_topo" / run_id(j) / "trajectory.npz").exists()


def launch(j):
    log = open(LOGS / (run_id(j) + ".log"), "w", encoding="utf-8")
    cmd = [PY, str(REPO / "tools" / "train_topology.py"),
           "--task", j["task"], "--p", str(j["p"]), "--seed", str(j["seed"]),
           "--lr", "0.01", "--weight_decay", "0.03",       # the corpus cell convention
           "--train_fraction", str(j["tf"]), "--num_epochs", str(j["epochs"]),
           "--hidden", str(j["hidden"]), "--tag", j["tag"]]
    print(f"LAUNCH {run_id(j)}", flush=True)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO)), log, j


def main():
    queue = [j for j in JOBS if not done(j)]
    print(f"{len(JOBS)} jobs, {len(JOBS) - len(queue)} already done, {len(queue)} to run",
          flush=True)
    active, failures = [], []
    while queue or active:
        while queue and len(active) < MAX_CONCURRENT:
            active.append(launch(queue.pop(0)))
        time.sleep(20)
        still = []
        for proc, log, j in active:
            rc = proc.poll()
            if rc is None:
                still.append((proc, log, j))
                continue
            log.close()
            if rc == 0 and done(j):
                print(f"DONE {run_id(j)}", flush=True)
            else:
                failures.append(run_id(j))
                print(f"FAILED rc={rc} {run_id(j)}", flush=True)
        active = still
    print(f"ALL FINISHED, {len(failures)} failures" + (": " + ", ".join(failures)
                                                       if failures else ""), flush=True)


if __name__ == "__main__":
    main()
