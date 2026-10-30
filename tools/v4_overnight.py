"""
v4 Phase-1 training queue.

P1.1  Z_59 addition, width 16, ONE fixed configuration (lr 0.01, wd 0.03, tf 0.8, 60k epochs,
      stablemax, full batch) -- seeds 3..19, joining the existing seeds 0-2 for 20 seeds total.
      Every seed is kept in the analysis, grokked or not.

P1.2  Z_23 addition, width 200 (the corpus standard), weight decay swept over
      {0, 0.003, 0.03, 0.1}, 3 seeds each; wd = 0.03 seeds already exist as the canonical cell,
      so only the other three levels are trained.

Same queue discipline as v3_overnight.py: MAX_CONCURRENT processes, resumable, skips completed.
"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGS = REPO / "logs" / "v4_overnight"
LOGS.mkdir(parents=True, exist_ok=True)
MAX_CONCURRENT = 5

JOBS = []
# P1.1: 17 more seeds of the fixed Z59 w16 configuration
for seed in range(3, 20):
    JOBS.append(dict(task="add", p=59, tf=0.8, epochs=60000, hidden=16, wd=0.03,
                     tag="w16", seed=seed))
# P1.2: weight-decay sweep at width 200 (wd=0.03 is the existing canonical cell)
for wd in (0.0, 0.003, 0.1):
    for seed in (0, 1, 2):
        JOBS.append(dict(task="add", p=23, tf=0.8, epochs=50000, hidden=200, wd=wd,
                         tag="wdsweep", seed=seed))


def run_id(j):
    # train_topology.py embeds str(float(wd)) in the directory name, so match it exactly
    return (f"arch-mlp__task-{j['task']}__p-{j['p']}__seed-{j['seed']}"
            f"__lr-0.01__wd-{j['wd']}__tf-{j['tf']}__{j['tag']}")


def done(j):
    return (REPO / "runs_topo" / run_id(j) / "trajectory.npz").exists()


def launch(j):
    log = open(LOGS / (run_id(j) + ".log"), "w", encoding="utf-8")
    cmd = [PY, str(REPO / "tools" / "train_topology.py"),
           "--task", j["task"], "--p", str(j["p"]), "--seed", str(j["seed"]),
           "--lr", "0.01", "--weight_decay", str(j["wd"]),
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
