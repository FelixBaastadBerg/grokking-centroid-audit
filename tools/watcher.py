"""
Long-lived job watcher.

Polls the registry + disk sentinels every ~30 s and prints ONE line whenever a job
CHANGES state. Designed to be run as a background task and polled with BashOutput.

Never exits while any job is RUNNING or awaiting analysis. Only stops if
jobs/STOP_WATCHER appears, so a newly queued batch is picked up without restarting it.

State is derived from disk (trajectory.npz / results COMPLETE / STATUS files), so a
job that dies with a stale registry entry is still reported correctly.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REG = REPO / "jobs" / "registry.json"
RUNS = REPO / "runs_topo"
RES = REPO / "results_runs"
STOP = REPO / "jobs" / "STOP_WATCHER"

POLL = 30
HEARTBEAT_EVERY = 10          # cycles between liveness summaries


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def descr(run_dir: str | None, cmd: str | None) -> str:
    """Compact 'task tf/seed' label for the report line."""
    s = run_dir or cmd or ""
    task = re.search(r"task-(\w+)", s) or re.search(r"--task\s+(\w+)", s)
    tf = re.search(r"tf-([\d.]+)", s) or re.search(r"train_fraction\s+([\d.]+)", s)
    sd = re.search(r"seed-(\d+)", s) or re.search(r"--seed\s+(\d+)", s)
    p = re.search(r"p-(\d+)", s)
    return (f"task={task.group(1) if task else '?'} p={p.group(1) if p else '?'} "
            f"tf={tf.group(1) if tf else '?'} seed={sd.group(1) if sd else '?'}")


def disk_state(jid: str, rec: dict) -> str:
    """Truth from disk, falling back to the registry's own STATUS file."""
    rd = rec.get("run_dir")
    sf = REPO / "jobs" / jid / "STATUS"
    status = sf.read_text(encoding="utf-8").strip() if sf.exists() else rec.get("status", "UNKNOWN")
    if status.startswith("FAILED"):
        return status
    if rd:
        trained = (RUNS / rd / "trajectory.npz").exists() and (RUNS / rd / "curves.csv").exists()
        analyzed = (RES / rd / "COMPLETE").exists()
        if trained and analyzed:
            return "DONE_ANALYZED"
        if trained:
            return "TRAINED_PENDING_ANALYSIS"
    return status


def tail_error(jid: str, n: int = 3) -> str:
    log = REPO / "jobs" / jid / "log"
    if not log.exists():
        return ""
    lines = [l.rstrip() for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    return " | ".join(lines[-n:])[:300]


def main() -> int:
    last: dict[str, str] = {}
    cycle = 0
    print(f"[{ts()}] WATCHER START -- polling every {POLL}s; reports state changes only", flush=True)
    while True:
        cycle += 1
        try:
            reg = json.loads(REG.read_text(encoding="utf-8") or "{}")
        except Exception as e:
            print(f"[{ts()}] registry unreadable: {e}", flush=True)
            time.sleep(POLL); continue

        counts = {"RUNNING": 0, "DONE_ANALYZED": 0, "TRAINED_PENDING_ANALYSIS": 0, "FAILED": 0, "OTHER": 0}
        for jid, rec in reg.items():
            st = disk_state(jid, rec)
            key = st if st in counts else ("FAILED" if st.startswith("FAILED") else "OTHER")
            counts[key] += 1
            if last.get(jid) != st:
                if jid in last:                        # a genuine transition
                    extra = ""
                    if st.startswith("FAILED"):
                        extra = f"  ERROR: {tail_error(jid)}"
                    elif st == "DONE_ANALYZED":
                        sj = RES / (rec.get("run_dir") or "") / "summary.json"
                        if sj.exists():
                            try:
                                s = json.loads(sj.read_text(encoding="utf-8"))
                                extra = (f"  test={s.get('final_test_acc'):.3f}"
                                         f" groks={s.get('groks')}"
                                         f" amb_b1={s.get('ambient_b1')}"
                                         f" planeH1={s.get('plane_H1_real')}"
                                         f" z_vs_null={s.get('plane_H1_z_vs_null')}")
                            except Exception:
                                pass
                    print(f"[{ts()}] {last.get(jid)} -> {st}  {descr(rec.get('run_dir'), rec.get('cmd'))}"
                          f"  [{jid[:52]}]{extra}", flush=True)
                else:
                    print(f"[{ts()}] TRACK {st:<26} {descr(rec.get('run_dir'), rec.get('cmd'))}"
                          f"  [{jid[:52]}]", flush=True)
                last[jid] = st

        active = counts["RUNNING"] + counts["TRAINED_PENDING_ANALYSIS"]
        if cycle % HEARTBEAT_EVERY == 0 or (active == 0 and cycle % 2 == 0):
            print(f"[{ts()}] HEARTBEAT total={len(reg)} running={counts['RUNNING']} "
                  f"pending_analysis={counts['TRAINED_PENDING_ANALYSIS']} "
                  f"done_analyzed={counts['DONE_ANALYZED']} failed={counts['FAILED']} "
                  f"-> {'ACTIVE' if active else 'IDLE (no jobs running)'}", flush=True)

        if STOP.exists():
            print(f"[{ts()}] STOP flag present -- watcher exiting (active={active})", flush=True)
            return 0
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
