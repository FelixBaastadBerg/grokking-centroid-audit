"""
Supervisor: reconcile sentinels from disk, mark analysis state, report.
Completion is read from disk, never inferred from a command returning.
"""
import sys, json; sys.path.insert(0,"."); sys.path.insert(0,"tools")
from pathlib import Path
import jobctl

def refresh_from_disk() -> int:
    reg = json.loads(Path("jobs/registry.json").read_text() or "{}")
    changed = 0
    for jid, rec in reg.items():
        st = rec.get("status", "RUNNING")
        rd = rec.get("run_dir")
        if not rd or st in ("DONE_ANALYZED", "FAILED"):
            continue
        trained = (Path("runs_topo")/rd/"trajectory.npz").exists() and (Path("runs_topo")/rd/"curves.csv").exists()
        analyzed = (Path("results_runs")/rd/"COMPLETE").exists()
        if trained and analyzed and st != "DONE_ANALYZED":
            jobctl.mark(jid, "DONE_ANALYZED"); changed += 1
    return changed

if __name__ == "__main__":
    print(f"[supervise] newly DONE_ANALYZED: {refresh_from_disk()}")
    print(jobctl.summary())
