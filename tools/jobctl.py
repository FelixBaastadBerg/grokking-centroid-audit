"""
Job registry + sentinel protocol.

Completion is decided by SENTINELS ON DISK and liveness of the recorded PID -- never
by a foreground command returning. Each job owns jobs/<id>/ containing:
    STATUS      one of RUNNING | DONE | FAILED
    PID         the OS pid of the worker
    COMPLETE    sentinel file, written only on success
    log         stdout/stderr
jobs/registry.json maps id -> {status, analyzed, ...}.

'All finished' is only true when every entry is (DONE and analyzed) or FAILED.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOBS = REPO / "jobs"
REG  = JOBS / "registry.json"
JOBS.mkdir(exist_ok=True)


def _load() -> dict:
    if REG.exists():
        try: return json.loads(REG.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}


def _save(reg: dict) -> None:
    REG.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def pid_alive(pid: int) -> bool:
    if not pid: return False
    try:
        out = subprocess.run(["powershell","-NoProfile","-Command",
                              f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{'1'}} else {{'0'}}"],
                             capture_output=True, timeout=30).stdout.decode().strip()
        return out.endswith("1")
    except Exception:
        return False


def register(job_id: str, cmd: str, pid: int, kind: str = "train") -> None:
    d = JOBS / job_id; d.mkdir(parents=True, exist_ok=True)
    (d / "STATUS").write_text("RUNNING", encoding="utf-8")
    (d / "PID").write_text(str(pid), encoding="utf-8")
    if (d / "COMPLETE").exists(): (d / "COMPLETE").unlink()
    reg = _load()
    reg[job_id] = {"status": "RUNNING", "analyzed": False, "kind": kind,
                   "cmd": cmd, "pid": pid, "t_start": time.time()}
    _save(reg)


def mark(job_id: str, status: str) -> None:
    d = JOBS / job_id; d.mkdir(parents=True, exist_ok=True)
    (d / "STATUS").write_text(status, encoding="utf-8")
    if status in ("DONE", "DONE_ANALYZED"): (d / "COMPLETE").write_text("ok", encoding="utf-8")
    if status == "DONE_ANALYZED":
        reg0 = _load(); reg0.setdefault(job_id, {})["analyzed"] = True; _save(reg0)
    reg = _load()
    reg.setdefault(job_id, {})["status"] = status
    reg[job_id]["t_end"] = time.time()
    _save(reg)


def set_interpreted(job_id: str, val: bool = True) -> None:
    reg = _load()
    if job_id in reg:
        reg[job_id]["interpreted"] = bool(val); _save(reg)


def set_analyzed(job_id: str, val: bool = True) -> None:
    reg = _load()
    if job_id in reg:
        reg[job_id]["analyzed"] = bool(val); _save(reg)


def reconcile() -> dict:
    """
    Refresh statuses from sentinels + PID liveness. A job whose STATUS is RUNNING but
    whose PID is dead and which has no COMPLETE sentinel is FAILED (crashed/killed).
    """
    reg = _load()
    for jid, rec in reg.items():
        d = JOBS / jid
        st = (d / "STATUS").read_text(encoding="utf-8").strip() if (d / "STATUS").exists() else rec.get("status","UNKNOWN")
        if st == "RUNNING":
            if (d / "COMPLETE").exists():
                st = "DONE"
            else:
                # Disk evidence wins. A job may have been registered without a usable
                # PID (e.g. backfilled for an already-running sweep); in that case a
                # dead PID is NOT evidence of failure.
                rd = rec.get("run_dir")
                if rd:
                    out = REPO / "runs_topo" / rd
                    if (out / "trajectory.npz").exists() and (out / "curves.csv").exists():
                        st = "DONE"
                        (d / "COMPLETE").write_text("ok", encoding="utf-8")
                        (d / "STATUS").write_text("DONE", encoding="utf-8")
                pid = int(rec.get("pid") or 0)
                if st == "RUNNING" and pid and not pid_alive(pid):
                    st = "FAILED"
        rec["status"] = st
    _save(reg)
    return reg


def summary() -> str:
    reg = reconcile()
    n = len(reg)
    done_an = sum(1 for r in reg.values() if r["status"] in ("DONE","DONE_ANALYZED") and r.get("analyzed"))
    done_un = sum(1 for r in reg.values() if r["status"] in ("DONE","DONE_ANALYZED") and not r.get("analyzed"))
    uninterp = sum(1 for r in reg.values() if r["status"]=="DONE_ANALYZED" and not r.get("interpreted"))
    run     = sum(1 for r in reg.values() if r["status"]=="RUNNING")
    fail    = sum(1 for r in reg.values() if r["status"]=="FAILED")
    lines=[f"total={n}  RUNNING={run}  DONE_ANALYZED={done_an}  pending-analysis={done_un}  "
           f"awaiting-INTERPRETATION={uninterp}  FAILED={fail}"]
    if run:
        lines.append("  still RUNNING: " + ", ".join(j for j,r in reg.items() if r["status"]=="RUNNING")[:400])
    if done_un:
        lines.append("  awaiting ANALYSIS: " + ", ".join(j for j,r in reg.items() if r["status"]=="DONE" and not r.get("analyzed"))[:400])
    lines.append("IDLE -- nothing running, nothing to analyze or interpret"
                 if (run==0 and done_un==0 and uninterp==0)
                 else "WORK REMAINS")
    return "\n".join(lines)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0]=="summary": print(summary())
    elif a[0]=="register": register(a[1], a[2], int(a[3]), a[4] if len(a)>4 else "train")
    elif a[0]=="mark": mark(a[1], a[2])
    elif a[0]=="analyzed": set_analyzed(a[1], a[2].lower()!="false" if len(a)>2 else True)
    elif a[0]=="rundir":
        reg=_load(); reg.setdefault(a[1],{})["run_dir"]=a[2]; _save(reg)
    elif a[0]=="interpreted": set_interpreted(a[1], a[2].lower()!="false" if len(a)>2 else True)
    elif a[0]=="json": print(json.dumps(reconcile(), indent=2))
