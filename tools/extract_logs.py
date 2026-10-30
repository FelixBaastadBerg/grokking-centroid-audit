"""
D1 back-compat shim: materialise the 225 cluster-made `loggs/` run directories
into Windows-safe paths, with an explicit `run_meta.json` per run.

Why this exists
---------------
Runs were written on Linux with the run identity encoded in the DIRECTORY NAME,
joined by '|' (grokking_experiments.py:149). '|' is illegal in NTFS paths, so
these directories cannot be checked out or created on Windows at all.

This script reads them straight out of the git object store and writes:

    data_runs/<safe_run_id>/{metrics.csv,dimensions.csv,args.json,run_meta.json}

`run_meta.json` carries explicit fields (arch, task, p, lr, weight_decay,
train_fraction, seed, ...) so downstream analysis NEVER parses a directory name.

stdlib only -- runs before the venv exists.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "data_runs"

# characters Windows forbids in path components
UNSAFE = re.compile(r'[|:*?<>"\\/]')


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True, check=True,
    ).stdout.decode("utf-8", errors="replace")


def git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, check=True
    ).stdout


def safe_id(raw: str) -> str:
    """'|'-delimited legacy dir name -> Windows-safe id. Never lossy-collides."""
    return UNSAFE.sub("__", raw).strip().rstrip(".")


def parse_legacy_name(name: str) -> dict:
    """
    Legacy format: '<dataset>|key-value|key-value|...'
    Only non-default args were included (utils.py:94 get_specified_args), so
    absent keys mean "argparse default" -- we fill those from args.json instead.
    """
    parts = name.split("|")
    out = {"_legacy_dataset": parts[0]}
    for p in parts[1:]:
        if "-" not in p:
            continue
        k, _, v = p.partition("-")
        out[k] = v
    return out


# task inferred from the binary operation actually used
OP_TO_TASK = {
    "add_mod": "add",
    "product_mod": "mul",
    "subtract_mod": "sub",
    "divide_mod": "div",
}


def build_meta(legacy_name: str, args: dict) -> dict:
    """
    args.json is authoritative (it is json.dump(vars(args)) -- ALL args, including
    defaults). The legacy dir name is only a cross-check.
    """
    arch = "transformer" if args.get("use_transformer") else "mlp"
    dataset = args.get("dataset", "")
    if dataset == "sparse_parity":
        task = "parity"
    else:
        task = OP_TO_TASK.get(args.get("binary_operation", ""), args.get("binary_operation", "unknown"))

    return {
        # --- identity (the whole point of D1) ---
        "arch": arch,
        "task": task,
        "p": args.get("modulo"),
        "lr": args.get("lr"),
        "weight_decay": args.get("weight_decay"),
        "train_fraction": args.get("train_fraction"),
        "seed": args.get("seed"),
        # --- context needed to interpret the trajectory ---
        "num_epochs": args.get("num_epochs"),
        "log_frequency": args.get("log_frequency"),
        "loss_function": args.get("loss_function"),
        "optimizer": args.get("optimizer"),
        "batch_size": args.get("batch_size"),
        "hidden_sizes": args.get("hidden_sizes"),
        "beta2": args.get("beta2"),
        "alpha": args.get("alpha"),
        "cross_entropy_dtype": args.get("cross_entropy_dtype"),
        # --- provenance ---
        "_source": "legacy_loggs_git",
        "_legacy_dir": legacy_name,
        "_legacy_parsed": parse_legacy_name(legacy_name),
    }


def main() -> int:
    rev = sys.argv[1] if len(sys.argv) > 1 else "HEAD"

    # every blob under loggs/
    listing = git("ls-tree", "-r", "--format=%(objectname) %(path)", rev, "loggs/")
    by_dir: dict[str, dict[str, str]] = {}
    for line in listing.splitlines():
        if not line.strip():
            continue
        sha, _, path = line.partition(" ")
        d, _, fname = path.rpartition("/")
        by_dir.setdefault(d, {})[fname] = sha

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    written, skipped, collisions = 0, [], {}
    for d, files in sorted(by_dir.items()):
        legacy_name = d.split("/", 1)[1] if "/" in d else d
        need = {"metrics.csv", "dimensions.csv", "args.json"}
        if not need.issubset(files):
            skipped.append((legacy_name, f"missing {sorted(need - set(files))}"))
            continue

        args = json.loads(git_bytes("cat-file", "blob", files["args.json"]).decode("utf-8"))

        sid = safe_id(legacy_name)
        if sid in collisions and collisions[sid] != legacy_name:
            # sanitisation must never merge two distinct runs
            n = 1
            while f"{sid}__dup{n}" in collisions:
                n += 1
            sid = f"{sid}__dup{n}"
        collisions[sid] = legacy_name

        dest = OUT_ROOT / sid
        dest.mkdir(parents=True, exist_ok=True)
        for fname in ("metrics.csv", "dimensions.csv", "args.json"):
            (dest / fname).write_bytes(git_bytes("cat-file", "blob", files[fname]))

        meta = build_meta(legacy_name, args)
        (dest / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        written += 1

    print(f"wrote      {written} runs -> {OUT_ROOT}")
    print(f"skipped    {len(skipped)}")
    for name, why in skipped[:10]:
        print("   ", why, "|", name[:80])

    # quick identity census so a silent merge is visible immediately
    census: dict[tuple, int] = {}
    for dpath in OUT_ROOT.iterdir():
        if not (dpath / "run_meta.json").exists():
            continue
        m = json.loads((dpath / "run_meta.json").read_text(encoding="utf-8"))
        key = (m["arch"], m["task"], m["loss_function"], m["num_epochs"])
        census[key] = census.get(key, 0) + 1
    print("\n(arch, task, loss, num_epochs) census:")
    for k, v in sorted(census.items(), key=lambda kv: -kv[1]):
        print(f"   {v:>4}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
