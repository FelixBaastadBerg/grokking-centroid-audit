"""
LAYER 1 -- mechanical auto-analysis. Runs as part of the job pipeline, with no agent
alive. Given a run directory name, computes the full topology result set and writes
results/<run_id>/ plus a COMPLETE sentinel.

Everything here uses only statistics that passed the synthetic ground-truth checks:
  * ambient PH of the per-number embedding cloud (H0/H1), diameter-normalised, raw
  * per-frequency circularity WITH its null (row-permuted + iid Gaussian), because
    projecting onto a self-derived Fourier plane is partially circularising by
    construction and scores ~0.22 on random data
  * Fourier concentration and radius CV as independent (harmonic / geometric) checks

Exit code 0 on success, non-zero on failure, so the caller can set STATUS correctly.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.topo_runs import TopoRun, fourier_concentration, circularity_score
from analysis.tda import prepare_cloud, persistence_diagrams, diagram_summaries

RESULTS = REPO / "results_runs"
N_NULL = 8          # null replicates per run; cheap and enough to bound the null mean


# ---------------------------------------------------------------- helpers
def _circ_feats(P: np.ndarray, seed: int = 0) -> dict:
    Xp = prepare_cloud(P, max_points=400, seed=seed, scale="diameter")
    if Xp.shape[0] < 8:
        return {}
    d = persistence_diagrams(Xp, maxdim=1)
    s = diagram_summaries(d[1], "H1")
    r = np.linalg.norm(P - P.mean(0), axis=1)
    return {"H1_max": s["H1_max_pers"], "b1": s["H1_n_above_0.15"],
            "radius_cv": float(r.std() / r.mean()) if r.mean() > 0 else np.nan}


def _freq_planes(E: np.ndarray):
    """Project the embedding onto each Fourier frequency's own 2-D plane."""
    Ec = E - E.mean(0, keepdims=True)
    F = np.fft.rfft(Ec, axis=0)
    power = (np.abs(F) ** 2).sum(1)
    power[0] = 0.0
    tot = power.sum()
    out = {}
    if tot <= 0:
        return out
    for k in range(1, len(power)):
        u, v = np.real(F[k]), np.imag(F[k])
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-12 or nv < 1e-12:
            continue
        out[k] = (float(power[k] / tot), np.c_[Ec @ (u / nu), Ec @ (v / nv)])
    return out


def _top_plane_H1(E: np.ndarray, seed: int = 0):
    fr = _freq_planes(E)
    if not fr:
        return np.nan, np.nan, np.nan
    k, (pw, P) = max(fr.items(), key=lambda kv: kv[1][0])
    f = _circ_feats(P, seed=seed)
    return f.get("H1_max", np.nan), pw, f.get("radius_cv", np.nan)


# ---------------------------------------------------------------- main
def analyse(run_dir: Path) -> dict:
    run = TopoRun(run_dir)
    out = RESULTS / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    eps = run.epochs
    curves = run.curves.set_index("epoch")
    is_cyclic = run.meta.get("task") in ("add", "sub", "mul", "div")

    # ---- per-checkpoint trajectory of the topology ----
    rows = []
    for i, ep in enumerate(eps):
        E = run.embeddings("a", i)
        amb = _circ_feats(E, seed=0)
        row = {"epoch": int(ep),
               "amb_H1_max": amb.get("H1_max"), "amb_b1": amb.get("b1"),
               "amb_radius_cv": amb.get("radius_cv")}
        if is_cyclic:
            fc = fourier_concentration(E)
            row.update({"fourier_top5": fc["fourier_topk_frac"],
                        "fourier_neff": fc["fourier_n_eff"]})
            h1, pw, rcv = _top_plane_H1(E)
            row.update({"plane_H1": h1, "top_freq_power": pw, "plane_radius_cv": rcv})
        row.update(circularity_score(E))
        for c in ("train_acc", "test_acc", "train_loss", "test_loss", "w0_norm"):
            if int(ep) in curves.index:
                row[c] = float(curves.loc[int(ep), c])
        rows.append(row)
    per_ck = pd.DataFrame(rows)
    per_ck.to_csv(out / "per_checkpoint.csv", index=False)

    # ---- final-checkpoint per-frequency analysis WITH null ----
    summary = {"run_id": run_dir.name, **{k: run.meta.get(k) for k in
               ("arch", "task", "p", "seed", "weight_decay", "train_fraction", "num_epochs")},
               "final_test_acc": float(run.curves.test_acc.iloc[-1]),
               "final_train_acc": float(run.curves.train_acc.iloc[-1]),
               "groks": bool(run.groks), "t_grok": run.t_grok, "t_memorize": run.t_memorize}

    E = run.embeddings("a", -1)
    amb = _circ_feats(E)
    summary.update({"ambient_H1_max": amb.get("H1_max"), "ambient_b1": amb.get("b1")})

    if is_cyclic:
        real_h1, real_pw, real_cv = _top_plane_H1(E)
        rng = np.random.default_rng(0)
        null_h1, null_cv = [], []
        for j in range(N_NULL):
            for variant in ("perm", "gauss"):
                X = E[rng.permutation(E.shape[0])] if variant == "perm" else \
                    rng.normal(size=E.shape) * E.std()
                h, _, cv = _top_plane_H1(X, seed=j)
                if np.isfinite(h):
                    null_h1.append(h); null_cv.append(cv)
        nh = np.array(null_h1, float)
        summary.update({
            "plane_H1_real": real_h1, "top_freq_power": real_pw, "plane_radius_cv_real": real_cv,
            "plane_H1_null_mean": float(nh.mean()) if nh.size else np.nan,
            "plane_H1_null_sd": float(nh.std()) if nh.size else np.nan,
            "plane_H1_null_p95": float(np.percentile(nh, 95)) if nh.size else np.nan,
            "plane_H1_z_vs_null": float((real_h1 - nh.mean()) / nh.std()) if nh.size and nh.std() > 0 else np.nan,
            "plane_radius_cv_null_mean": float(np.mean(null_cv)) if null_cv else np.nan,
        })
        fr = _freq_planes(E)
        if fr:
            top = sorted(fr.items(), key=lambda kv: -kv[1][0])[:5]
            pd.DataFrame([{"freq": k, "power_frac": pw, **_circ_feats(P)}
                          for k, (pw, P) in top]).to_csv(out / "freq_planes.csv", index=False)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # ---- figures ----
    try:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(per_ck.epoch, per_ck.train_acc, label="train")
        ax[0].plot(per_ck.epoch, per_ck.test_acc, label="test")
        if np.isfinite(run.t_grok):
            ax[0].axvline(run.t_grok, ls="--", c="k", lw=1, label="onset")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("accuracy"); ax[0].legend(); ax[0].set_title(run_dir.name[:38])
        ax[1].plot(per_ck.epoch, per_ck.amb_H1_max, label="ambient H1")
        if "plane_H1" in per_ck:
            ax[1].plot(per_ck.epoch, per_ck.plane_H1, label="top-frequency plane H1")
            ax[1].axhline(summary.get("plane_H1_null_mean", np.nan), ls=":", c="r", label="null mean")
        if np.isfinite(run.t_grok):
            ax[1].axvline(run.t_grok, ls="--", c="k", lw=1)
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("H1 max persistence"); ax[1].legend()
        if "fourier_neff" in per_ck:
            ax[2].plot(per_ck.epoch, per_ck.fourier_neff)
            ax[2].set_ylabel("effective # Fourier freqs")
        ax[2].set_xlabel("epoch")
        if np.isfinite(run.t_grok):
            ax[2].axvline(run.t_grok, ls="--", c="k", lw=1)
        fig.tight_layout(); fig.savefig(out / "overview.png", dpi=110); plt.close(fig)
    except Exception:
        traceback.print_exc()

    (out / "COMPLETE").write_text("ok", encoding="utf-8")
    return summary


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: analyze_job.py <run_dir_name>"); return 2
    rd = REPO / "runs_topo" / sys.argv[1]
    if not (rd / "trajectory.npz").exists():
        print(f"[analyze_job] no trajectory for {rd.name}"); return 3
    try:
        s = analyse(rd)
        print(f"[analyze_job] OK {rd.name} test={s['final_test_acc']:.3f} "
              f"amb_b1={s.get('ambient_b1')} plane_H1={s.get('plane_H1_real')} "
              f"z_vs_null={s.get('plane_H1_z_vs_null')}")
        return 0
    except Exception:
        traceback.print_exc(); return 1


if __name__ == "__main__":
    raise SystemExit(main())
