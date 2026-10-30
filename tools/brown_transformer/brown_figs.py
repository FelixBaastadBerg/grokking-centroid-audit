"""
Appendix figure for the Brown-style modular-division Transformer confirmation cell:
validation accuracy, TwoNN, PH-dimension, and L2 over training for all confirmation
seeds. Reads RESULTS/brown_transformer_trajectories.csv; writes
paper/figs/fig_brown_traj.pdf.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONF = "#C1442E"
RELI = "#2E6DA4"
GREY = "#7A7A7A"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "grid.linewidth": 0.4,
    "lines.linewidth": 1.0, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def main():
    T = pd.read_csv(REPO / "RESULTS" / "brown_transformer_trajectories.csv")
    M = pd.read_csv(REPO / "RESULTS" / "brown_transformer_run_manifest.csv")
    M = M[M.tag == "confirm"].set_index("dir")
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.0))
    fig.subplots_adjust(wspace=0.42)
    panels = [("val_acc", "validation accuracy", axes[0], None),
              ("twonn", "TwoNN", axes[1], None),
              ("ph_dim", "PH-dimension", axes[2], None),
              ("L2", "$L_2$ (2-NN leakage)", axes[3], (-0.03, 1.03))]
    for d_, g in T.groupby("dir"):
        g = g.sort_values("step")
        grokked = bool(M.loc[d_, "grokked"])
        c = CONF if grokked else GREY
        for col, _, ax, _ in panels:
            ax.plot(g.step, g[col], color=c, alpha=0.6, lw=0.9)
        tg = int(M.loc[d_, "t_grok_step"])
        if tg > 0:
            axes[0].axvline(tg, color=c, lw=0.4, ls=":", alpha=0.4)
    for col, name, ax, ylim in panels:
        ax.set_xlabel("training step")
        ax.set_title(name, loc="left", fontweight="bold", fontsize=8)
        ax.grid(alpha=0.22)
        ax.set_xticks([0, 20000, 40000])
        ax.set_xticklabels(["0", "20k", "40k"])
        if ylim:
            ax.set_ylim(*ylim)
    n_grok = int(M.grokked.sum())
    if n_grok < len(M):
        from matplotlib.lines import Line2D
        axes[0].legend(handles=[
            Line2D([], [], color=CONF, label=f"grokked ({n_grok})"),
            Line2D([], [], color=GREY, label=f"not grokked ({len(M) - n_grok})")],
            frameon=False, loc="center right")
    else:
        axes[0].text(0.97, 0.35, f"all {n_grok} seeds grok", transform=axes[0].transAxes,
                     ha="right", fontsize=6.4, color="0.35")
    fig.savefig(REPO / "paper" / "figs" / "fig_brown_traj.pdf")
    print("wrote fig_brown_traj.pdf")


if __name__ == "__main__":
    main()
