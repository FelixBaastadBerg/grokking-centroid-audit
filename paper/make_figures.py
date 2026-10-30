"""
Regenerates every figure of the paper from the result CSVs (no training runs needed).

  fig_v6_11_intervention.pdf   Figure 1  construction schematic + alpha sweep, segmented at alpha = 1
  fig_v6_11_attribution.pdf    Figure 2  endpoint recombination cells, interaction, factor orders
  fig_v6_11_brown.pdf          Figure 3  modular-division Transformer trajectories
  fig_v6_11_mechanism.pdf      Figure 4  support schematic, two-term MST energy on real clouds
  fig_v6_11_gride.pdf          supplementary: GRIDE order response
  fig_v6_11_synth.pdf          supplementary: closed-form check on 288 synthetic mixtures
  fig_v6_11_leakage.pdf        supplementary: leakage along the intervention
  fig_v6_11_within.pdf         supplementary: within-class change across the corpus

File names keep the v6_11 tag of the paper source they were made for.

Every population and threshold shown here is read from the result files, never hard-coded.
Three conventions are fixed once and used everywhere:

  grokking rule    final test accuracy = mean over the last five checkpoints >= 0.9,
                   taken from RESULTS/v3_t11_run_table.csv
  containment      the shaded band marks alpha >= the smallest sweep point at which L_2 = 0
                   in EVERY seed, which is the condition the invariance proposition needs
  alpha segments   alpha = 1 is the recorded converged representation, so the sweep is read as
                   two counterfactual segments 0 -> 1 and 1 -> 8 rather than as one 0 -> 8 fold;
                   a fold is always the ratio of the seed medians at the two named alphas

Run from the repository root:
    python paper/make_figures.py                 # all figures
    python paper/make_figures.py 1 2 brown 3     # the four figures of the paper
"""
from __future__ import annotations
import os, sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.transforms as mtransforms

R = REPO / "RESULTS"
R4 = REPO / "RESULTS_v6_4"
R5 = REPO / "RESULTS_v6_5"
R8 = REPO / "RESULTS_v6_8"
OUT = REPO / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# Figure 1's alpha axis is linear below this value and logarithmic above it. 0.75 is a swept
# value and sits between the 0.5 and 1 ticks, which keeps those two labels visibly apart.
LINTHRESH = 0.75

CONF = "#C1442E"      # PH-dimension and other global-support statistics
CONF2 = "#E08A2E"
RELI = "#2E6DA4"      # TwoNN
RELI2 = "#3FA69B"
GREY = "#7A7A7A"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "grid.linewidth": 0.4,
    "lines.linewidth": 1.3, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


# ------------------------------------------------------------------ shared populations
def run_table():
    """The authoritative corpus classification: 96 MLP runs + 3 random-label controls."""
    t = pd.read_csv(R / "v3_t11_run_table.csv")
    t["is_random"] = t.random_labels.astype(bool)
    t["is_grokked"] = t.grokked.astype(bool) & ~t.is_random
    t["is_memoriser"] = ~t.grokked.astype(bool) & ~t.is_random
    return t


def containment_alpha(S, col="L2"):
    """Smallest swept alpha at which the leakage is zero in EVERY seed."""
    p = S.pivot_table(index="alpha", columns="seed", values=col)
    ok = [a for a in p.index if (p.loc[a].dropna() == 0).all()]
    return min(ok) if ok else np.nan


# ------------------------------------------------------------------ Fig 1
def schematic(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    cols = [CONF, RELI, CONF2]
    base = np.array([[0.50, 0.86], [0.14, 0.20], [0.86, 0.28]])
    grand = base.mean(0)
    off = np.array([[-0.09, 0.08], [0.10, 0.04], [0.01, -0.10], [-0.05, -0.03]]) * 0.85

    def place(p, x0):
        return np.column_stack([x0 + 0.42 * p[:, 0], 0.20 + 0.56 * p[:, 1]])

    for alpha, x0, ttl in ((0.30, 0.03, r"$\alpha = 0.3$"), (1.00, 0.55, r"$\alpha = 1$")):
        cen = grand + alpha * (base - grand)
        C = place(cen, x0)
        for k in range(3):
            pts = place(off + cen[k], x0)
            ax.add_patch(Circle(C[k], 0.055, facecolor=cols[k], alpha=0.14,
                                edgecolor="none", zorder=1))
            ax.plot(pts[:, 0], pts[:, 1], "o", ms=2.9, color=cols[k], zorder=3)
            ax.plot([C[k, 0]], [C[k, 1]], "x", ms=4.2, mew=1.2, color="0.15", zorder=4)
            if k == 1:
                ax.plot(pts[[0, 1], 0], pts[[0, 1], 1], "-", color="0.1", lw=1.4, zorder=5)
        ax.text(x0 + 0.21, 0.91, ttl, ha="center", fontsize=7.2)
    ax.annotate("", xy=(0.535, 0.54), xytext=(0.465, 0.54),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.0))
    ax.text(0.50, 0.60, "centroids\nseparate", ha="center", fontsize=6.2, color="0.25",
            linespacing=0.95)
    ax.text(0.5, 0.02, r"$\times$ centroid;  solid: one within-class distance,"
                       "\n" r"the same in both up to one global scale", ha="center",
            fontsize=5.4, color="0.15", linespacing=1.1)
    ax.set_title("a  the intervention", loc="left", fontweight="bold", fontsize=7.6)


def fig1():
    """
    The sweep is a sensitivity analysis, not a trajectory, so alpha = 1 -- the recorded
    converged representation -- is drawn as the reference state and the response is reported
    as two segments, 0 -> 1 and 1 -> 8, rather than as one end-to-end fold.
    """
    S = pd.read_csv(R4 / "fullgrid_sweep.csv")
    S = S[S.pipeline == "raw"]
    a_contain = containment_alpha(S)
    show = [("ph_dim", CONF, "PH-dimension"), ("twonn", RELI, "TwoNN"),
            ("L2", GREY, r"leakage $L_2$")]

    fig = plt.figure(figsize=(7.2, 2.18))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.22, 1, 1, 1], wspace=0.52)
    schematic(fig.add_subplot(gs[0, 0]))

    for j, (e, c, nm) in enumerate(show):
        ax = fig.add_subplot(gs[0, j + 1])
        for _, g in S.groupby("seed"):
            g = g.sort_values("alpha")
            ax.plot(g.alpha, g[e], "-", color=c, alpha=0.40, lw=0.9)
        med = S.groupby("alpha")[e].median()
        ax.plot(med.index, med.values, "-o", color="0.1", lw=1.4, ms=2.6, zorder=5)
        ax.axvspan(a_contain, 8.0, color=RELI, alpha=0.07, zorder=0)
        # alpha = 1 is the recorded representation: a solid rule, not a hairline dotted one
        ax.axvline(1.0, color="0.15", lw=1.1, zorder=6)
        ax.set_xlabel(r"centroid scale $\alpha$")
        # the join sits at 0.75, between the two labelled ticks, so that "0.5" and "1" are far
        # enough apart to be read as two numbers rather than as "0.51"
        ax.set_xscale("symlog", linthresh=LINTHRESH)
        ax.set_xticks([0, 0.5, 1, 8]); ax.set_xticklabels(["0", "0.5", "1", "8"])
        for lbl in ax.get_xticklabels():
            if lbl.get_text() == "1":
                lbl.set_fontweight("bold")
        # mark the join rather than letting the spacing imply one uniform scale
        # the break straddles the spine rather than sitting in the tick-label row, so it cannot
        # crowd the "0.5" and "1" labels it separates
        for dy in (-0.022, 0.016):
            ax.plot([LINTHRESH], [dy], transform=ax.get_xaxis_transform(), marker=(2, 0, 60),
                    ms=5.2, mew=1.1, color="0.25", clip_on=False, zorder=8)
        ax.axvline(LINTHRESH, color="0.75", lw=0.7, ls=(0, (1, 2)), zorder=1)
        ax.grid(alpha=0.22)
        # the segment folds go in the title: any in-panel placement either clips against the
        # axes or collides with the neighbouring panel's title at in-document scale
        letter = "bcd"[j]
        if e == "ph_dim":
            sub = (f"($0\\!\\to\\!1$: {med[0.0] / med[1.0]:.1f}$\\times$;  "
                   f"$1\\!\\to\\!8$: {med[1.0] / med[8.0]:.1f}$\\times$)")
            ax.set_ylabel("estimator value", fontsize=7.5)
        elif e == "twonn":
            sub = (f"($0\\!\\to\\!1$: {med[0.0] / med[1.0]:.2f}$\\times$;  "
                   f"$1\\!\\to\\!8$: {med[1.0] / med[8.0]:.2f}$\\times$)")
        else:
            sub = f"($= 0$ from $\\alpha = {a_contain:g}$, all seeds)"
            ax.set_ylabel("fraction of points", fontsize=7.5)
        ax.set_title(f"{letter}  {nm}\n{sub}", loc="left", fontweight="bold", fontsize=7.1)
        ax.margins(y=0.20)
    # the two conventions a reader needs to decode the panels, stated once under the figure
    fig.text(0.60, -0.11,
             r"vertical rule: $\alpha = 1$, the recorded converged representation"
             "      "
             r"$\alpha$ axis: linear below $0.75$, logarithmic above; the break marks the join",
             ha="center", va="top", fontsize=6.4, color="0.25")
    fig.savefig(OUT / "fig_v6_11_intervention.pdf")
    plt.close(fig)
    m = S.groupby("alpha")[["ph_dim", "twonn"]].median()
    print(f"fig_v6_11_intervention ok   (containment band from alpha = {a_contain}; "
          f"PH {m.ph_dim[0.0]:.2f}/{m.ph_dim[1.0]:.2f}/{m.ph_dim[8.0]:.2f}, "
          f"TwoNN {m.twonn[0.0]:.2f}/{m.twonn[1.0]:.2f}/{m.twonn[8.0]:.2f} at alpha 0/1/8)")


# ------------------------------------------------------------------ Fig 2
def _cellgrid(ax, vals, title, cmap, unit_fmt="{:.1f}"):
    """One 2x2 of raw estimator values, drawn as a labelled heat cell block."""
    M = np.array([[vals[(0, 0)], vals[(0, 1)]],
                  [vals[(1, 0)], vals[(1, 1)]]])
    ax.imshow(M, cmap=cmap, aspect="auto", alpha=0.30,
              vmin=np.nanmin(M) - 0.15 * np.ptp(M), vmax=np.nanmax(M) + 0.15 * np.ptp(M))
    for i in range(2):
        for j in range(2):
            corner = (i == j)
            ax.text(j, i, unit_fmt.format(M[i, j]), ha="center", va="center",
                    fontsize=10.5, fontweight="bold" if corner else "normal", color="0.1")
    # the diagonal cells are the network's own recorded states
    for i in (0, 1):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="0.1",
                                   lw=1.6, zorder=5))
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["$R_0$", "$R_1$"], fontsize=8.8)
    ax.set_yticklabels(["$C_0$", "$C_1$"], fontsize=8.8)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)


def fig2():
    """
    The raw endpoint evidence first, then the order dependence and the interaction. The
    two 2x2 blocks are the dominant object, because they are what the allocation is computed
    from. The four-segment partition is an appendix sensitivity result and is not drawn here.
    """
    E = pd.read_csv(R8 / "endpoint_interaction.csv")

    fig = plt.figure(figsize=(7.2, 2.42))
    # the raw 2x2 blocks are the evidential entry point, so they carry more of the width
    gs = fig.add_gridspec(2, 3, width_ratios=[0.86, 0.82, 1.63], height_ratios=[1, 1],
                          wspace=0.40, hspace=0.72)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.14)

    # ---- (a, b) the two raw 2x2 blocks, medians over the nine seeds
    for k, (est, nm, cmap) in enumerate((("ph_dim", "PH-dimension", "Reds"),
                                         ("twonn", "TwoNN", "Blues"))):
        g = E[E.estimator == est]
        vals = {(0, 0): g.f00.median(), (1, 0): g.f10.median(),
                (0, 1): g.f01.median(), (1, 1): g.f11.median()}
        ax = fig.add_subplot(gs[k, 0])
        _cellgrid(ax, vals, f"{'ab'[k]}  {nm}: raw $2\\times2$", cmap)
        if k == 1:
            ax.set_xlabel("residual state", fontsize=7.5)
        ax.set_ylabel("centroid state", fontsize=7.5)

        # ---- the four derived quantities in raw units, same row
        axd = fig.add_subplot(gs[k, 1])
        lab = [r"$\Delta$", r"$c_{\rm first}$", r"$c_{\rm second}$", r"$I$"]
        med = [g.diag.median(), g.centroid_first_raw.median(),
               g.centroid_second_raw.median(), g.interaction_raw.median()]
        lo = [g.diag.min(), g.centroid_first_raw.min(),
              g.centroid_second_raw.min(), g.interaction_raw.min()]
        hi = [g.diag.max(), g.centroid_first_raw.max(),
              g.centroid_second_raw.max(), g.interaction_raw.max()]
        ys = np.arange(4)[::-1]
        c = CONF if est == "ph_dim" else RELI
        cols = ["0.25", c, c, "#7A1E12" if est == "ph_dim" else "#1F4E79"]
        axd.barh(ys, med, color=cols, height=0.62, zorder=3)
        axd.errorbar(med, ys, xerr=[np.array(med) - lo, np.array(hi) - np.array(med)],
                     fmt="none", ecolor="0.25", elinewidth=0.8, capsize=1.8, zorder=4)
        for y, m_, h_, l_ in zip(ys, med, hi, lo):
            x = (h_ + 1.7) if m_ >= 0 else (l_ - 1.7)
            axd.text(x, y, f"{m_:.1f}", va="center",
                     ha="left" if m_ >= 0 else "right", fontsize=7.2)
        axd.axvline(0, color="0.4", lw=0.8)
        axd.set_yticks(ys); axd.set_yticklabels(lab, fontsize=8)
        axd.set_xlim(min(0, min(lo)) - 7, max(hi) + 8)
        axd.grid(alpha=0.20, axis="x")
        axd.set_title(f"{'cd'[k]}  raw units", loc="left", fontweight="bold", fontsize=8.5)
        if k == 1:
            axd.set_xlabel("estimator units", fontsize=7.5)

    # ---- (e) the symmetric endpoint allocation, and how far the two factor orders sit from it
    ax = fig.add_subplot(gs[:, 2])
    rows = []
    for est, nm, c in (("ph_dim", "PH-dimension", CONF), ("twonn", "TwoNN", RELI)):
        g = E[E.estimator == est]
        rows.append((nm, c,
                     100 * g.centroid_first_share.median(),
                     100 * g.centroid_second_share.median(),
                     100 * g.symmetric_share.median(),
                     100 * g.symmetric_share.values))
    # one row per estimator. Captions sit at the right margin in one column, so no caption can
    # land on another group's markers however close the two orders are.
    ys = [1.05, 0.20]
    for y, (nm, c, f_, s_, sym, per) in zip(ys, rows):
        # the order interval: the whole point is that it is wide
        ax.plot([min(f_, s_), max(f_, s_)], [y, y], "-", color=c, lw=8, alpha=0.28,
                solid_capstyle="butt", zorder=2)
        for x_ in (f_, s_):
            ax.plot([x_, x_], [y - 0.11, y + 0.11], "-", color=c, lw=1.8, zorder=4)
        lo_x, hi_x = (f_, s_) if f_ <= s_ else (s_, f_)
        lo_l, hi_l = (("1st", "2nd") if f_ <= s_ else ("2nd", "1st"))
        ax.text(lo_x - 3, y + 0.21, lo_l, ha="right", va="center", fontsize=6.8, color=c)
        ax.text(hi_x + 3, y + 0.21, hi_l, ha="left", va="center", fontsize=6.8, color=c)
        ax.plot(per, np.full(len(per), y - 0.21), "o", ms=2.6, color=c, alpha=0.45, zorder=3)
        ax.plot([sym], [y], "o", ms=7.0, mfc="white", mec=c, mew=1.8, zorder=6)
        ax.text(-118, y + 0.05, nm, ha="left", va="center", fontsize=8,
                fontweight="bold", color=c)
        # captions live outside the data area: x in axes fraction, y in data units, so a
        # caption can never land on a marker however the shares fall
        lab_tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.02, y, f"symmetric {sym:.0f}%", transform=lab_tr, ha="left",
                va="center", fontsize=7.0, color="0.15")

    for x in (0, 100):
        ax.axvline(x, color="0.6", lw=0.8, ls=":")
    ax.axvline(50, color="0.8", lw=0.8, ls="--")
    ax.set_xlim(-122, 150)
    ax.set_ylim(-0.30, 1.55)
    ax.set_yticks([])
    ax.set_xlabel("share assigned to the centroid factor (%)")
    ax.set_title("e  the symmetric allocation, and the two factor orders", loc="left",
                 fontweight="bold", fontsize=8.5)
    ax.grid(alpha=0.20, axis="x")
    ax.text(0.5, -0.30, "circle: symmetric allocation    small dots: individual seeds    "
            "1st/2nd: centroids advanced first or second",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.0, color="0.4")
    fig.savefig(OUT / "fig_v6_11_attribution.pdf")
    plt.close(fig)
    print("fig_v6_11_attribution ok")


# ------------------------------------------------------------------ Fig 3 (Transformer)
def fig_brown():
    """
    The interval reversal, made visible: the same runs, with the two decomposed intervals drawn
    as brackets so a reader sees that PH-dimension rises then falls while TwoNN does the reverse.
    """
    T = pd.read_csv(R / "brown_transformer_trajectories.csv")
    # the grokked flag lives in the run manifest, not the trajectory file
    M = pd.read_csv(R / "brown_transformer_run_manifest.csv")
    grok = dict(zip(M.seed, M.grokked.astype(bool)))
    PL = pd.read_csv(R / "brown_transformer_plateau.csv")
    PL = PL[(PL.frame == "unaligned") & (PL.estimator == "ph_dim")]
    plateau = float(PL.step_plateau.median())
    final = float(PL.step_final.median())

    panels = [("val_acc", "validation accuracy", None),
              ("twonn", "TwoNN", RELI),
              ("ph_dim", "PH-dimension", CONF),
              ("L2", "$L_2$ (2-NN leakage)", GREY)]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.16))
    fig.subplots_adjust(wspace=0.34, bottom=0.44, top=0.86, left=0.055, right=0.99)

    for ax, (col, nm, c) in zip(axes, panels):
        if col not in T.columns:
            continue
        for seed, g in T.groupby("seed"):
            g = g.sort_values("step")
            ok = bool(grok.get(seed, True))
            ax.plot(g.step, g[col], lw=0.9,
                    color=(c or "#1F5C99") if ok else "0.35",
                    alpha=0.85 if ok else 1.0, zorder=3 if ok else 5,
                    ls="-" if ok else (0, (3, 1.4)))
        ax.set_xlabel("training step", fontsize=7.2, labelpad=42)
        ax.set_title(nm, loc="left", fontweight="bold", fontsize=8)
        ax.set_xlim(0, 40000)
        ax.set_xticks([0, 20000, 40000]); ax.set_xticklabels(["0", "20k", "40k"])
        ax.grid(alpha=0.22)
        # The two decomposed intervals hang BELOW the axes, in axes-fraction coordinates.
        # Drawing them inside the data area would force the leakage panel to show values under
        # zero, which would imply a negative fraction of points.
        tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for yy, x0, x1, lab, cc in ((-0.26, 0, 40000, "init $\\to$ conv.", "0.25"),
                                    (-0.50, plateau, final, "plateau $\\to$ conv.", CONF)):
            ax.annotate("", xy=(x0, yy), xytext=(x1, yy), xycoords=tr, textcoords=tr,
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle="|-|,widthA=0.3,widthB=0.3", color=cc,
                                        lw=1.35, shrinkA=0, shrinkB=0))
            ax.text((x0 + x1) / 2, yy - 0.05, lab, ha="center", va="top", transform=tr,
                    fontsize=6.9, color=cc, clip_on=False)
    # no legend: the panels are too narrow to hold one clear of the curves, and the caption
    # already says solid = the nine grokked seeds, dashed grey = the one that did not grok
    fig.savefig(OUT / "fig_v6_11_brown.pdf")
    plt.close(fig)
    print(f"fig_v6_11_brown ok   (plateau {plateau:.0f} -> {final:.0f})")

# ------------------------------------------------------------------ Fig 4 (mechanism)
def support_schematic(ax):
    """
    The visual argument, drawn on one separated labelled cloud shown twice.

    Left: the two neighbours TwoNN consumes for each point, all of which stay inside the class.
    Right: the minimum spanning tree of the same points, which needs K-1 cross-class connectors
    to span. Same points, same positions, two different supports -- that is the whole reason the
    two estimators respond differently to the intervention. Drawn from a small synthetic cloud
    rather than traced by hand, so the trees are the real ones.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import squareform, pdist
    from sklearn.neighbors import NearestNeighbors

    g = np.random.default_rng(3)
    cen = np.array([[0.20, 0.72], [0.50, 0.24], [0.83, 0.70]])
    pts, y = [], []
    for k, c in enumerate(cen):
        p = c + g.normal(size=(7, 2)) * 0.052
        pts.append(p); y += [k] * len(p)
    P = np.vstack(pts); y = np.asarray(y)
    cols = [CONF, RELI, CONF2]

    ax.set_axis_off()
    # the two copies sit side by side in one axes, so the point coordinates are reused with an
    # x offset; the y range leaves a strip at the top for the two support labels
    ax.set_xlim(-0.02, 2.10); ax.set_ylim(-0.16, 1.20)

    def draw(x0, edges, cross=None):
        for i, j in edges:
            is_x = cross is not None and (i, j) in cross
            ax.plot([x0 + P[i, 0], x0 + P[j, 0]], [P[i, 1], P[j, 1]], "-",
                    color=CONF if is_x else "0.3", lw=2.0 if is_x else 0.8,
                    zorder=3 if is_x else 2, solid_capstyle="round")
        for k in range(3):
            q = P[y == k]
            ax.plot(x0 + q[:, 0], q[:, 1], "o", ms=3.6, color=cols[k], zorder=4,
                    mec="white", mew=0.4)

    # left: TwoNN's consumed support, two nearest neighbours of each point
    nn = NearestNeighbors(n_neighbors=3).fit(P)
    _, idx = nn.kneighbors(P)
    tw = {(min(i, j), max(i, j)) for i in range(len(P)) for j in idx[i, 1:]}
    n_tw_cross = sum(1 for i, j in tw if y[i] != y[j])
    draw(0.0, sorted(tw))

    # right: the MST of the same points, with the cross-class connectors picked out
    mst = minimum_spanning_tree(squareform(pdist(P))).tocoo()
    edges = [(min(i, j), max(i, j)) for i, j in zip(mst.row, mst.col)]
    cross = {(i, j) for i, j in edges if y[i] != y[j]}
    draw(1.07, edges, cross)

    ax.plot([1.035, 1.035], [-0.10, 1.02], "-", color="0.8", lw=0.8, zorder=1)
    ax.text(0.52, 1.13, "TwoNN support", ha="center", fontsize=7.2, color="0.1",
            fontweight="bold")
    ax.text(0.52, 1.02, f"{n_tw_cross} cross-class edges", ha="center", fontsize=6.6,
            color=RELI)
    ax.text(1.59, 1.13, "MST support", ha="center", fontsize=7.2, color="0.1",
            fontweight="bold")
    ax.text(1.59, 1.02, f"{len(cross)} cross-class connectors", ha="center", fontsize=6.6,
            color=CONF)
    ax.text(1.05, -0.13, "same points, same positions", ha="center", fontsize=6.2,
            color="0.4")
    ax.set_title("a  what each support touches", loc="left", fontweight="bold", fontsize=8)


def fig3():
    """
    Panels (b) and (c) use the REAL nine-seed alpha-sweep clouds rather than synthetic
    mixtures. (b) is the measured two-term MST energy on one converged cloud at the recorded
    alpha = 1, split on the same subsamples the estimator fits. (c) is the closed form against
    the fitted estimate across the whole sweep, with the region where the separated-mixture
    assumption actually holds marked -- and the region where it fails shown rather than hidden.
    The synthetic 288-mixture validation moves to the appendix.
    """
    S = pd.read_csv(R8 / "real_mst_sweep.csv")
    M = pd.read_csv(R8 / "real_mst_summary.csv")

    fig = plt.figure(figsize=(7.2, 2.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.42, 1, 1], wspace=0.40)
    fig.subplots_adjust(bottom=0.23, top=0.82, left=0.035, right=0.985)
    support_schematic(fig.add_subplot(gs[0, 0]))

    # ---- (b) the two-term energy on the real recorded representation
    ax = fig.add_subplot(gs[0, 1])
    g = S[S.alpha == 1.0].groupby("n")[["E_total", "E_within", "E_between"]].median()
    ax.plot(g.index, g.E_total, "-o", color="0.1", ms=3.4, lw=1.5, label=r"total $E(n)$")
    ax.plot(g.index, g.E_within, "-o", color=RELI, ms=3.4, lw=1.5,
            label=r"within: $\propto n^{1-\eta/d}$")
    ax.plot(g.index, g.E_between, "--s", color=CONF, ms=3.2, lw=1.5,
            label=r"connectors: $\approx$ const")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(g.E_between.min() * 0.7, g.E_total.max() * 2.8)
    ax.set_xticks([100, 200, 350, 529]); ax.set_xticklabels(["100", "200", "350", "529"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("subsample size $n$")
    ax.set_ylabel("MST edge energy", labelpad=1, fontsize=7.6)
    ax.set_title("b  real cloud, $\\alpha = 1$", loc="left", fontweight="bold", fontsize=8)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left", handlelength=1.5,
              borderaxespad=0.2)
    ax.grid(alpha=0.22, which="both")

    # ---- (c) closed form against the fit, over the real sweep, with the valid region marked
    ax = fig.add_subplot(gs[0, 2])
    hold = M[M.cross_over_Km1 <= 1.05]      # the separated-mixture assumption, measured
    fail = M[M.cross_over_Km1 > 1.05]
    ax.scatter(fail.d_closed, fail.d_hat_fit, s=9, facecolor="none", edgecolor="0.55",
               linewidths=0.7, label="assumption fails", zorder=3)
    ax.scatter(hold.d_closed, hold.d_hat_fit, s=11, color=CONF, alpha=0.7, linewidths=0,
               label=r"$\#$cross $\leq 1.05(K{-}1)$", zorder=4)
    lim = [0.8, 18]
    ax.plot(lim, lim, ":", color="0.35", lw=1.0)
    ax.axhline(1.0, color="0.45", lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.85, 20); ax.set_ylim(0.85, 20)
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks([1, 2, 5, 10]); ax.set_yticks([1, 2, 5, 10])
    ax.set_xlabel(r"closed form, $u$ measured", labelpad=1)
    ax.set_ylabel(r"fitted $\hat d$", labelpad=1, fontsize=7.6)
    ax.set_title("c  real sweep, 9 seeds", loc="left", fontweight="bold", fontsize=8)
    ax.legend(frameon=False, fontsize=6.0, loc="lower right", handletextpad=0.3,
              borderaxespad=0.3)
    ax.grid(alpha=0.22, which="both")
    fig.savefig(OUT / "fig_v6_11_mechanism.pdf")
    plt.close(fig)
    n_hold = len(hold)
    print(f"fig_v6_11_mechanism ok   ({n_hold}/{len(M)} seed-alpha cells satisfy the "
          f"separated-mixture assumption)")


# ------------------------------------------------------- supplementary: synthetic validation
def fig_synth():
    """The 288-mixture closed-form check (supplementary figure)."""
    M = pd.read_csv(R / "v3_mixture_grid.csv").dropna(subset=["d_hat", "d_pred_closedform"])
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    fig.subplots_adjust(bottom=0.22, left=0.14, right=0.98, top=0.92)
    for dd, c in zip(sorted(M.d.unique()), [RELI, CONF2, CONF]):
        q = M[M.d == dd]
        ax.scatter(q.d_pred_closedform, q.d_hat, s=7, color=c, alpha=0.55, linewidths=0,
                   label=f"$d = {dd}$")
    lim = [0.8, max(M.d_pred_closedform.max(), M.d_hat.max()) * 1.02]
    ax.plot(lim, lim, ":", color="0.35", lw=1.0)
    ax.axhline(1.0, color="0.45", lw=0.9, ls="--")
    r2 = 1 - (((M.d_hat - M.d_pred_closedform) ** 2).sum()
              / ((M.d_hat - M.d_hat.mean()) ** 2).sum())
    ax.set_xlabel(r"closed form, $u$ measured from the same mixtures")
    ax.set_ylabel(r"measured $\hat d$", fontsize=8)
    ax.set_title(f"synthetic mixtures ($n = {len(M)}$), $R^2 = {r2:.2f}$", loc="left",
                 fontweight="bold", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left", handletextpad=0.2)
    ax.grid(alpha=0.22)
    fig.savefig(OUT / "fig_v6_11_synth.pdf")
    plt.close(fig)
    print(f"fig_v6_11_synth ok   (R2 = {r2:.3f}, {len(M)} mixtures)")

# ------------------------------------------------------------------ supplementary: GRIDE
def fig_gride():
    """
    The panel that used to be Figure 3c, now on its own so its labels can be read. A line plot of
    the departure from invariance is monotone in the quantity of interest, so it needs no colour
    scale. Two markers per class size: the counting bound, and the order at which the response is
    first measured -- different quantities, plotted differently.
    """
    W = pd.read_csv(R4 / "gride_synth_ratio_dense.csv")
    C = pd.read_csv(R4 / "gride_synth_crossover_dense.csv")

    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    fig.subplots_adjust(bottom=0.30, left=0.13, right=0.98, top=0.95)
    for mm, c in zip([8, 16, 32], [RELI, CONF2, CONF]):
        q = W[W.m == mm].groupby("n1").ratio.median()
        dep = np.abs(np.log2(np.clip(q.values, 0.04, None)))
        ax.plot(q.index, dep, "-", color=c, lw=1.5, label=f"$m = {mm}$")
        ax.plot([int(np.ceil(mm / 2))], [0], marker="^", color=c, ms=6, clip_on=False, zorder=6)
        onset = float(C[C.m == mm].n1_cross_measured.median())
        if np.isfinite(onset):
            ax.plot([onset], [0], marker="o", mfc="white", mec=c, mew=1.2, ms=5,
                    clip_on=False, zorder=6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 24])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "24"])
    ax.set_ylim(-0.15, 5.2)
    ax.set_xlabel(r"GRIDE order $n_1$")
    ax.set_ylabel(r"$|\log_2$ fold$|$", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", handlelength=1.5,
              borderaxespad=0.2)
    ax.grid(alpha=0.22)
    # below the axes, not inside them: the two marker rows sit on y = 0, where an in-axes note
    # lands on top of them
    fig.text(0.55, 0.015, r"$\blacktriangle$ counting bound $\lceil m/2 \rceil$"
                          "      " r"$\circ$ measured onset",
             ha="center", va="bottom", fontsize=7, color="0.3")
    fig.savefig(OUT / "fig_v6_11_gride.pdf")
    plt.close(fig)
    print("fig_v6_11_gride ok")


# ------------------------------------------------------------------ supplementary: leakage
def fig_leakage():
    """
    One panel. The old panel (b) pooled 441 clouds to show that leakage is not a general
    estimator-error axis; that branch is gone from the submitted appendix, because L_q is used
    here only as the condition the invariance argument needs.
    """
    S = pd.read_csv(R4 / "fullgrid_sweep.csv")
    S = S[S.pipeline == "raw"]
    a_contain = containment_alpha(S)

    fig, ax1 = plt.subplots(figsize=(4.4, 2.5))
    fig.subplots_adjust(bottom=0.22, left=0.13, right=0.98, top=0.96)
    p = S.pivot_table(index="alpha", columns="seed", values="L2")
    a = p.index.values
    for s in p.columns:
        ax1.plot(a, p[s].values, "-", color="0.55", lw=0.7, alpha=0.7)
    ax1.plot(a, p.median(axis=1).values, "-o", color="0.15", ms=3, lw=1.5,
             label=r"$L_2$, median of 9 seeds")
    ax1.plot(a, p.max(axis=1).values, "--", color="0.15", lw=1.0,
             label=r"$L_2$, worst seed")
    ax1.set_xscale("symlog", linthresh=0.1)
    ax1.set_xlabel(r"centroid scale $\alpha$")
    ax1.set_ylabel("$L_2$")
    ax1.set_ylim(-0.05, 1.12)
    ax1.axvspan(a_contain, a.max(), color=RELI, alpha=0.09, zorder=0)
    ax1.text(a_contain * 1.10, 0.52, f"$L_2 = 0$ in every\nseed ($\\alpha \\geq {a_contain:g}$)",
             fontsize=7, color=RELI)
    ax1.legend(frameon=False, fontsize=7, loc="upper right", handlelength=1.7)
    ax1.grid(alpha=0.22)
    fig.savefig(OUT / "fig_v6_11_leakage.pdf")
    plt.close(fig)
    print(f"fig_v6_11_leakage ok   (all-seed containment at alpha = {a_contain})")


# ------------------------------------------------------------------ supplementary: within-class
def fig_within():
    """
    One panel. The narrow-width conditional-TwoNN panel went to the supplement with the
    appendix subsection that discussed it, so the figure now shows only the corpus comparison
    the surviving claim rests on.
    """
    W = pd.read_csv(R / "v3_within_series.csv")
    W = W[W.layer == "last_hidden"]
    t = run_table().set_index("dir")
    # the corpus population only: the separately trained narrow-width cells are not part of it
    W = W[W.dir.isin(t.index)]

    fig, ax1 = plt.subplots(figsize=(4.4, 2.5))
    fig.subplots_adjust(bottom=0.22, left=0.16, right=0.98, top=0.90)
    groups = {"grokked": (CONF, []), "memoriser": (GREY, []), "random labels": ("0.15", [])}
    for d_, g in W.groupby("dir"):
        g = g.sort_values("epoch")
        r = t.loc[d_]
        key = ("random labels" if bool(r.random_labels)
               else "grokked" if bool(r.grokked) else "memoriser")
        v = g.within_eff_rank.values / max(g.within_eff_rank.iloc[0], 1e-9)
        tt = np.linspace(0, 1, len(v))
        groups[key][1].append(np.interp(np.linspace(0, 1, 40), tt, v))
    for key, (c, curves) in groups.items():
        if not curves:
            continue
        A = np.stack(curves)
        ax1.fill_between(np.linspace(0, 1, 40), np.percentile(A, 25, 0),
                         np.percentile(A, 75, 0), color=c, alpha=0.18, linewidths=0)
        ax1.plot(np.linspace(0, 1, 40), np.median(A, 0), color=c, lw=1.6,
                 label=f"{key} ($n$={len(curves)})")
    ax1.set_xlabel("training progress (fraction of run)")
    ax1.set_ylabel("within-class eff. rank / initial", fontsize=7.6)
    ax1.set_title("width 200: compression is fitting-generic", loc="left",
                  fontweight="bold", fontsize=8)
    ax1.legend(frameon=False, fontsize=7)
    ax1.grid(alpha=0.22)
    fig.savefig(OUT / "fig_v6_11_within.pdf")
    plt.close(fig)
    counts = {k: len(v[1]) for k, v in groups.items()}
    print(f"fig_v6_11_within ok   {counts}")


# fig1/fig2/fig_brown/fig3 are Figures 1-4 of the paper; the rest are supplementary figures.
ALL = {"1": fig1, "2": fig2, "3": fig3, "brown": fig_brown, "gride": fig_gride, "synth": fig_synth,
       "leak": fig_leakage, "within": fig_within}

if __name__ == "__main__":
    which = sys.argv[1:] or list(ALL)
    for k in which:
        ALL[k]()
