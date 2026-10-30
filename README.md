# Auditing Geometric Change in Grokking: Centroid Configuration versus Within-Class Geometry

**Felix Baastad Berg** (Imperial College London) and **Ulrik Unneberg** (Harvard SEAS)

*Proceedings of the Symmetry and Geometry in Neural Representations Workshop (NeurReps 2026),
Proceedings Track, PMLR.* PMLR link: to be added.

This repository holds the training code, the analysis code, every result CSV the paper quotes,
the scripts that regenerate the paper's figures and tables, and an automated check that
recomputes the paper's headline numbers from those CSVs.

## Summary

Geometric statistics of a grokking network's representation (intrinsic-dimension estimates,
persistent-homology summaries) change over training, but a labelled point cloud can change in two
different ways: its class centroids can move, or the geometry inside each class can change. The
paper separates the two with the exact decomposition

$$x_i = \mu_{y_i} + r_i ,$$

where $\mu_{y_i}$ is the centroid of point $i$'s class and $r_i = x_i - \mu_{y_i}$ its within-class
residual. Two tools build on it. The **centroid intervention** rescales only the centroid part,
$x_i(\alpha) = \mu + \alpha(\mu_{y_i} - \mu) + r_i$, holding every residual (and so the complete
within-class metric geometry, up to one global scale) fixed. **Checkpoint recombination** builds
hybrid clouds $\tilde x_i = \mu_{y_i,s} + r_{i,t}$ with centroids from checkpoint $s$ and
residuals from checkpoint $t$, and divides a statistic's training-time change between the two
factors with a **symmetric two-factor Shapley allocation**. On nine $\mathbb{Z}_{23}$-addition
MLPs, the finite-sample MST-scaling PH-dimension estimate is centroid-sensitive: the centroid
contribution exceeds the residual contribution in every seed (median symmetric share 90%), and
changing only the centroid configuration moves it by a factor of 3.0 from collapsed to recorded
centroids and 10.4 over the whole sweep. TwoNN is residual-dominated, and becomes numerically
invariant to centroid motion once every point's two nearest neighbours lie in its own class. The
difference follows from estimator support: TwoNN is local, while the minimum spanning tree always
keeps cross-class edges. The same contrast replicates on $S_4$ and $\mathbb{Z}_{59}$ MLPs, and a
modular-division Transformer shows that the attribution depends on the training interval.

## Citation

```bibtex
@InProceedings{berg26,
  title     = {Auditing Geometric Change in Grokking: Centroid Configuration versus Within-Class Geometry},
  author    = {Berg, Felix Baastad and Unneberg, Ulrik},
  booktitle = {Proceedings of the Symmetry and Geometry in Neural Representations Workshop (NeurReps 2026)},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026}
}
```

Volume, pages and the PMLR URL will be added once the proceedings are published.

## Quick start (no training needed)

Everything below reads only the CSVs shipped in this repository and runs in seconds on a laptop.

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu

python paper/check_headline_numbers.py   # recompute every headline number; exits 1 on mismatch
python paper/make_figures.py             # regenerate the figures into paper/figs/
python examples/synthetic_intervention.py   # the centroid intervention on a synthetic cloud
```

`paper/check_headline_numbers.py` checks 157 values (populations, the intervention sweep,
the endpoint and four-segment allocations, the replications, the mechanism and GRIDE tables, the
group-structure control) and then runs `paper/check_transformer_numbers.py`, which checks a
further 101 values and structural invariants for the modular-division Transformer. Each line prints the value printed in the paper next to the
recomputed one.

## Repository layout

```
analysis/geom.py              estimators: PH-dimension (MST scaling), TwoNN, MLE-ID, GRIDE,
                              effective rank, correlation dimension, H1 persistence, silhouette, ...
analysis/                     run loaders (lsrun.py, runlib.py, tfrun.py) and further TDA helpers
tools/train_topology.py       MLP training driver for every MLP in the paper (Z_23, S_4, Z_59)
tools/brown_transformer/      modular-division Transformer: training, analysis, plateau
                              decomposition, figures, and the configuration-selection record
tools/v6_4/ v6_5/ v6_6/ v6_8/ analysis scripts of the final revision (complete-grid
                              recombination and intervention, exactness, GRIDE, leakage audits,
                              endpoint interaction, four-segment allocation, layer audit, real-MST
                              mechanism, Z_59 population audit, table generators)
tools/camera_ready/           post hoc analysis added for the camera-ready (separation scale ratio)
tools/*.py                    earlier analysis stages (calibration, synthetic sweeps, mixture
                              model, cross-cell recombination, within-class corpus statistics, ...)
paper/check_headline_numbers.py    automated headline-number check (MLP + mechanism)
paper/check_transformer_numbers.py automated check for the Transformer numbers
paper/make_figures.py         regenerates every figure from the CSVs
paper/figs/                   figures as regenerated by make_figures.py
RESULTS/                      result CSVs of the earlier analysis stages, run manifests
RESULTS_v6_4/ RESULTS_v6_5/   result CSVs of the final revision rounds (folder names record the
RESULTS_v6_6/ RESULTS_v6_8/   revision in which each analysis was added; all are used by the paper)
RESULTS_camera_ready/         per-seed separation-scale ratio (Appendix D.2)
examples/                     minimal synthetic demonstration
*.py (top level)              model, loss and optimiser code adapted from Prieto et al. (2025);
                              see LICENSE-THIRD-PARTY
requirements-lock.txt         exact package versions of the analysis environment
requirements.txt              the same pins without the optional TDA extras
```

All scripts resolve paths relative to the repository root and are run from it.

## Environment

The analysis environment is pinned in `requirements-lock.txt` (Python 3.12, CPU-only PyTorch
2.13, numpy 2.5, scipy 1.18, scikit-learn 1.9, pandas 3.0, matplotlib 3.11, ripser 0.6, gudhi
3.13). The `+cpu` PyTorch wheels need the PyTorch CPU index (`--extra-index-url` above). The
headline check, figure script and example also run with older numpy/pandas/matplotlib (tested
with Python 3.12, numpy 1.26, pandas 2.2, matplotlib 3.8). `ripser` is needed by any script
that imports `tools/linchpin.py` (most of the measurement scripts).

The modular-division Transformer additionally needs HuggingFace `transformers` (GPT-2
backbone). Its version was not recorded in the training metadata; see "Known gaps" below.

## Reproducing the figures and tables

Every figure and table can be regenerated from the shipped CSVs. The "producing script" column
gives the command that recomputes the CSV itself; those commands need the training runs (see
"Full pipeline"), except where marked *CSV-only*.

| Paper item | Result file(s) | Producing script | Rendered by |
|---|---|---|---|
| Fig. 1, Sec. 3: centroid intervention sweep | `RESULTS_v6_4/fullgrid_sweep.csv` | `tools/v6_4/fullgrid.py` | `paper/make_figures.py 1` |
| Fig. 2, Sec. 4: endpoint recombination, interaction, Shapley shares | `RESULTS_v6_8/endpoint_interaction.csv` (from `RESULTS_v6_4/fullgrid_recombine.csv`) | `tools/v6_8/attribution_v6_8.py` (*CSV-only*), after `tools/v6_4/fullgrid.py` | `paper/make_figures.py 2` |
| Fig. 3, Sec. 4: Transformer trajectories and plateau decomposition | `RESULTS/brown_transformer_trajectories.csv`, `brown_transformer_plateau.csv` | `tools/brown_transformer/brown_analyze.py --phase traj`, `brown_plateau.py` | `paper/make_figures.py brown` |
| Fig. 4, Sec. 5: MST support on real clouds | `RESULTS_v6_8/real_mst_sweep.csv`, `real_mst_summary.csv` | `tools/v6_8/real_mst.py` | `paper/make_figures.py 3` |
| Table 1, App. C.3: intervention audits | `RESULTS/robust_c1_armA.csv`, `robust_c1_armB.csv`, `robust_size_armA.csv`, `b5_splits.csv`, `RESULTS_v6_6/splits_leakage_summary.csv` | `tools/robustness_c1.py`, `tools/robustness_samplesize.py`, `tools/b5_splits.py`, `tools/v6_6/leakage_audit.py` | (inline table) |
| Table 2, App. D.1: symmetric shares per statistic | `RESULTS_v6_4/fullgrid_summary.csv`, `fullgrid_shapley.csv`, `fullgrid_vs_400.csv` | `tools/v6_4/fullgrid_analyse.py` (*CSV-only*) | (inline table); phase-resolved shares: `tools/v6_8/make_tables.py` writes `paper/tables_v6_8/phase_shares.tex` |
| Table 3, App. D.2: endpoint decomposition per seed | `RESULTS_v6_8/endpoint_interaction.csv` | `tools/v6_8/attribution_v6_8.py` (*CSV-only*) | (inline table) |
| Table 4, App. D.2: four-segment allocation | `RESULTS_v6_8/checkpoint_path.csv`, `checkpoint_path_summary.csv` | `tools/v6_8/attribution_v6_8.py` (*CSV-only*) | (inline table) |
| App. D.2: separation scale ratio | `RESULTS_camera_ready/scale_ratio.csv` | `tools/camera_ready/scale_ratio.py --acts ...` (see Known gaps) | (inline text) |
| App. D.3: first hidden layer | `RESULTS_v6_8/layer_audit.csv`, `layer_audit_summary.csv` | `tools/v6_8/layer_audit.py` | (inline text) |
| Table 5, App. D.4: cross-cell shares | `RESULTS/v3_crosscell.csv`, `RESULTS_v6_8/z59_by_train_fraction.csv` | `tools/v3_t21_crosscell.py`, then `tools/v6_8/attribution_v6_8.py` | (inline table) |
| App. B.2: populations, Z_59 audit | `RESULTS_v6_8/z59_run_audit.csv`, `z59_population_summary.csv`; `RESULTS/_inventory.csv` (116-network manifest), `RESULTS/v3_t11_run_table.csv`; `RESULTS_v6_4/v6_4_*.csv` (claim/run ledger) | `tools/v6_8/z59_audit.py`, `tools/extract_logs.py`, `tools/v3_t11_analyze.py`, `tools/v6_4/build_ledger.py` | (inline tables) |
| Table 6, App. D.5: leakage on replication cells | `RESULTS_v6_6/crosscell_leakage.csv`, `crosscell_leakage_summary.csv` | `tools/v6_6/leakage_audit.py` | (inline table) |
| Tables 7-8, App. D.6: Transformer per-seed and plateau decompositions | `RESULTS/brown_transformer_*.csv`, `RESULTS/brown_parts/`, `RESULTS_v6_4/brown_exact_attribution.csv`, `brown_second_config.csv`, `RESULTS_v6_8/brown_hybrid_leakage.csv` | `tools/brown_transformer/brown_analyze.py`, `brown_plateau.py`, `brown_sweep_leakage.py`; `tools/v6_4/brown_exact.py`, `tools/v6_4/brown_second_config.py`; `tools/v6_8/brown_hybrid_leakage.py` | (inline tables) |
| App. D.7: alignment, exactness, preprocessing | `RESULTS/v2_align_*.csv`, `RESULTS_v6_4/exactness_summary.csv`, `exactness_checks.csv`, `fullgrid_recombine.csv` | `tools/v2_align_layers.py`, `tools/v6_4/exactness_checks.py`, `tools/v6_4/fullgrid.py` | (inline text) |
| App. C.1: complete estimator battery | `RESULTS_v6_5/intervention_battery_summary.csv` | `tools/v6_5/intervention_battery.py` | `tools/v6_8/make_tables.py` writes `paper/tables_v6_8/intervention_battery.tex` |
| App. C.2: calibration and positive controls | `RESULTS/interv_gate.csv`, `RESULTS/e8_summary.csv` | `tools/intervention.py`, `tools/e8_calibration.py` | (inline text) |
| App. E.2: support leakage $L_q$ | `RESULTS_v6_4/near_containment_exact.csv`, `near_containment_bands.csv`, `RESULTS/v2_leakage_*.csv` | `tools/v6_4/near_containment.py` (*CSV-only*), `tools/v2_leakage_alpha.py`, `tools/v2_leakage_axis.py` | `paper/make_figures.py leak` (supplementary figure) |
| App. E.3, Table 9: mixture model of PH-dimension | `RESULTS/v3_mixture_grid.csv`, `v3_mixture_summary.txt`, `RESULTS_v6_8/real_mst_summary.csv`, `RESULTS/v3_stratified.csv` | `tools/v3_t13_mixture.py` (synthetic, no runs needed), `tools/v6_8/real_mst.py`, `tools/v3_t32_stratified.py` | `paper/make_figures.py synth` (supplementary figure) |
| App. E.4, Table 10: GRIDE forced-leakage order and crossover | `RESULTS/b1_gride_*.csv`, `RESULTS_v6_4/gride_*.csv` | `tools/b1_gride.py`, `tools/v6_4/gride_threshold.py` | `paper/make_figures.py gride` (supplementary figure) |
| App. F, Table 11: separation, centroid rank, group structure | `RESULTS/v3_dissociate.csv`, `v3_dissociate_summary.csv` | `tools/v3_t12_dissociate.py` (synthetic, no runs needed) | (inline table) |
| Supplement: within-class change across the corpus | `RESULTS/v3_within_series.csv`, `RESULTS/v3_t11_summary.txt` | `tools/v3_t11_within.py`, `tools/v3_t11_analyze.py` | `paper/make_figures.py within` (supplementary figure) |

The paper uses four figures (`fig_v6_11_intervention`, `_attribution`, `_brown`, `_mechanism`);
`paper/make_figures.py` also writes four supplementary figures (`_gride`, `_synth`, `_leakage`,
`_within`). Figure file names keep the `v6_11` tag of the paper source they were made for.

Re-running the *CSV-only* steps (`tools/v6_4/fullgrid_analyse.py`,
`tools/v6_4/near_containment.py`, `tools/v6_8/attribution_v6_8.py`, `tools/v6_8/make_tables.py`)
on the shipped inputs reproduces the shipped outputs exactly.

## Full pipeline (training and measurement)

Training runs, checkpoints and activations are not distributed (they total several GB). The
measurement scripts expect MLP runs under `runs_topo/<run-id>/` and Transformer runs under
`runs_brown/<run-id>/`, exactly as written by the training scripts below. Run identifiers encode
the configuration, e.g. `arch-mlp__task-add__p-23__seed-0__lr-0.01__wd-0.03__tf-0.8`;
`RESULTS/_inventory.csv` lists every run directory used.

**Primary experiment, Z_23 addition MLP** (two hidden layers of width 200, no biases, AdamW
lr 1e-2, weight decay 0.03, eps 1e-25, StableMax cross-entropy, full batch, 50,000 epochs, a
checkpoint every 200 epochs = 251 checkpoints):

```bash
for S in 0 1 2 3 4; do
  python tools/train_topology.py --task add --p 23 --seed $S --lr 0.01 --weight_decay 0.03 \
      --train_fraction 0.8 --hidden 200 --num_epochs 50000 --save_every 200
done
# seeds 10-13 carry the tag ss777 in their run directory, i.e. they were trained with a fixed
# split seed of 777 (see Known gaps):
for S in 10 11 12 13; do
  python tools/train_topology.py --task add --p 23 --seed $S --lr 0.01 --weight_decay 0.03 \
      --train_fraction 0.8 --hidden 200 --num_epochs 50000 --save_every 200 \
      --split_seed 777 --tag ss777
done
```

**Replication cells.** S_4 composition: `--task perm --sym_n 4 --seed {0..4} --train_fraction 0.8`
with the same optimiser settings and 50,000 epochs (251 checkpoints). Z_59 addition: `--task add --p 59 --tag hb`, seeds 0 and 1 at
train fractions 0.4, 0.6, 0.8 and seeds 2-4 at 0.8, 60,000 epochs (`tools/v3_overnight.py` is the
queue that trained seeds 2-4; 301 checkpoints per run). Random-label controls add
`--random_labels`.

**Modular-division Transformer** (HuggingFace GPT-2 backbone, 2 layers, 4 heads, width 128, Adam
with coupled weight decay, lr 1e-3, 40,000 steps; specification and deviations in the
docstring of `brown_train.py` and in `tools/brown_transformer/config_selection.md`):

```bash
for S in 100 101 102 103 104 105 106 107 108 109; do
  python tools/brown_transformer/brown_train.py --frac 0.15 --wd 0.001 --seed $S \
      --tag confirm --impl hf --optim adam
done
# pilots used for configuration selection: --wd {0.01,0.001} --seed {0,1} --tag pilot2
python tools/brown_transformer/brown_analyze.py --phase traj
python tools/brown_transformer/brown_analyze.py --phase five
python tools/brown_transformer/brown_analyze.py --phase collect
python tools/brown_transformer/brown_plateau.py
python tools/brown_transformer/brown_sweep_leakage.py
```

**Measurement, primary experiment and mechanism** (each script writes the CSVs named in the
table above):

```bash
python tools/extract_logs.py               # run inventory (RESULTS/_inventory.csv)
python tools/v6_4/fullgrid.py              # complete-grid recombination + intervention sweep
python tools/v6_4/fullgrid_analyse.py      # Shapley shares, phases, 400-point comparison
python tools/v6_4/exactness_checks.py
python tools/v6_4/near_containment.py
python tools/v6_4/gride_threshold.py
python tools/v6_5/intervention_battery.py
python tools/v6_6/leakage_audit.py
python tools/v6_8/z59_audit.py
python tools/v6_8/attribution_v6_8.py      # endpoint interaction, four-segment allocation, Z_59
python tools/v6_8/layer_audit.py
python tools/v6_8/real_mst.py
python tools/v6_8/brown_hybrid_leakage.py
python tools/v6_4/brown_exact.py
python tools/v6_8/make_tables.py
python paper/make_figures.py
python paper/check_headline_numbers.py
```

The earlier-stage scripts in `tools/` (calibration, synthetic sweeps, cross-cell recombination,
within-class statistics) each name their output files in their docstring.

## Runtime and hardware

All training and analysis ran on CPUs; no GPU was used (the MLP driver forces CPU). Training and
analysis were run on different machines, and the exact CPU models were not recorded. Approximate
wall-clock times, from the logs shipped in `RESULTS_v6_*/*.log` (5 worker processes):

| Stage | Wall clock |
|---|---|
| `tools/v6_4/fullgrid.py` (9 seeds, recombination grid + sweep, 2 pipelines) | about 8 min |
| `tools/v6_5/intervention_battery.py` (9 seeds, full battery) | about 2 min |
| `tools/v6_4/exactness_checks.py` | about 3 min |
| `tools/v6_4/gride_threshold.py` | under 1 min |
| `tools/v6_4/brown_exact.py` (10 Transformer seeds, 97 checkpoints each) | about 45 min |
| `tools/v6_4/brown_second_config.py` | about 22 min |
| headline check, figures, CSV-only steps | seconds |

Training time per run was not logged and is therefore not reported here.

## Known gaps

* **Training runs, checkpoints and activations are not included** (size). Every number can be
  re-derived from the shipped CSVs; re-deriving the CSVs themselves needs retraining.
* **Transformer environment.** The paper states the Transformer reimplementation was pinned to
  its own environment; that environment file is not in this repository, and the `transformers`
  version is not recorded in the run metadata (only the torch version is).
* **Separation scale ratio (Appendix D.2).** `RESULTS_camera_ready/scale_ratio.csv` holds the
  per-seed values the paper quotes. They were computed post hoc from stored endpoint activations
  that are not distributed here; `tools/camera_ready/scale_ratio.py` reproduces the computation
  from any activation file in the documented format (regenerable from the runs with the
  activation code in `tools/v6_4/fullgrid.py`). The stored activations for seed 2 came from a
  different run of that seed and are excluded, as the paper states.
* **Split seed of seeds 10-13.** Their run directories carry the tag `ss777`, which in this code
  base denotes a train/test split seeded with 777 rather than with the run seed (see
  `tools/v3_t11_within.py`). The per-run `run_meta.json` that would confirm this is not
  distributed. The main-text measurements use the complete 529-point input grid and do not depend
  on the split. `analysis/lsrun.py` re-derives the split from the run seed and ignores
  `split_seed`, so for seeds 10-13 the train / held-out rows of the intervention audits
  (`tools/b5_splits.py`, `tools/v6_6/leakage_audit.py`) and the 400-point training subsample use a
  random 423/106 partition rather than the training split. The camera-ready paper (App. C.3, App. B.1)
  states this; the train / held-out contrast rests on seeds 0-4.
* The modular-division dataset follows Sea-Snell/grokking (cited by Brown et al., 2022); that
  repository is not vendored here.

## License

MIT, see `LICENSE`. Files adapted from Prieto et al. (2025) keep their original MIT notice in
`LICENSE-THIRD-PARTY`.
