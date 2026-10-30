# Brown-style Transformer experiment: configuration selection record

Written BEFORE any pilot run was started. Basis: Brown et al., arXiv:2211.13239, Figure 3
(rendered from the arXiv PDF at 260 dpi and inspected panel by panel on 2026-08-12), and
the cited implementation github.com/Sea-Snell/grokking (cloned to `external/`).

## The published 16-configuration grid (Figure 3 subplot titles)

p (training fraction) ∈ {0.15, 0.20, 0.25, 0.30} × WD (weight decay) ∈ {0, 0.0001, 0.001, 0.01}.

Published outcomes read from the figure (red curve = "does not achieve perfect validation
accuracy in the allotted compute budget of 40k training steps", per Brown's caption):

| p | WD=0 | WD=0.0001 | WD=0.001 | WD=0.01 |
|---|---|---|---|---|
| 0.15 | red (no gen.) | red (no gen.) | groks, sharp rise ≈17k | groks, sharp rise ≈11k |
| 0.20 | red (no gen.) | groks ≈22k | groks ≈8k | groks ≈5k |
| 0.25 | gradual rise to 1.0 (slow, no sharp transition) | groks ≈13k | groks ≈8k | groks ≈5k |
| 0.30 | rises early ≈5k | groks ≈8k | groks ≈5k | groks ≈3k |

## Pilot configurations selected (rule: the two published curves that most clearly reach
## high validation accuracy after delayed generalization)

1. **p = 0.15, WD = 0.01** — long memorisation plateau, sharp late transition to
   validation accuracy 1.0 at roughly 11k of 40k steps; this is also the configuration
   Brown et al. single out in their text ("ex. plot with p = 0.15, weight decay = 0.01").
2. **p = 0.15, WD = 0.001** — the longest clearly-successful delay in the grid: sharp
   transition to 1.0 at roughly 17k steps, still with a wide margin inside the 40k budget.

Configurations with comparable delay were rejected for pilot use as follows:
p = 0.20 / WD = 0.0001 groks at ≈22k in the published curve, close enough to the budget
that seed variation could produce censored runs; p = 0.25 / WD = 0 rises gradually rather
than showing the delayed sharp transition the replication is meant to reproduce; all
higher-p, higher-WD cells grok early with little delay.

## Pilot protocol (fixed in advance)

- pilot seeds 0 and 1 per configuration; 4 runs; 40,000 steps each;
- pilot runs are used ONLY to select the confirmation configuration; no recombination or
  Shapley computation is run on pilot checkpoints before the confirmation configuration
  is locked;
- deterministic confirmation-selection rule (from the v6.2 instruction file):
  1. the configuration with more pilot seeds reaching final validation accuracy ≥ 0.90;
  2. tie → higher median final validation accuracy;
  3. tie → larger median delay between train accuracy 0.99 and t_grok;
  4. tie → the configuration appearing first in Brown's published grid.
- confirmation seeds 100–109, locked before any attribution analysis.

## Pilot round 1 outcome and diagnosis (added 2026-08-12, before any attribution analysis)

Round 1 (tag `pilot`, `--impl port --optim adamw`: the Sea-Snell architecture port and
AdamW training loop) produced 0/4 grokked runs: all four reached train accuracy 1.000 by
step ~1500 and stayed at chance validation accuracy (max 0.015–0.017) through 40k steps,
with validation loss still at or near its peak at the end. All four runs are retained
under `runs_brown/*__pilot/`.

Fidelity diagnosis against Brown et al.: their Appendix A specifies the model was
"obtained from the huggingface transformers library" and trained with "The ADAM
optimizer"; only the modular-division DATASET comes from Sea-Snell's repository. Round 1
therefore deviated from Brown in two substantive ways: (i) optimiser — AdamW's decoupled
weight decay instead of Adam's coupled L2 penalty, which is amplified by the adaptive
normaliser once gradients are small and is plausibly the regularisation that drives
Brown's grokking at nominal WD 1e-4..1e-2; (ii) model — Sea-Snell's hand-written blocks
with PyTorch-default initialisation instead of a HuggingFace GPT-2 backbone (init std
0.02, no sqrt(d) embedding scaling).

Remedy (per the pre-stated fallback: diagnose implementation fidelity first): re-run the
SAME two selected configurations — still the two clearest published delayed-generalization
curves — with the corrected implementation (`--impl hf --optim adam`, tag `pilot2`),
pilot seeds 0 and 1, and apply the unchanged deterministic selection rule to round 2. No
hyperparameter differs from Brown's stated specification; nothing was selected on any
geometric outcome (no recombination or attribution has been computed on any pilot run).

## Pilot round 2 outcome and LOCKED confirmation configuration (2026-08-13)

Round 2 (tag `pilot2`, `--impl hf --optim adam`) reproduced Brown's published behaviour
in all four runs — delayed sharp transitions to validation accuracy 1.0:

| config | seed | final val (last-5 mean) | t_train99 | t_grok | delay |
|---|---|---|---|---|---|
| p 0.15, WD 0.01 | 0 | 0.9999 | 1000 | 14500 | 13500 |
| p 0.15, WD 0.01 | 1 | 1.0000 | 800 | 11600 | 10800 |
| p 0.15, WD 0.001 | 0 | 1.0000 | 300 | 17800 | 17500 |
| p 0.15, WD 0.001 | 1 | 1.0000 | 300 | 13500 | 13200 |

## Plateau-checkpoint rule for the matched decomposition (fixed 2026-08-13, before any
## plateau decomposition output was produced or inspected)

Brown et al.'s phenomenon is a TwoNN decline from the memorisation plateau to convergence.
To decompose that interval rather than initialisation-to-convergence, the plateau endpoint
is defined by an outcome-only rule that uses no representation statistic:

> the last saved checkpoint strictly before `t_grok` at which validation accuracy is at
> most 0.10 and training accuracy is at least 0.99.

The convergence endpoint is the final saved checkpoint, as everywhere else in the paper.
The rule is applied identically to every grokked confirmation seed; no per-seed choice is
made, and no checkpoint is selected for maximising TwoNN or for producing a preferred
attribution. Implemented in `brown_plateau.py::plateau_step`.

## Deterministic configuration rule (round 2)

Deterministic rule: (1) seeds ≥ 0.90 — tied 2/2 vs 2/2; (2) median final validation
accuracy — WD 0.001: 1.0000 > WD 0.01: 0.99995 → **locked confirmation configuration:
p = 0.15, weight decay = 0.001, `--impl hf --optim adam`** (rule 3 would give the same
choice: median delay 15,350 vs 12,150 steps). Confirmation seeds 100–109, tag `confirm`,
40,000 steps each, all runs kept regardless of outcome. This file is committed before any
recombination or attribution is computed on any run.
