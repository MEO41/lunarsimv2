---
id: paper-one-results
title: Paper 1 results, sensor faults (S3 fault grid)
type: result
status: open
updated: 2026-11-16
summary: Paper 1 on the held-out fault grid - ZEM/ZEV (ekf 78.6 %, ekf_fdi 95.6 %), PPO MLP 97.0 % at 600 iterations, attention 94.7 % and GRU 90.6 % at 1,500; with health flags in the observation, memory gives no benefit.
---

# Paper 1 results (first pass, 2026-11-15)

Setup: scenario `s3_faults` ([[faults-and-detection]]), one fault per landing on the radar
altimeter or the Doppler velocimeter. Trainings: 5 seeds per encoder, 4,096 landers, 600
iterations, five runs side by side under CUDA MPS on `minsky` ([[workstation-setup-and-machines]]).
Evaluation: the held-out fault grid (`fault_grid=true`, seed 1000, 4,096 landings per run).
Every run is in `results/` with its code commit ([[rules-from-the-user]]): MLP runs on
`542574f`, GRU and attention runs and all grid evaluations on `cbd0937` (that change touched
only scripts and notes).

## Success on the fault grid (%, mean over seeds)

| Fault / sensor | ZEM/ZEV `ekf` | ZEM/ZEV `ekf_fdi` | PPO MLP (5) | PPO GRU (5) | PPO attention (5) |
|---|---|---|---|---|---|
| bias / Doppler | 46.7 | 94.9 | 97.2 | 87.6 | 89.1 |
| bias / radar | 96.5 | 74.9 | 87.3 | 83.7 | 87.5 |
| delay spike / Doppler | 90.2 | 98.0 | 97.8 | 90.5 | 88.9 |
| delay spike / radar | 98.8 | 98.8 | 97.8 | 92.2 | 90.4 |
| drift / Doppler | 42.0 | 97.6 | 97.6 | 88.5 | 90.5 |
| drift / radar | 89.8 | 97.6 | 98.1 | 88.3 | 90.8 |
| dropout burst / Doppler | 98.5 | 98.5 | 98.2 | 90.3 | 87.3 |
| dropout burst / radar | 97.7 | 97.7 | 98.2 | 91.4 | 91.3 |
| outlier spike / Doppler | 77.3 | 98.0 | 98.2 | 92.4 | 90.6 |
| outlier spike / radar | 96.1 | 96.1 | 96.3 | 91.7 | 89.0 |
| scale / Doppler | 81.6 | 96.9 | 96.6 | 89.0 | 91.8 |
| scale / radar | 96.1 | 88.2 | 95.8 | 87.4 | 90.7 |
| stuck / Doppler | 4.7 | 98.4 | 98.4 | 90.3 | 90.0 |
| stuck / radar | 45.5 | 98.8 | 99.2 | 88.2 | 90.0 |
| total loss / Doppler | 96.5 | 95.3 | 96.9 | 86.6 | 89.8 |
| total loss / radar | 99.2 | 99.2 | 99.0 | 89.1 | 90.0 |
| **all** | **78.6** | **95.6** | **97.0** | **89.2** | **89.9** |

Reference: ZEM/ZEV without faults (S2, same starts) 97.7 %. MLP per seed: 97.46, 97.66, 95.21,
97.63, 97.27. GRU: 96.19, 90.72, 84.01, 91.89, 83.23. Attention: 88.84, 92.14, 88.50, 92.68, 87.16
(seeds 2-4 first stopped with CUDA out of memory, five attention trainings need about 55 GB;
rerun two at a time on `0e2f9d3`).

## What it shows so far

- Fault detection is the big step: `ekf_fdi` lifts ZEM/ZEV from 78.6 to 95.6 %.
- The MLP with the filter's health flags reaches 97.0 %, close to the no-fault reference, and
  recovers part of the one case where `ekf_fdi` hurts ZEM/ZEV (radar bias: the gate rejects the
  altimeter, 74.9 % against 96.5 % with plain `ekf`; MLP 87.3 %).
- The GRU and attention policies are about 87-92 % on every fault type, also on easy ones: they
  had not finished learning. Training success (mean over seeds, last 20 iterations before):

| Iteration | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|
| MLP | 35 | 96 | 96 | 96 | 96 | 96.5 |
| GRU | 3 | 53 | 84 | 86 | 89 | 88.6 |
| attention | 0.5 | 2 | 44 | 87 | 92 | 90.5 |

## Second step: GRU and attention continued to 1,500 iterations (2026-11-16)

Each GRU and attention run continued from its `model_600.pt` (`train_ppo.py resume_from=`,
commit `f3e2119`), runs `ppo_<kind>_s3_seed<s>_to1500`, then flew the same fault grid.

| Agent | Iterations | Fault-grid success per seed (%) | Mean ± SD |
|---|---|---|---|
| ZEM/ZEV `ekf` | - | 78.64 | 78.6 |
| ZEM/ZEV `ekf_fdi` | - | 95.58 | 95.6 |
| PPO MLP | 600 | 97.46, 97.66, 95.21, 97.63, 97.27 | **97.0 ± 1.0** |
| PPO GRU | 600 | 96.19, 90.72, 84.01, 91.89, 83.23 | 89.2 ± 5.5 |
| PPO GRU | 1,500 | 95.85, 90.99, 96.00, 89.21, 80.71 | 90.6 ± 6.3 |
| PPO attention | 600 | 88.84, 92.14, 88.50, 92.68, 87.16 | 89.9 ± 2.4 |
| PPO attention | 1,500 | 93.85, 91.06, 96.04, 96.92, 95.46 | **94.7 ± 2.3** |

By fault type (%, mean over 5 seeds):

| Fault / sensor | MLP 600 | GRU 1,500 | attention 1,500 | ZEM/ZEV `ekf_fdi` |
|---|---|---|---|---|
| bias / Doppler | 97.2 | 89.4 | 95.1 | 94.9 |
| bias / radar | 87.3 | 85.6 | 90.4 | 74.9 |
| delay spike / Doppler | 97.8 | 92.2 | 95.1 | 98.0 |
| delay spike / radar | 97.8 | 92.5 | 95.5 | 98.8 |
| drift / Doppler | 97.6 | 90.0 | 95.9 | 97.6 |
| drift / radar | 98.1 | 90.2 | 95.2 | 97.6 |
| dropout burst / Doppler | 98.2 | 91.1 | 93.0 | 98.5 |
| dropout burst / radar | 98.2 | 92.1 | 95.3 | 97.7 |
| outlier spike / Doppler | 98.2 | 93.0 | 95.5 | 98.0 |
| outlier spike / radar | 96.3 | 91.4 | 93.6 | 96.1 |
| scale / Doppler | 96.6 | 91.0 | 95.4 | 96.9 |
| scale / radar | 95.8 | 90.2 | 93.1 | 88.2 |
| stuck / Doppler | 98.4 | 91.3 | 97.3 | 98.4 |
| stuck / radar | 99.2 | 90.0 | 95.4 | 98.8 |
| total loss / Doppler | 96.9 | 89.5 | 94.5 | 95.3 |
| total loss / radar | 99.0 | 89.4 | 94.7 | 99.2 |

Training success (mean over seeds, last 20 iterations before): GRU 88.6 (600), 85.7 (800),
87.4 (1,000), 88.5 (1,200), 89.0 (1,400), 89.7 (1,500); attention 89.4, 89.2, 90.1, 92.1, 93.4,
93.9. Both dip or stall for about 200 iterations after the resume (every environment restarts at
once and the recurrent states reset), so a single 1,500-iteration run could end a little higher.

What it shows:

- Attention gains 4.8 points with the longer budget and is still rising at 1,500; it is the best
  learned agent on radar bias (90.4 %) but stays below the MLP overall (94.7 against 97.0).
- The GRU barely moves (89.2 to 90.6) and its seeds spread widely (80.7 to 96.0); seed 4 is near
  80 % in training and evaluation.
- With the filter's health flags in the observation, the memoryless MLP is the best agent and
  learns in a third of the iterations. The flags already tell the policy what a history encoder
  would have to infer, so this setup does not show a benefit of memory.

## Next

- Ablation (user decision 2026-11-16): the same three encoders with `ekf_fdi` but without the
  health flags (`observation.include_navigation_health=false`), MLP 600 and GRU and attention
  1,500 iterations from scratch, 5 seeds each (`scripts/minsky/queue_paper_one_ablation.sh`;
  runs `ppo_<kind>_s3_noflags_seed<s>`, grid runs `grid_ppo_<kind>_noflags_seed<s>`). The variant
  with plain `ekf` was not chosen for now.
- Figures: success by fault type and agent; training curves by encoder.

- part of: [[milestone-roadmap-and-status]]
- uses: [[faults-and-detection]], [[ppo-landing-agent]]
