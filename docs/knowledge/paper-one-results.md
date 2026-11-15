---
id: paper-one-results
title: Paper 1 results, sensor faults (S3 fault grid)
type: result
status: draft
updated: 2026-11-15
summary: First pass of paper 1 - ZEM/ZEV with ekf and ekf_fdi, and PPO with MLP, GRU and attention encoders (600 iterations) on the held-out fault grid; the memory encoders were still learning, so they continue to 1,500 iterations.
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

| Fault / sensor | ZEM/ZEV `ekf` | ZEM/ZEV `ekf_fdi` | PPO MLP (5) | PPO GRU (5) | PPO attention (2) |
|---|---|---|---|---|---|
| bias / Doppler | 46.7 | 94.9 | 97.2 | 87.6 | 90.8 |
| bias / radar | 96.5 | 74.9 | 87.3 | 83.7 | 88.2 |
| delay spike / Doppler | 90.2 | 98.0 | 97.8 | 90.5 | 89.8 |
| delay spike / radar | 98.8 | 98.8 | 97.8 | 92.2 | 89.8 |
| drift / Doppler | 42.0 | 97.6 | 97.6 | 88.5 | 92.7 |
| drift / radar | 89.8 | 97.6 | 98.1 | 88.3 | 91.4 |
| dropout burst / Doppler | 98.5 | 98.5 | 98.2 | 90.3 | 85.2 |
| dropout burst / radar | 97.7 | 97.7 | 98.2 | 91.4 | 92.1 |
| outlier spike / Doppler | 77.3 | 98.0 | 98.2 | 92.4 | 90.4 |
| outlier spike / radar | 96.1 | 96.1 | 96.3 | 91.7 | 87.8 |
| scale / Doppler | 81.6 | 96.9 | 96.6 | 89.0 | 93.5 |
| scale / radar | 96.1 | 88.2 | 95.8 | 87.4 | 91.0 |
| stuck / Doppler | 4.7 | 98.4 | 98.4 | 90.3 | 92.5 |
| stuck / radar | 45.5 | 98.8 | 99.2 | 88.2 | 90.0 |
| total loss / Doppler | 96.5 | 95.3 | 96.9 | 86.6 | 92.2 |
| total loss / radar | 99.2 | 99.2 | 99.0 | 89.1 | 90.4 |
| **all** | **78.6** | **95.6** | **97.0** | **89.2** | **90.5** |

Reference: ZEM/ZEV without faults (S2, same starts) 97.7 %. MLP per seed: 97.46, 97.66, 95.21,
97.63, 97.27. GRU: 96.19, 90.72, 84.01, 91.89, 83.23. Attention: 88.84, 92.14 (seeds 2-4 stopped
with CUDA out of memory, five attention trainings need about 55 GB; rerun two at a time).

## What it shows so far

- Fault detection is the big step: `ekf_fdi` lifts ZEM/ZEV from 78.6 to 95.6 %.
- The MLP with the filter's health flags reaches 97.0 %, close to the no-fault reference, and
  recovers part of the one case where `ekf_fdi` hurts ZEM/ZEV (radar bias: the gate rejects the
  altimeter, 74.9 % against 96.5 % with plain `ekf`; MLP 87.3 %).
- The GRU and attention policies are about 89-90 % on every fault type, also on easy ones: they
  had not finished learning. Training success (mean over seeds, last 20 iterations before):

| Iteration | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|
| MLP | 35 | 96 | 96 | 96 | 96 | 96.5 |
| GRU | 3 | 53 | 84 | 86 | 89 | 88.6 |
| attention | 0.5 | 2 | 44 | 87 | 92 | 90.5 |

## Next

User decision 2026-11-15: continue every GRU and attention run from `model_600.pt` to 1,500
iterations (`scripts/minsky/queue_paper_one_continue.sh`, runs `ppo_<kind>_s3_seed<s>_to1500`,
grid runs `grid_ppo_<kind>_seed<s>_to1500`); the MLP stays at 600. Results are then reported by
training budget.

- part of: [[milestone-roadmap-and-status]]
- uses: [[faults-and-detection]], [[ppo-landing-agent]]
