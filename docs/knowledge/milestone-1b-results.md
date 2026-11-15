---
id: milestone-1b-results
title: Milestone 1b results, PPO against ZEM/ZEV
type: result
status: current
updated: 2026-11-15
summary: The M1b gate - five PPO seeds (600 iterations, RTX 5090) against ZEM/ZEV on the same 4,096 held-out starts, in S1 and flown on the EKF estimate in S2.
---

# Milestone 1b results (2026-11-15)

Training: `ppo_mlp` ([[ppo-landing-agent]]), S1 flat, 4,096 envs, seeds 0-4, stopped at
iteration 600 (user decision), on `minsky` (RTX 5090, five runs side by side, about 14 s per
iteration each). Evaluation: seed 1000 (held-out), 4,096 first episodes, identical starts for
every agent (checked by `lunarsim compare`). Runs (git-ignored): `runs/eval_*_heldout`.

| | ZEM/ZEV | PPO MLP (5 seeds) |
| --- | --- | --- |
| Success, all starts | 97.68 % | **97.95 % ± 0.44** (97.22, 97.88, 98.17, 98.22, 98.29) |
| Low-and-fast starts | 85.82 % | **88.24 %** (85-90 per seed) |
| Other starts | **100.00 %** | 99.85 % |
| Touchdown vertical speed, median | 1.60 m/s | 0.72 m/s |
| Touchdown horizontal speed, median | 0.07 m/s | 0.11 m/s |
| Propellant, median | 107 kg | 48 kg |

Flown on the EKF estimate (S2) without retraining: ZEM/ZEV 97.68 %, PPO 97.72 % ± 0.48 (low and
fast 87.13 %): the sensor gap of the S1 policy is 0.23 points.

Findings:

- **Gate passed**: five PPO seeds next to ZEM/ZEV on the same held-out starts. The env changes
  for M1b were the reward hook, the critic group and the richer `EpisodeEvents` (task list).
- PPO is slightly better overall and on low-and-fast starts, slightly worse elsewhere (6 of
  4,096 per seed fail where ZEM/ZEV never does; mostly hard_horizontal).
- PPO lands with the engine running to a slow contact (0.72 m/s) instead of the probe-mode drop
  of ZEM/ZEV (1.6 m/s), and with less than half the propellant: ZEM/ZEV's time-to-go stretch
  (1.25 × shortest + 2 s) costs propellant.
- Figures: `docs/m1b/ppo_vs_zem_zev_s1_terminal_truth.png`, `docs/m1b/ppo_vs_zem_zev_s2_terminal_ekf.png`
  (`lunarsim.tools.agent_comparison.comparison_figure`).

- compares: [[classical-agents]]
- follows: [[milestone-1a-results]]
