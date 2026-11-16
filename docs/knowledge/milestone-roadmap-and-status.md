---
id: milestone-roadmap-and-status
title: Milestone roadmap and status
type: milestone
status: current
updated: 2026-11-16
summary: The eight milestones of the design, which gates have passed, and what comes next.
---

# Milestone roadmap and status

| Milestone | Goal | Status |
| --- | --- | --- |
| M0 Skeleton | Empty slot loop runs many envs | **Gate passed** 2026-11-14: 1,024 envs step without error, hook order tested |
| M1a Lander and classical S1 | Lander, physics validation, ZEM/ZEV on S1 | **Gate passed** 2026-11-14, see [[milestone-1a-results]] and `M1A_REPORT.md` |
| M1b RL on S1 | `ppo_mlp` with rsl_rl, 5 seeds, against ZEM/ZEV | **Gate passed** 2026-11-15: PPO 97.95 % ± 0.44 vs ZEM/ZEV 97.68 % on the same held-out starts ([[milestone-1b-results]]) |
| M2 Sensors and EKF | IMU, altimeter, Doppler, LiDAR; EKF; S2 | **Gate test passes** 2026-11-15 (NEES 9.78 on recorded truth, navigation error logged; [[sensors-and-ekf]]). `lidar_cheap` built. S2 PPO policies 98.25 % vs S1 policies flown in S2 97.72 % and ZEM/ZEV 97.68 % (held-out seed 1000, `results/eval_ppo_s2policy_*`); comparison figure not made yet |
| M3 Faults and history agents | Fault schedules, fault-detecting EKF, GRU and attention agents; S3, S4; paper 1. Gate: G5 (paper 1 results table filled) and the paper 1 draft complete | **Training part done 2026-11-16, gate half passed.** Built: fault schedule, `ekf_fdi`, `ekf_late_fusion`, `faults: delay`, `fault_response`, `ppo_gru`, `ppo_attention`, S3, S4, the held-out fault grid ([[faults-and-detection]]). G5 table filled ([[paper-one-results]]): ZEM/ZEV `ekf` 78.6 %, `ekf_fdi` 95.6 %; PPO with health flags MLP 97.0 % (600 iterations), attention 94.7 %, GRU 90.6 % (1,500); without flags MLP 97.6 %, attention 95.6 %, GRU 94.9 %. **Left:** paper 1 draft and figures; plain-`ekf` memory study (planned, [[pending-user-decisions]]); optional: learned agents in S4, `ekf_fdi` radar-bias weakness |
| M4 Real terrain, hazards and agents | `dem_tile` surface from LROC NAC DTMs, hazard maps, site pick; SAC, Dreamer, TD-MPC2; S6, S9; paper 2 | Terrain part pulled forward and **built** 2026-11-15 ([[measured-terrain-surface]]); hazard maps, site pick and agent adapters not started |
| M5 Plume and footpads | Plume forces, Bekker footpad sinkage, dust; S8 | Not started |
| M6 Orbit, phases and TRN | float64 orbit, frame handoff, camera TRN; S5, S7 | Not started |
| M7 ROS 2 bridge | S10 | Not started |

Each gate must pass before the next milestone builds on it. Every new binding needs its unit
test, the scenario smoke test, the truth-leak check and a speed log.

## What happens next (as of 2026-11-16)

1. Paper 1 draft and figures from the runs in `results/` ([[paper-one-results]]); this closes
   the M3 gate together with the filled results table.
2. When `minsky` is free: the plain-`ekf` memory study ([[pending-user-decisions]] item 1).
3. Optional for paper 1: PPO in S4 (late Doppler data), a fix for the radar-bias case of
   `ekf_fdi`, the M2 comparison figure (S1 vs S2 policies).
4. Then M4 (hazard maps, site pick, SAC, Dreamer, TD-MPC2; paper 2) when the user opens it.

- follows: [[project-goal-and-design-documents]]
- built: [[measured-terrain-surface]]
- history: [[session-history]]
