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
| M4 Real terrain, hazards and agents | `dem_tile` surface from LROC NAC DTMs, hazard maps, site pick; SAC, Dreamer, TD-MPC2; S6, S9; paper 2 | Terrain part pulled forward and **built** 2026-11-15 ([[measured-terrain-surface]]). **Hazard part built 2026-11-16** ([[hazard-detection-and-avoidance]]): S6 with pointed LiDAR, hazard map, `site_pick_divert`, `site_safety`; 31 % safe touchdowns with the LiDAR map, 83 % with the true map. Agent adapters (SAC, Dreamer, TD-MPC2), S9 and paper 2 skipped ([[skipped-learning-work]]) |
| M5 Plume and footpads | Plume forces, Bekker footpad sinkage, dust; S8. Gate: L1 vs L2 check written up | **Built 2026-11-16** except L2 ([[plume-regolith-and-dust]]): `regolith_plume_l1` (Bekker soil from TN D-6850, L1 plume), `faults: dust`, `metrics: dust`, S8 and its no-plume variant, L3 data plan. **L2 does not work in this Isaac Sim** (no particle tensor view); fix proposed, waiting for the user ([[pending-user-decisions]] item 2). S8 learned agent skipped |
| M6 Orbit, phases and TRN | float64 orbit, frame handoff, camera TRN; S5, S7 | Done without learning (2026-11-16): S7 orbit to touchdown and S5 TRN fly the Apollo 11 timeline ([[orbit-to-touchdown-mission]]); the learned terminal `ppo_gru` is skipped ([[skipped-learning-work]]) |
| M7 ROS 2 bridge | S10 | **Built 2026-11-16** ([[software-in-the-loop-bridge]]): `nav`/`agent: ros2_bridge` run the unchanged `ekf_fdi` and `ppo_gru` (ONNX) as node processes over a loopback transport; ROS 2 transport written, untested (ROS 2 not installed, [[pending-user-decisions]] item 3). Parity: exact (0.000 m over 20 S3 episodes) when the nodes compute like the simulator; with CPU nodes and the ONNX policy 5.2 m largest difference, the same as the 3.1 m that switching off compilation alone gives. The gate's tolerance waits for the user ([[pending-user-decisions]] item 4) |

Each gate must pass before the next milestone builds on it. Every new binding needs its unit
test, the scenario smoke test, the truth-leak check and a speed log.

## What happens next (as of 2026-11-16)

1. M7 on real ROS 2 once it is installed ([[pending-user-decisions]] item 3); hardware-in-the-loop later (design S10 step 3).
2. Paper 1 draft and figures from the runs in `results/` (the user writes it); this closes the
   M3 gate together with the filled results table.
3. When `minsky` is free: the plain-`ekf` memory study ([[pending-user-decisions]] item 1).
4. Waiting for the user: M5 L2 ([[pending-user-decisions]] item 2).
5. Later: the learned work skipped in M3 to M6 ([[skipped-learning-work]]).

- follows: [[project-goal-and-design-documents]]
- built: [[measured-terrain-surface]]
- history: [[session-history]]
