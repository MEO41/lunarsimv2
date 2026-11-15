---
id: milestone-roadmap-and-status
title: Milestone roadmap and status
type: milestone
status: current
updated: 2026-11-15
summary: The eight milestones of the design, which gates have passed, and what comes next.
---

# Milestone roadmap and status

| Milestone | Goal | Status |
| --- | --- | --- |
| M0 Skeleton | Empty slot loop runs many envs | **Gate passed** 2026-11-14: 1,024 envs step without error, hook order tested |
| M1a Lander and classical S1 | Lander, physics validation, ZEM/ZEV on S1 | **Gate passed** 2026-11-14, see [[milestone-1a-results]] and `M1A_REPORT.md` |
| M1b RL on S1 | `ppo_mlp` with rsl_rl, 5 seeds, against ZEM/ZEV | **Opened 2026-11-15** by the user. Agent, reward and training built ([[ppo-landing-agent]]); 5-seed training on `minsky` |
| M2 Sensors and EKF | IMU, altimeter, Doppler, LiDAR; EKF; S2 | **Gate test passes** 2026-11-15 (NEES 9.78 on recorded truth, navigation error logged; [[sensors-and-ekf]]). `lidar_cheap` built. Still to do: S2 PPO policy vs S1 policy (queued on `minsky`) |
| M3 Faults and history agents | Fault schedules, fault-detecting EKF, GRU and attention agents; S3, S4; paper 1 | **Started 2026-11-15**: fault schedule and `ekf_fdi` built and tested ([[faults-and-detection]]) |
| M4 Real terrain, hazards and agents | `dem_tile` surface from LROC NAC DTMs, hazard maps, site pick; SAC, Dreamer, TD-MPC2; S6, S9; paper 2 | Terrain part pulled forward and **built** 2026-11-15 ([[measured-terrain-surface]]); hazard maps, site pick and agent adapters not started |
| M5 Plume and footpads | Plume forces, Bekker footpad sinkage, dust; S8 | Not started |
| M6 Orbit, phases and TRN | float64 orbit, frame handoff, camera TRN; S5, S7 | Not started |
| M7 ROS 2 bridge | S10 | Not started |

Each gate must pass before the next milestone builds on it. Every new binding needs its unit
test, the scenario smoke test, the truth-leak check and a speed log.

## What happens next (as of 2026-11-15)

1. Done 2026-11-15: real Apollo 17 terrain with craters and rocks, LM-accurate lander, 15th
   observation value ([[measured-terrain-surface]], [[terrain-results]]). Milestone 4 work done
   before M1b at the user's request; the slot interface is unchanged apart from the new
   `surface_geometry` message and the 15th value.
2. The user moves to an RTX 5090 workstation ([[workstation-setup-and-machines]]).
3. M1b only when the user says so ([[pending-user-decisions]]).

- follows: [[project-goal-and-design-documents]]
- built: [[measured-terrain-surface]]
- history: [[session-history]]
