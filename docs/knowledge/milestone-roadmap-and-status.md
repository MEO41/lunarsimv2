---
id: milestone-roadmap-and-status
title: Milestone roadmap and status
type: milestone
status: current
updated: 2026-11-14
summary: The eight milestones of the design, which gates have passed, and what comes next.
---

# Milestone roadmap and status

| Milestone | Goal | Status |
| --- | --- | --- |
| M0 Skeleton | Empty slot loop runs many envs | **Gate passed** 2026-11-14: 1,024 envs step without error, hook order tested |
| M1a Lander and classical S1 | Lander, physics validation, ZEM/ZEV on S1 | **Gate passed** 2026-11-14, see [[milestone-1a-results]] and `M1A_REPORT.md` |
| M1b RL on S1 | `ppo_mlp` with rsl_rl, 5 seeds, against ZEM/ZEV | **Not started.** Closed until the user opens it ([[rules-from-the-user]]). Task list: `M1A_REPORT.md` section 9 |
| M2 Sensors and EKF | IMU, altimeter, Doppler, LiDAR; EKF; S2 | Not started |
| M3 Faults and history agents | Fault schedules, fault-detecting EKF, GRU and attention agents; S3, S4; paper 1 | Not started |
| M4 Real terrain, hazards and agents | `dem_tile` surface from LROC NAC DTMs, hazard maps, site pick; SAC, Dreamer, TD-MPC2; S6, S9; paper 2 | Not started; its terrain part is being pulled forward, see below |
| M5 Plume and footpads | Plume forces, Bekker footpad sinkage, dust; S8 | Not started |
| M6 Orbit, phases and TRN | float64 orbit, frame handoff, camera TRN; S5, S7 | Not started |
| M7 ROS 2 bridge | S10 | Not started |

Each gate must pass before the next milestone builds on it. Every new binding needs its unit
test, the scenario smoke test, the truth-leak check and a speed log.

## What happens next (as of 2026-11-14)

1. The user moves to an RTX 5090 workstation ([[workstation-setup-and-machines]]).
2. Real terrain and LM-accurate footpads: planned in [[terrain-integration-plan]], **waiting for
   "Proceed"** ([[pending-user-decisions]]). In the design this is milestone 4 work (the
   `dem_tile` surface), done before M1b at the user's request. The plan keeps the slot
   interface unchanged except for the proposed 15th observation value.
3. M1b afterwards, only when the user says so.

- follows: [[project-goal-and-design-documents]]
- next step: [[terrain-integration-plan]]
- history: [[session-history]]
