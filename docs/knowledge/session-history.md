---
id: session-history
title: Session history
type: history
status: current
updated: 2026-11-17
summary: Timeline of the work so far with commit hashes, so a new session knows how the project got here.
---

# Session history

## 2026-11-14, one long session (RTX 5060 machine)

| Time | Commit | What happened |
| --- | --- | --- |
| morning | — | The user first asked to build on a fresh clone of the old LunarSim repo, then replaced that with the from-scratch brief and the two design documents. The old repo became off limits ([[rules-from-the-user]]) |
| — | — | `PLAN.md` written, open decisions settled with the user, "Proceed" given |
| 15:56 | `0d8e54a` | Repo initialised: template layout, license, plan, design docs |
| 15:58 | `aa972c5` | Core: message types, binding, bus, registry, clock, generators, frames stub |
| 16:04 | `118352b` | Milestone 0: slot loop in `LanderEnv`, null bindings, Hydra loader (CI file written, push blocked) |
| 16:11 | `fe3d9dc` | Apollo LM preset with sources, loader, USD builder |
| 16:18 | `7428a9d` | Plant, actuators, `terminal_box`, physics validation tests |
| 16:30 | `ee903c4` | Contact study: PGS with 16 iterations ([[physx-contact-settings]]) |
| 16:33 | `965dc70` | Touchdown metric, episode records, run manifest |
| 16:36–16:45 | `e2d542a`, `b3a9504`, `478f9b4` | Controller steps 1–4 up to ZEM/ZEV, evaluation script, S1 scenario |
| 17:17 | `95174f5` | Procedural surface, figures, `lunarsim` command, benchmark, failure grouping |
| 17:21 | `009762c` | Slot-loop overview figure and lander render in the README |
| 17:30 | `3594bdc` | **M1a gate**: report and final figures ([[milestone-1a-results]]) |
| 17:50–17:55 | `1a43788`, `2d6861b` | 3-D glide-slope figure; result figures in the README |
| evening | — | The user pointed to `apollo_lander_modeling` ([[apollo-lander-modeling-folder]]). First plan proposed using the NASA mesh as a visual; the user rejected it and set the terrain requirements. Two research agents gathered LM geometry and surface statistics ([[apollo-lm-geometry-research]], [[lunar-surface-statistics-research]]). Revised [[terrain-integration-plan]] written; waiting for "Proceed" |
| — | — | The user changed models and effort levels several times during the session |
| 23:12 | `2769e96` | Workstation setup script, pinned requirements, installation check (for the RTX 5090 move) |
| 23:14 | `7e35240` | CPU-test workflow pushed after the user added the `workflow` token scope; first GitHub run passed (1 min 10 s) |
| late | this commit | Knowledge graph in `docs/knowledge/` ([[knowledge-graph-tooling]]) |

## 2026-11-15 (RTX 5060 machine)

| Commit | What happened |
| --- | --- |
| — | User: "proceed"; yes to the 15th observation value |
| — | Iyy checked on the page image (13,867); RCS arm error found (R-567); download addresses and the LM coordinates (Haase et al. 2019) confirmed |
| — | First GPU run showed hover drift: Isaac Lab applies forces at the body's own c.g.; fixed in the plant |
| — | Terrain package, `measured_terrain` surface, 15th value, terrain metrics, tests; one terrain test fixed (drop height) |
| — | GPU runner: shm clean-up, launch watchdog, unbuffered child output |
| `d26d7a4` | All of the above ([[measured-terrain-surface]]) |
| — | Evaluations, 3 seeds × 4,096 on flat ground and on Apollo 17 ([[terrain-results]]) |
| — | User asked for a video of 5 successes and 5 failures (body contacts first) and for 1,000 drop tests from 5 m |
| — | First drop trial showed long slides from 10°: footpad friction was 0.2 (material combine mode); fixed and tested; evaluations rerun |
| — | Drop test on a steep part of the map ([[drop-test-results]]); landing video script |
| 2026-11-16 | User: "reach the M7", no training over 20 min. M4 hazard part (S6), M5 (S8, L2 stopped), M6 (S7, S5; [[orbit-to-touchdown-mission]]) built without learning; M7 bridge and S10 ([[software-in-the-loop-bridge]]) |
| 2026-11-17 | S7 mission video and README GIF; M5 L2 built as torch grains (user: "It has to be efficiently run on the gpu") and the L1 vs L2 check; grain video for a visual check (user: "I want a video to visual confirm"), then a realistic render (user: "Video look only") |
| 2026-11-17 | User: "Lets go back to the terrain navigation and landing site selection part", chose "S6: close the LiDAR gap". Measured LiDAR noise (94 % safe) vs specification (31 %); the true-map 83 % found to be the agent's landing accuracy ([[hazard-detection-and-avoidance]]) |

## Next

Move to the RTX 5090 ([[workstation-setup-and-machines]]), then the answers in
[[pending-user-decisions]].

- leads to: [[milestone-roadmap-and-status]]
