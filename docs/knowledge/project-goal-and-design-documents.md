---
id: project-goal-and-design-documents
title: Project goal and design documents
type: design
status: current
updated: 2026-11-14
summary: What LunarSim v2 is for, which documents are the authority, the ten scenarios and the six goals.
---

# Project goal and design documents

**LunarSim v2** is a modular lunar-landing research simulator on Isaac Lab, written from scratch,
for guidance, navigation and control research (sensors, faults, estimation, agents) and the papers
that come from it. Package name `lunarsim`; repository https://github.com/MEO41/lunarsimv2
(private, BSD-3-Clause). Author of the design: Muhammed Emin Onay.

## Authority

- `docs/design/LunarSim v2 from scratch architecture and build plan.md` (and `.pdf`): **the main
  authority**. Where it disagrees with the next document, it wins (user decision).
- `docs/design/LunarSim Scenario Bindings.md` (and `.pdf`): the bindings of every scenario.
- `PLAN.md`: the build plan that the user approved with "Proceed" at the start.
- Deviations from the design are listed in [[decisions-log]] and in `M1A_REPORT.md` section 7.

## Key design choices (from the architecture document)

- Isaac Lab `DirectRLEnv` subclass; its hooks map one to one onto the slot loop ([[slot-loop-architecture]]).
- torch tensors on the GPU, batch dimension first; float32 at the site, float64 for the orbit plant later.
- Hydra scenario files, one per scenario, with command-line overrides.
- rsl_rl first for RL (milestone 1b), skrl and adapters later.
- Parametric lander; first preset from public Apollo LM data ([[lander-preset-apollo-lm]]).
- Terrain: "Procedural height fields first; public DEM tiles (LROC NAC DTM) later". The real
  DEM surface (`dem_tile`) belongs to milestone 4 in the design; see [[terrain-integration-plan]].
- One root seed and one `torch.Generator` per slot.

## The ten scenarios (design names; files get descriptive names)

| Scenario | Short description |
| --- | --- |
| S1 | Terminal touchdown with truth state: PPO against ZEM/ZEV. **Built** (`s1_terminal_truth.yaml`). |
| S2 | S1 with IMU, altimeter, velocimeter and an EKF |
| S3 | Sensor failures: fault schedule, EKF with fault check; PPO with MLP, GRU, attention (paper 1) |
| S4 | Late fusion: multi-rate sensors, delay and jitter |
| S5 | Terrain-relative navigation descent with camera |
| S6 | Hazard avoidance: flash LiDAR, hazard map, site pick and divert |
| S7 | Full mission from orbit (float64 then PhysX) |
| S8 | Plume and regolith, footpads, dust |
| S9 | Model-based against model-free agents (paper 2) |
| S10 | ROS 2 bridge, one env |

## The six goals

| Goal | Meaning |
| --- | --- |
| G1 | Swappable modules: a slot change is one line in a scenario file |
| G2 | Correct physics: every physics part matches a known answer |
| G3 | No truth leaks: only `nav: truth`, metrics and the critic read `TruthState` |
| G4 | Known speed per scenario and GPU ([[speed-benchmarks]]) |
| G5 | Research ready: the first study runs end to end |
| G6 | Reproducible: every paper number can be rebuilt from a run manifest |

- governs: [[milestone-roadmap-and-status]]
- governs: [[slot-loop-architecture]]
