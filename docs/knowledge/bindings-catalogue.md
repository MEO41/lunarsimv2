---
id: bindings-catalogue
title: Bindings catalogue
type: component
status: current
updated: 2026-11-15
summary: Every binding that exists, per slot, with its file and what it does; the two scenario files.
---

# Bindings catalogue

| Slot | Registry key | File (`src/lunarsim/...`) | What it does |
| --- | --- | --- | --- |
| start | `fixed` | `slots/start/fixed.py` | Same start state every reset |
| start | `terminal_box` | `slots/start/terminal_box.py` | Uniform ranges: altitude 5–50 m (footpad plane), vertical speed −6–0 m/s, each horizontal component ±2 m/s, tilt 0–10° in any direction. `set_fixed_start_states` / `clear_fixed_start_states` for held-out sets |
| plant | `rigid_box` | `slots/plant/rigid_box.py` | Box stand-in (milestone 0) |
| plant | `physx_site` | `slots/plant/physx_site.py`, base `physx_root_body_plant.py` | Lander articulation ([[lander-articulation-model]]); wrench at the vehicle c.g. through the permanent wrench composer each substep; mass and inertia updated in PhysX once per control step; contact sensors on legs and body; reports height above ground from `SurfaceGeometry` and the surface-probe contact (probe tips of legs 2-4 at or below the ground lookup) |
| surface | `flat` | `slots/surface/flat.py` | Ground plane, friction 0.4, restitution 0; site = env grid origin |
| surface | `procedural` | `slots/surface/procedural.py` | Random height field through `TerrainImporter` (±0.15 m, 0.5 m cells). **Smoke-tested only**; no height lookup, so height above ground is measured from z = 0 |
| surface | `measured_terrain` | `slots/surface/measured_terrain.py` | Apollo 17 height map plus craters and rocks; one target per environment per reset; exact height lookup ([[measured-terrain-surface]]) |
| sensors | `none` | `slots/sensors/none.py` | No sensors |
| faults | `none` | `slots/faults/none.py` | No faults |
| nav | `truth` | `slots/navigation/truth.py` | Copies `TruthState` into `NavEstimate` (16 values, including the probe contact light), zero covariance |
| actuators | `none` | `slots/actuators/none.py` | Zero wrench |
| actuators | `engine_rcs_pulse` | `slots/actuators/engine_rcs_pulse.py`, `propulsion_models.py` | `DescentEngineModel` (throttle range, slew, lag, 0.5 s straight-line thrust tail-off after shutdown, propellant) and `ReactionControlPulseModel` (16 jets, minimum pulse, valve delay); engine starts at hover throttle; gimbal ignored |
| metrics | `none` | `slots/metrics/none.py` | Never ends an episode |
| metrics | `touchdown` | `slots/metrics/touchdown.py` | See [[metrics-and-run-records]] |
| agent | `zero`, `attitude_hold`, `hover_hold`, `vertical_descent`, `zem_zev` | `agents/classical/` | See [[classical-agents]] |

## Scenario files (`configs/scenarios/`)

- `m0_null_loop.yaml`: start fixed, plant rigid_box, surface flat, agent zero, metrics none;
  1,024 envs, spacing 4 m, episode 5 s.
- `s1_terminal_truth.yaml`: start terminal_box, plant physx_site, surface flat, nav truth, agent
  zem_zev, actuators engine_rcs_pulse, metrics touchdown, lander apollo_lunar_module; 64 envs,
  spacing 30 m, episode 90 s.
- `apollo17_terminal_truth.yaml`: as S1 with surface `apollo17_taurus_littrow`
  (`measured_terrain`); spacing 1 m, first spawn 40 m up; needs the downloaded height maps.

Not yet built from the design list: start `orbit_elements`, `pdi_state`, `from_log`; surface
`bekker`, plume; all real sensors; faults; EKF navigations; RL agents. The design's
`dem_tile` is covered by `measured_terrain` for one site.

- part of: [[slot-loop-architecture]]
- uses: [[lander-preset-apollo-lm]]
