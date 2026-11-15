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
| sensors | `terminal_suite` | `slots/sensors/terminal_suite.py`, `sensor_models.py` | IMU, radar altimeter, Doppler velocimeter, propellant gauge, probe switches; with `terminal_suite_with_lidar.yaml` also the cheap flash LiDAR ([[sensors-and-ekf]]) |
| faults | `schedule` | `slots/faults/fault_schedule.py` | One seeded fault per episode on altimeter or Doppler ([[faults-and-detection]]) |
| faults | `none` | `slots/faults/none.py` | No faults |
| nav | `ekf_fdi` | `slots/navigation/ekf_fdi.py` | `ekf` plus age check, chi-square gate, isolation, loss ([[faults-and-detection]]) |
| nav | `ekf` | `slots/navigation/ekf.py`, `error_state_kalman_filter.py` | 15-state error-state filter on IMU, altimeter, Doppler; never reads truth ([[sensors-and-ekf]]) |
| nav | `truth` | `slots/navigation/truth.py` | Copies `TruthState` into `NavEstimate` (16 values, including the probe contact light), zero covariance |
| actuators | `none` | `slots/actuators/none.py` | Zero wrench |
| actuators | `engine_rcs_pulse` | `slots/actuators/engine_rcs_pulse.py`, `propulsion_models.py` | `DescentEngineModel` (throttle range, slew, lag, 0.5 s straight-line thrust tail-off after shutdown, propellant) and `ReactionControlPulseModel` (16 jets, minimum pulse, valve delay); engine starts at hover throttle; gimbal ignored |
| metrics | `none` | `slots/metrics/none.py` | Never ends an episode |
| metrics | `fault_response` | `slots/metrics/fault_response.py` | Touchdown plus fault onset, detection delay, false alarms ([[faults-and-detection]]) |
| metrics | `touchdown` | `slots/metrics/touchdown.py` | See [[metrics-and-run-records]] |
| agent | `zero`, `attitude_hold`, `hover_hold`, `vertical_descent`, `zem_zev` | `agents/classical/` | See [[classical-agents]] |
| agent | `ppo_mlp` | `agents/rsl_rl/ppo_policy_agent.py` | Flies an exported PPO policy ([[ppo-landing-agent]]) |

## Scenario files (`configs/scenarios/`)

- `m0_null_loop.yaml`: start fixed, plant rigid_box, surface flat, agent zero, metrics none;
  1,024 envs, spacing 4 m, episode 5 s.
- `s1_terminal_truth.yaml`: start terminal_box, plant physx_site, surface flat, nav truth, agent
  zem_zev, actuators engine_rcs_pulse, metrics touchdown, lander apollo_lunar_module; 64 envs,
  spacing 30 m, episode 90 s.
- `s2_terminal_ekf.yaml`: S1 with `sensors: terminal_suite`, `nav: ekf`, `start.nav_error` 1 m / 0.2 m/s / 0.1°.
- `s3_faults.yaml`: S2 with `faults: schedule`, `nav: ekf_fdi`, `metrics: fault_response`; ZEM/ZEV baseline row.
- `apollo17_terminal_truth.yaml`: as S1 with surface `apollo17_taurus_littrow`
  (`measured_terrain`); spacing 1 m, first spawn 40 m up; needs the downloaded height maps.

Not yet built from the design list: start `orbit_elements`, `pdi_state`, `from_log`; surface
`bekker`, plume; cameras; faults; `ekf_fdi`, `ekf_late_fusion`; recurrent and other RL agents. The design's
`dem_tile` is covered by `measured_terrain` for one site.

- part of: [[slot-loop-architecture]]
- uses: [[lander-preset-apollo-lm]]
