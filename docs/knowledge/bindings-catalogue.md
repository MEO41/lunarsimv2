---
id: bindings-catalogue
title: Bindings catalogue
type: component
status: current
updated: 2026-11-16
summary: Every binding that exists, per slot, with its file and what it does; the two scenario files.
---

# Bindings catalogue

| Slot | Registry key | File (`src/lunarsim/...`) | What it does |
| --- | --- | --- | --- |
| start | `fixed` | `slots/start/fixed.py` | Same start state every reset |
| start | `approach_gate` | `slots/start/approach_gate.py` | Apollo low-gate starts flying toward the site (S6, [[hazard-detection-and-avoidance]]) |
| start | `orbit_elements` | `slots/start/orbit_elements.py` | Circular parking orbit at the DOI point of a descent to the site (S7, [[orbit-to-touchdown-mission]]) |
| start | `from_log` | `slots/start/from_log.py` | Gate states saved by the previous mission phase; optional handoff to the SITE frame with a jump check (S7, S5) |
| start | `terminal_box` | `slots/start/terminal_box.py` | Uniform ranges: altitude 5–50 m (footpad plane), vertical speed −6–0 m/s, each horizontal component ±2 m/s, tilt 0–10° in any direction. `set_fixed_start_states` / `clear_fixed_start_states` for held-out sets |
| plant | `rigid_box` | `slots/plant/rigid_box.py` | Box stand-in (milestone 0) |
| plant | `physx_site` | `slots/plant/physx_site.py`, base `physx_root_body_plant.py` | Lander articulation ([[lander-articulation-model]]); `footpad_contact: soil` hands the footpads to a soil model (S8); wrench at the vehicle c.g. through the permanent wrench composer each substep; mass and inertia updated in PhysX once per control step; contact sensors on legs and body; reports height above ground from `SurfaceGeometry` and the surface-probe contact (probe tips of legs 2-4 at or below the ground lookup) |
| plant | `orbit_f64` | `slots/plant/orbit_float64.py` | float64 RK4 point-mass orbit in MCI or MCMF (S7) |
| plant | `local_f64` | same file | The same in the SITE frame over terrain (S5) |
| surface | `flat` | `slots/surface/flat.py` | Ground plane, friction 0.4, restitution 0; site = env grid origin |
| surface | `procedural` | `slots/surface/procedural.py` | Random height field through `TerrainImporter` (±0.15 m, 0.5 m cells). **Smoke-tested only**; no height lookup, so height above ground is measured from z = 0 |
| surface | `regolith_plume_l1` | `slots/surface/regolith_plume_surface.py`, `terrain/regolith_models.py` | Measured terrain plus Bekker footpad soil and the L1 plume; writes `regolith_state` (S8, [[plume-regolith-and-dust]]) |
| surface | `measured_terrain` | `slots/surface/measured_terrain.py` | Apollo 17 height map plus craters and rocks; one target per environment per reset; exact height lookup ([[measured-terrain-surface]]) |
| sensors | `none` | `slots/sensors/none.py` | No sensors |
| sensors | `terminal_suite` | `slots/sensors/terminal_suite.py`, `sensor_models.py`, `pointed_flash_lidar.py` | IMU, radar altimeter, Doppler velocimeter, propellant gauge, probe switches; with `terminal_suite_with_lidar.yaml` also the cheap flash LiDAR ([[sensors-and-ekf]]); with `hazard_detection_suite.yaml` the LiDAR on a pointing mount (S6); with `trn_suite.yaml` the shaded-relief TRN camera (`slots/sensors/shaded_relief_camera.py`, S5) |
| faults | `schedule` | `slots/faults/fault_schedule.py` | One seeded fault per episode on altimeter or Doppler ([[faults-and-detection]]) |
| faults | `delay` | `slots/faults/delayed_delivery.py` | Late delivery with capture stamps (S4) |
| faults | `dust` | `slots/faults/dust.py` | Plume dust degrades LiDAR and Doppler lidar (S8) |
| faults | `none` | `slots/faults/none.py` | No faults |
| nav | `ekf_fdi` | `slots/navigation/ekf_fdi.py` | `ekf` plus age check, chi-square gate, isolation, loss ([[faults-and-detection]]) |
| nav | `ekf_late_fusion` | `slots/navigation/ekf_late_fusion.py` | Applies late samples at their capture time, replays to now (S4) |
| nav | `ekf` | `slots/navigation/ekf.py`, `error_state_kalman_filter.py` | 15-state error-state filter on IMU, altimeter, Doppler; never reads truth ([[sensors-and-ekf]]) |
| nav | `hazard_map` | `slots/navigation/hazard_map_navigation.py`, `onboard_hazard_map.py` | `ekf` plus the LiDAR hazard map and site map (S6) |
| nav | `truth_with_hazard_map` | same file | True state plus the true hazard map (S6 upper limit; may read truth) |
| nav | `truth_noise` | `slots/navigation/truth_noise.py` | Truth plus a per-episode bias and white noise, optional local-vertical bias (S7 orbit phases, S5 variant) |
| nav | `trn_ekf` | `slots/navigation/terrain_relative_navigation.py` | `ekf` plus camera terrain fixes against the onboard map (S5) |
| nav | `ros2_bridge` | `slots/navigation/ros2_bridge.py`, `bridge/` | Runs the inner navigation binding (`node`) as a separate node; deadline model (S10, [[software-in-the-loop-bridge]]) |
| nav | `truth` | `slots/navigation/truth.py` | Copies `TruthState` into `NavEstimate` (16 values, including the probe contact light), zero covariance |
| actuators | `none` | `slots/actuators/none.py` | Zero wrench |
| actuators | `engine_rcs_pulse` | `slots/actuators/engine_rcs_pulse.py`, `propulsion_models.py` | `DescentEngineModel` (throttle range, slew, lag, 0.5 s straight-line thrust tail-off after shutdown, propellant) and `ReactionControlPulseModel` (16 jets, minimum pulse, valve delay); engine starts at hover throttle; gimbal ignored |
| metrics | `none` | `slots/metrics/none.py` | Never ends an episode |
| metrics | `fault_response` | `slots/metrics/fault_response.py` | Touchdown plus fault onset, detection delay, false alarms ([[faults-and-detection]]) |
| metrics | `dust` | `slots/metrics/dust.py` | Touchdown plus dust, ejecta, plume and sinkage columns (S8) |
| metrics | `phase_gate` | `slots/metrics/phase_gate.py` | Ends each lander at the phase gate (altitude, range, perilune, time; first met) and logs the gate state (S7, S5) |
| metrics | `site_safety` | `slots/metrics/site_safety.py` | Touchdown plus the true footprint safety of touched and nominal site (S6) |
| metrics | `touchdown` | `slots/metrics/touchdown.py` | See [[metrics-and-run-records]] |
| agent | `zero`, `attitude_hold`, `hover_hold`, `vertical_descent`, `zem_zev` | `agents/classical/` | See [[classical-agents]] |
| agent | `scripted_burns` | `agents/classical/powered_descent_guidance.py` | DOI burn to the 50,000 ft perilune, then attitude hold (S7) |
| agent | `quadratic_guidance` | same file | Apollo-style braking and approach guidance over the sphere with the LM throttle logic (S7, S5; [[orbit-to-touchdown-mission]]) |
| agent | `ros2_bridge` | `agents/bridge_agent.py`, `bridge/` | Runs the inner agent (`node`) as a separate node, optionally with the ONNX policy (S10) |
| agent | `site_pick_divert` | `agents/classical/site_pick_divert_agent.py` | ZEM/ZEV that picks and redesignates a safe site from the site map (S6) |
| agent | `ppo_mlp` | `agents/rsl_rl/ppo_policy_agent.py` | Flies an exported PPO policy ([[ppo-landing-agent]]) |

## Scenario files (`configs/scenarios/`)

- `m0_null_loop.yaml`: start fixed, plant rigid_box, surface flat, agent zero, metrics none;
  1,024 envs, spacing 4 m, episode 5 s.
- `s1_terminal_truth.yaml`: start terminal_box, plant physx_site, surface flat, nav truth, agent
  zem_zev, actuators engine_rcs_pulse, metrics touchdown, lander apollo_lunar_module; 64 envs,
  spacing 30 m, episode 90 s.
- `s2_terminal_ekf.yaml`: S1 with `sensors: terminal_suite`, `nav: ekf`, `start.nav_error` 1 m / 0.2 m/s / 0.1°.
- `s3_faults.yaml`: S2 with `faults: schedule`, `nav: ekf_fdi`, `metrics: fault_response`; ZEM/ZEV baseline row.
- `s4_late_fusion.yaml`: S2 with `faults: delay` (Doppler 50-200 ms) and `nav: ekf_late_fusion`.
- `s8_plume_regolith.yaml`, `s8_plume_regolith_no_plume.yaml`: S8 ([[plume-regolith-and-dust]]).
- `s6_hazard_avoidance.yaml`, `s6_hazard_avoidance_truth_map.yaml`: S6 ([[hazard-detection-and-avoidance]]).
- `apollo17_terminal_truth.yaml`: as S1 with surface `apollo17_taurus_littrow`
  (`measured_terrain`); spacing 1 m, first spawn 40 m up; needs the downloaded height maps.

Not yet built from the design list: start `orbit_elements`, `pdi_state`, `from_log`; surface
`bekker`, plume; cameras; faults; `ekf_fdi`, `ekf_late_fusion`; recurrent and other RL agents. The design's
`dem_tile` is covered by `measured_terrain` for one site.

- part of: [[slot-loop-architecture]]
- uses: [[lander-preset-apollo-lm]]
