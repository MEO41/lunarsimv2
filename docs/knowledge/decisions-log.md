---
id: decisions-log
title: Decisions log
type: decision
status: current
updated: 2026-11-16
summary: Every decision taken so far, with date and reason; the place to add new ones.
---

# Decisions log

Newest at the bottom. "User" means the user decided; the others were proposed, explained and accepted.

| Date | Decision | Why |
| --- | --- | --- |
| 2026-11-14 | Build v2 from scratch; old LunarSim repo off limits | User ([[rules-from-the-user]]) |
| 2026-11-14 | Repo MEO41/lunarsimv2 (private), package `lunarsim`, BSD-3-Clause | User, open decisions of the design |
| 2026-11-14 | Apollo LM is the first lander preset | User |
| 2026-11-14 | Pin Isaac Sim 6.0.1 (pip) + Isaac Lab 3.0.0 develop@e86463d, PhysX backend | What runs on the machine; stock tasks default to Newton |
| 2026-11-14 | Forces through the permanent wrench composer | `set_external_force_and_torque` is deprecated and only forwards to it |
| 2026-11-14 | Plant state read twice per control step (hook 3 and hook 6) | Metrics must see the state after physics; observations after resets |
| 2026-11-14 | Registry keys and scenario file names as in the design; descriptive Python names and message fields | CLAUDE.md rule 1, user approved |
| 2026-11-14 | Architecture doc wins over the Scenario Bindings doc | User |
| 2026-11-14 | Package in `src/lunarsim`; setuptools build backend | Isaac Lab template layout; `uv` not installed |
| 2026-11-14 | `Wrench` gets `mass_flow_rate` | Plant must know propellant flow |
| 2026-11-14 | Truth access for start, plant, surface, sensors, metrics, and `nav: truth` | Design diagram feeds truth to sensors and surface |
| 2026-11-14 | Message `start_state` from start slot to plant | `reset()` has no bus |
| 2026-11-14 | Body origin = whole-vehicle centre of mass | Found by the RCS torque test |
| 2026-11-14 | Timing 5 ms PhysX / 20 ms control; action 19 values; observation 14 values; gimbal accepted but unused | Design plus M1a needs |
| 2026-11-14 | PGS solver, 16 position iterations | TGS softened the strut by 22 % ([[physx-contact-settings]]) |
| 2026-11-14 | Engine 10–65 % throttle, lag 0.08 s, slew 85 %/s; fixed point not modelled | TN D-7143, Klumpp |
| 2026-11-14 | `zem_zev` flies to a 3 m gate, then vertical descent; waits while horizontal speed is high | Full-box success 81 % → 97 % ([[classical-agents]]) |
| 2026-11-14 | `surface: procedural` smoke-tested only | Agents assume ground at z = 0 |
| 2026-11-14 | Visual struts in the lander USD (no mass, no collision) | Pictures; GPU tests rerun |
| 2026-11-14 | Stop at the M1a gate; M1b not started | User |
| 2026-11-14 | Pinned versions in `requirements-pinned.txt`; `scripts/setup_workstation.sh` and `scripts/check_installation.py` | User asked for the move to the RTX 5090 |
| 2026-11-14 | CPU-test workflow on GitHub Actions | After the user added the `workflow` token scope; first run passed |
| 2026-11-14 | **No NASA lander mesh in simulation**; keep our lander, match LM footpads, c.g. and geometry | User: the mesh is too slow |
| 2026-11-14 | **Ground must be real terrain** with craters, rocks and slopes | User |
| 2026-11-14 | **Height maps are not committed**; users download them into `assets/terrain/apollo17/` | User |
| 2026-11-14 | Keep existing preset numbers; change lander values only with a public source | User |
| 2026-11-14 | Project memory in `docs/knowledge/` (this graph), checked in CI | User asked for context that survives machine changes ([[knowledge-graph-tooling]]) |
| 2026-11-15 | User said "proceed" to [[terrain-integration-plan]] | User |
| 2026-11-15 | **Observation grows to 15 values**: footpad height above the ground straight below the c.g. | User chose it; agents need it on uneven ground |
| 2026-11-15 | Iyy = 13,867 slug ft² (18,801 kg m²), not 13,667 | Page image of Mission Report Table A-I |
| 2026-11-15 | RCS clusters at 1.679 m along each body axis, horizontal jets at R-567 offsets | MIT R-567; the M1a reading of Cox gave 29 % too little torque |
| 2026-11-15 | Lander gets c.g. offset, dish footpads (0.940 × 0.178 m), descent and ascent stage shapes; nozzle without collision (crushable), clearance recorded | User: match the real LM; sources in the preset |
| 2026-11-15 | Plant applies actuator forces at the vehicle c.g. (offset = minus the root body's c.g.) | Isaac Lab's composer measures offsets from the body c.g.; with the c.g. offset the old zero offset tilted the hover |
| 2026-11-15 | New message `SurfaceGeometry`: per-env site origin and ground lookup; surface resets before the plant | Site moves per reset on real terrain; the plant needs the ground for the height |
| 2026-11-15 | Terrain = measured 2 m Apollo 17 map (±600 m) + 0.25 m detail patch (±80 m) with craters (Minton 2019 law) and rocks (Surveyor 1 law); targets uniform ±50 m | [[measured-terrain-surface]] |
| 2026-11-15 | Height maps read with numpy from the raw `.TIF`; downloaded by `scripts/download_height_maps.py` into git-ignored `assets/terrain/apollo17/` | No new package in the Isaac environment; user: not in git |
| 2026-11-15 | Site centre = Apollo 17 LM position 20.19091° N, 30.77168° E | Haase et al. 2019, Table 3 |
| 2026-11-15 | New outcome `body_contact`; records nozzle clearance, footprint slope and relief | Real terrain needs strike and site metrics |
| 2026-11-15 | Hover-hold tilt tolerance 0.6° → 1.0° | Longer R-567 jet arms make the minimum-pulse cycle larger (measured 0.65°) |
| 2026-11-15 | GPU test runner cleans dead-process `/dev/shm` files and restarts a silent launch (unbuffered child output) | Intermittent launch stalls ([[isaac-lab-quirks]]) |
| 2026-11-15 | Lander default material: friction 1.0 with "multiply" (in `lander_env_cfg.py`) | The effective footpad friction was 0.2 (Isaac Lab default 0.5 × ground 0.4); now the surface's 0.4 acts. Test `test_footpad_friction.py` |
| 2026-11-15 | Contact-rest slide bound 0.1 → 0.02 m; procedural test averages the leg forces over 1 s with 3 % | Bounds were tuned under the wrong friction; single samples have 1.5 % solver noise |
| 2026-11-15 | Drop test on a massif flank 10.16 km W, 0.62 km S of the LM; sites even over 2.5° slope bins | User asked for 1,000 drops from 5 m to see tip-overs on steep slopes ([[drop-test-results]]) |
| 2026-11-15 | Landing videos by recording poses, then replaying them on camera in a second process | User asked for 5 successes and 5 failures, body contacts first; a rerun would not reproduce the same episodes |
| 2026-11-15 | One recorder and one renderer for all videos and stills (`replay_recording.py`, `replay_rendering.py`, `render_replay.py`) | Landings, drops and contact checks share them ([[replay-recording-and-rendering]]) |
| 2026-11-15 | Drop test: tip-over limit of the metric raised to 150° for this test; "tipped over" = final tilt above 45° | User asked to see flips; stopping at 45° cut them off |
| 2026-11-15 | Contact check rule: loaded pad needs |gap| < 3 cm, free pad must not be inside the ground | Contact offset 2 cm; result 32/32 ([[contact-check-results]]) |
| 2026-11-15 | **Engine cut-off = Apollo probe mode**: three 1.727 m surface probes (legs 2, 3, 4, not the ladder leg); the engine stops 1 s after the contact light once the lander is steady (rate < 1°/s, tilt < 3°), else it keeps descending and stops with the pads at the ground ("pad mode"). The old 0.25 m height rule stays as `engine_cutoff_signal: height_above_ground` | User: "go with the probe". Rogers 1973 p. 124–125; delay is a model choice (Apollo 16 "one-potato", Apollo 11 about 3 s). Without the steadiness rule, low-and-fast starts fell to 84 % success (rates of 4°/s at contact) |
| 2026-11-15 | **Observation grows to 16 values**: `SURFACE_PROBE_CONTACT` (the contact light) | Needed for probe mode; M1b has not started, so no trained policy breaks |
| 2026-11-15 | Engine thrust falls in a straight line to zero over 0.5 s after shutdown (was the 0.08 s throttle lag) | Rogers 1973 p. 124: 0.5 s design value; the flown engine took "several seconds" |
| 2026-11-15 | Nozzle clearance stays a record, not a failure; switch `nozzle_clearance_counts_as_failure` (default false) adds outcome `nozzle_contact` (code 11) | User: crushable skirt, keep it changeable |
| 2026-11-15 | Struts stay straight (along body z) for now; the 23° lean comes with the honeycomb crush model in milestone 5 | User left it to Claude: the lean is only in a CAD-based secondary source, and both leg changes need the same re-validation |
| 2026-11-15 | Heavy GPU runs move to the RTX 5090 machine `minsky` (SSH key login) | User request ([[workstation-setup-and-machines]]) |
| 2026-11-15 | **Milestone 1b opened** by the user ("open m1b and train on minsky") | User |
| 2026-11-15 | PPO action mapping: throttle (centred on hover) + wanted tilt flown by the classical attitude controller (`action_mode: attitude_commands`); `axis_commands` (raw jet couples) stays selectable | With raw jet commands the policy did not learn to hold attitude in 300 iterations; the comparison with ZEM/ZEV then isolates guidance |
| 2026-11-15 | Policy decides every 5 control steps (10 Hz), discount 0.99; attitude controller at 50 Hz | At 50 Hz with discount 0.995 the policy looked ~4 s ahead and learned to hover until the propellant ran out |
| 2026-11-15 | Reward: outcome (+100 / −100 / −300 without touchdown), capped touchdown term, Gaudet-style velocity tracking, attitude, propellant; time limit is a real end (no bootstrapping) | Trials learned to fly away (no-touchdown failure was cheaper than a crash) and to hover (potential shaping with discount); see [[ppo-landing-agent]] |
| 2026-11-15 | `EpisodeEvents` carries the outcome code, first-contact flag and touchdown values; env `_get_rewards` and a `critic` truth group are the only env changes for M1b | M1b task list items 2 and 4 |
| 2026-11-15 | Second Isaac environment on `minsky` (`~/lunarsim-workstation`) at the pinned versions; the user's `~/isaac-env` (Isaac Sim 6.2, newer Isaac Lab) is left untouched | Its Isaac Lab cloner API differs (`clone_plan_from_env_0`); pinned setup keeps results comparable |
| 2026-11-15 | M2 sensors as one `terminal_suite` binding (IMU, radar altimeter, Doppler, gauge, probes), each with rate, noise, valid, stamp; numbers from LN-200S, NDL IM-1 calibration, TN D-6849, Apollo 10 DPS report | A slot holds one binding; sources in `configs/sensors/terminal_suite.yaml` ([[sensors-and-ekf]]) |
| 2026-11-15 | Beams ray-cast on the surface's ground lookup, not Isaac Lab `RayCaster` | Same ground as plant and metrics, flat ground included |
| 2026-11-15 | `nav: ekf` = 15-state error-state filter, float64 covariance, perfect onboard map | Gate test NEES 9.78 on recorded truth ([[sensors-and-ekf]]) |
| 2026-11-15 | First navigation estimate written by the start slot (`initial_navigation_state`) from its own generator | The filter must not read truth; S1 and S2 keep identical true starts per seed |
| 2026-11-15 | PPO training length 600 iterations (was 1,500); running S1 seeds stopped at their 600 checkpoint | User: "stop at 600 iterations" |
| 2026-11-15 | **Milestone 3 opened** ("Proceed") | User |
| 2026-11-15 | The faults slot is truth-side (may read TruthState) | Onset at an altitude needs the true height; faults model the physical world like the sensors. The filter still never reads truth or `fault_status` |
| 2026-11-15 | Fault menu and sizes as in `configs/faults/schedule.yaml` (model choices); one fault per episode on the whole altimeter or Doppler sensor | Design S3 lists types and targets but no sizes ([[faults-and-detection]]) |
| 2026-11-15 | `ekf_fdi`: per-sensor chi-square gate 99.9 %, 3 rejections to fail, 10 passes to recover, 0.5 s loss, 0.06 s maximum sample age | Model choices; tested on the synthetic flight |
| 2026-11-15 | Attention encoder built inside rsl_rl (custom `HistoryAttentionModel`), not skrl | User choice: all encoders share one PPO implementation, so paper 1 compares encoders only |
| 2026-11-15 | Policy observation in S3 = 16 navigation values + filter health (failed flag, data age per aided sensor); history of 32 decision steps (3.2 s) | Design S3 ("the agent gets the flag and the data age"; history_len 32) |
| 2026-11-15 | Paper 1 trainings run 5 side by side at 4,096 landers each under CUDA MPS | User choice after the benchmark: 350k vs 172k env steps/s without MPS; 4,096 keeps the PPO batch of S1 and S2 ([[workstation-setup-and-machines]]) |
| 2026-11-15 | Results of every finished run are committed to `results/<run>/` (manifest, summary, episodes or policy + training curve, gzipped log), from a separate sparse clone; checkpoints and trajectories stay in `runs/` | User rule; the clone never pulls code under a running queue; about 0.5 MB per run |
| 2026-11-15 | Paper 1: GRU and attention runs continue from iteration 600 to 1,500 (new run directories `_to1500`); the MLP stays at 600 | User choice; the memory encoders were still learning at 600 ([[paper-one-results]]) |
| 2026-11-16 | Paper 1 ablation: the three encoders without the health flags (`ekf_fdi` kept), MLP 600 and GRU and attention 1,500 iterations from scratch, 5 seeds | User choice; with the flags the MLP is best, so that setup cannot show what memory adds ([[paper-one-results]]) |
| 2026-11-16 | Build M4 to M7 without training longer than 20 min; learned parts listed in [[skipped-learning-work]] | User rule |
| 2026-11-16 | S6 defaults taken without asking (user: "reach the M7"): Apollo 17 terrain with a rock-density setting, LiDAR on a pointing mount, classical site choice first | The three choices offered with recommendations; recorded here |
| 2026-11-16 | S6 hazard limits are the LM design surface of NASA TN D-6850 p. 3 (6 deg mean slope, 12 deg effective slope, 24 in relief), not the design file's unsourced `max_safe_slope_deg: 10` | Rule: every number from a public source |
| 2026-11-16 | S6 keeps the Doppler velocimeter (design row lists imu, radar_alt, lidar) | The filter needs velocity aiding; Apollo's landing radar had velocity beams ([D-6846] p. 4) |
| 2026-11-16 | The truth variant of S6 is a binding named `truth_with_hazard_map`; the registry allows it to read truth like `nav: truth` | One binding per slot; the design's variant is "nav: truth with the true hazard map" |
| 2026-11-16 | S6 starts at the Apollo low gate with the low-gate mass derived by the rocket equation | [D-6846] Table I and p. 10 |
| 2026-11-16 | Bekker soil from TN D-6850 p. 3 (Apollo 11: 2 to 3 psi/in, n = 1); the design soil as a preset | Sourced; Apollo 11 is the measured value |
| 2026-11-16 | With soil contact the footpads have no PhysX collision; the surface computes soil forces from the plant's current footpad states each substep and the plant adds them to the leg links | Design table: `bekker_pads` is a surface binding; rigid PhysX contact cannot sink |
| 2026-11-16 | S8 keeps the preset's sourced RCS minimum pulse and valve delay, not the design's unsourced 0.5 N s and 20 ms | Rule: every number from a public source; pulse width is the S8 variant to sweep |
| 2026-11-16 | M5 L2 (PhysX particles) stopped: not possible in Isaac Sim 6.0.1; fix proposed to the user | User rule: "If something in the design does not work in Isaac Lab, stop and explain" |
| 2026-11-16 | S7 runs phase by phase (`scripts/run_mission.py`): every phase is a scenario; the gate states go to the next phase through `episodes.parquet` and `start: from_log` | One plant per scenario; orbit phases in float64, the terminal phase in PhysX; the gate log is also the S7 dispersion record |
| 2026-11-16 | Braking and approach use Apollo-style quadratic guidance with the LM throttle logic (10-60 % or the fixed throttle point, latched until the command falls to 57 %) | [D-6846] p. 4, 8 |
| 2026-11-16 | The engine ignites once aligned and then burns without stopping to the low gate | [D-6846] p. 7: one DPS burn from PDI to touchdown |
| 2026-11-16 | Braking plan starts at the full fixed-throttle-point deceleration and changes linearly to the gate | A constant deceleration needs more thrust than the engine gives at the PDI mass ([[orbit-to-touchdown-mission]]) |
| 2026-11-16 | After the planned time the guidance target moves on at its target rates | Otherwise the quadratic law hovers v_D T / 2 short of a moving gate |
| 2026-11-16 | Orbit-phase navigation is `truth_noise`, 50 m bias per axis (model choice), 5 m vertical in the approach | S7 design has no orbit navigation; Apollo's landing radar updated the altitude from 39,000 ft ([D-6846] Table I) |
| 2026-11-16 | The TRN camera rays are marched onto the onboard map | The approach camera looks far ahead over hundreds of metres of relief; a level-ground model gave wrong fixes |
| 2026-11-16 | S7 terminal phase uses the S6 chain (`site_pick_divert`), plan up to 80 s; S5 and S7 terminal `ppo_gru` skipped | User rule: no training over 20 min ([[skipped-learning-work]]) |
