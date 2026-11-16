# PROGRESS

Last update: 2026-11-15

## Done
- Read the design docs and all diagrams; checked the machine; settled the open decisions (see PLAN.md).
- **Milestone 0 gate passed.** Core package, Hydra scenario loader, `LanderEnv` with the slot
  loop. 1,024 envs step without error; hook order is tested.
- **Milestone 1a gate passed.** See `M1A_REPORT.md`.
  - Apollo LM preset with a source for every number; lander articulation built from it.
  - `engine_rcs_pulse`, `terminal_box`, `physx_site`, `flat`, `procedural`, `truth`, `touchdown`.
  - Physics validation tests pass: free fall, hover, RCS torque, propellant, contact rest.
  - Controller in four tested steps: attitude hold, hover, vertical descent, full ZEM/ZEV.
  - ZEM/ZEV: 256 of 256 nominal starts; 96.8 % over the full `terminal_box` range (3 seeds x 4,096).
  - Run manifest, episode records, `lunarsim figures`, `lunarsim summary`.
  - Speed baseline: 69,646 env steps/s at 1,024 envs, 203,702 at 16,384 (RTX 5060).
- Overview figure and lander render in the README.

- Workstation setup script, pinned requirements and installation check (for the move to an RTX 5090).
- CPU tests run on GitHub Actions on every push.
- Project knowledge graph in `docs/knowledge/` (start at `docs/knowledge/README.md`); it holds
  the rules, decisions, results, plans and open questions so context survives machine changes.

- **Real terrain (2026-11-15).** `surface: measured_terrain` with the Apollo 17 height map,
  craters and rocks; LM-accurate lander; 15th observation value (height above ground); four
  milestone 1a errors found and fixed (M1A_REPORT.md section 10). ZEM/ZEV: 97.53 % on flat
  ground, 96.78 % on Apollo 17. Drop test: 1,000 drops from 5 m on steep ground.
- Landing video of chosen successes and failures: `scripts/record_landing_videos.py`.
- **Apollo probe-mode engine cut-off** (1.727 m probes, contact light), engine tail-off 0.5 s,
  nozzle failure switch; observation has 16 values.
- **Milestone 1b gate passed.** `ppo_mlp` with rsl_rl (10 Hz decisions, throttle + wanted tilt
  through the attitude controller, landing reward). Five seeds, 600 iterations, on the RTX 5090
  (`minsky`): PPO 97.95 % ± 0.44 vs ZEM/ZEV 97.68 % on the same 4,096 held-out starts
  (`docs/knowledge/milestone-1b-results.md`, `docs/m1b/`).
- **Milestone 2 gate passed.** Sensor suite (LN-200S IMU, radar altimeter, NDL Doppler,
  propellant gauge, probes, flash LiDAR), error-state EKF, S2: NEES 9.78 on recorded truth;
  ZEM/ZEV 97.46 % on the estimate vs 97.66 % on truth. Visual sensor check: free falls, every
  sensor matches its stated noise (`docs/sensor_check/`).
- **Milestone 3 built (training running).** Seeded fault schedule, `ekf_fdi`, `ekf_late_fusion`,
  `faults: delay`, `fault_response` metric, `ppo_gru`, `ppo_attention`, scenarios S3 and S4,
  held-out fault grid. ZEM/ZEV under one fault per landing: 95.4 % with `ekf_fdi`, 75.9 % with
  the plain `ekf`.

- **Milestone 3 training part done.** Paper 1 results table filled (`docs/knowledge/paper-one-results.md`).
- **Milestone 4 hazard part built (no learning).** S6: pointed flash LiDAR, onboard hazard map,
  LM footprint criteria from TN D-6850, `site_pick_divert`, `site_safety`
  (`docs/knowledge/hazard-detection-and-avoidance.md`).
- **Milestone 5 built except L2.** Bekker footpad soil, L1 plume, dust fault, S8
  (`docs/knowledge/plume-regolith-and-dust.md`); L2 waits for the user.
- **Milestone 6 built (no learning).** float64 orbit plant, MCI/MCMF/SITE frames, phase gates and
  gate logs, Apollo-style quadratic guidance, handoff to PhysX at the low gate, camera TRN; S7
  (orbit to touchdown) and S5 (TRN descent) fly the Apollo 11 timeline
  (`docs/knowledge/orbit-to-touchdown-mission.md`).

- **Milestone 7 built.** `nav`/`agent: ros2_bridge`, loopback node processes, ONNX export of
  `ppo_gru`, deadline model, S10 parity and timing tests
  (`docs/knowledge/software-in-the-loop-bridge.md`). ROS 2 transport untested (not installed).

## In progress
- Nothing running. Next: the user's decisions in `docs/knowledge/pending-user-decisions.md`.
  Learned work skipped in M3 to M6 is listed in `docs/knowledge/skipped-learning-work.md`.

## Waiting for the user
- Open questions: `docs/knowledge/pending-user-decisions.md`.

## Blocked
- Nothing.

## Decisions
| Date | Decision | Why |
| --- | --- | --- |
| 2026-11-14 | Pin Isaac Sim 6.0.1 + Isaac Lab 3.0.0 develop@e86463d, PhysX backend | It is what runs on this machine |
| 2026-11-14 | Forces go through the permanent wrench composer | `set_external_force_and_torque` is deprecated in this Isaac Lab and only forwards to it |
| 2026-11-14 | Plant state is read twice per control step (hook 3 and hook 6) | Metrics must see the state after physics |
| 2026-11-14 | Registry keys as in the design; descriptive names in code | CLAUDE.md rule 1 |
| 2026-11-14 | Architecture doc wins over the Scenario Bindings doc for `Binding` and `TruthState` | User: it is the main authority |
| 2026-11-14 | Package lives in `src/lunarsim` (PLAN.md said `source/lunarsim/lunarsim`) | That is the layout the template in this Isaac Lab version generates |
| 2026-11-14 | Build backend is setuptools, not the template's `uv_build` | `uv` is not installed here |
| 2026-11-14 | `Wrench` has an extra field `mass_flow_rate` | The plant must know how much propellant the actuators burn; this keeps "one output type per slot" |
| 2026-11-14 | Truth access: world-side slots (start, plant, surface, sensors, metrics) and `nav: truth` may set `reads_truth`; faults, other navigation, agent and actuators may not | The design text allows only `nav: truth` and metrics, but sensors and surface take `TruthState` as input in the slot-loop diagram |
| 2026-11-14 | A start slot message `start_state` carries the sampled state to the plant | `reset()` has no bus argument in the design interface |
| 2026-11-14 | Body frame origin = centre of mass of the whole vehicle (body + legs); forces act there | The Apollo mass table gives the whole-vehicle centre of mass; found by the RCS torque test |
| 2026-11-14 | PGS solver, 16 position iterations (not the default TGS) | TGS made the strut spring 22 % too soft at rest; PGS reproduces it to 0.2 % at the same speed |
| 2026-11-14 | Descent engine: throttle 10-65 % of rated thrust, time constant 0.08 s, slew 85 %/s; fixed throttle point not modelled | Values from NASA TN D-7143 and Klumpp (MIT R-695) |
| 2026-11-14 | `zem_zev` flies to a gate 3 m above the target, then uses the vertical-descent controller; it waits above the ground until the horizontal speed is small | Raised full-range success from 81 % to 97 % |
| 2026-11-14 | `surface: procedural` is smoke-tested only | Agents assume the ground at z = 0 until the altimeter exists (M2) |
| 2026-11-14 | The lander USD has thin visual struts (no collision, no mass) | For pictures; physics tests rerun after the change |
| 2026-11-14 | Project memory lives in `docs/knowledge/`, checked by a CPU test | Context must survive moves between machines and sessions |
| 2026-11-15 | Observation has 15 values (height above the ground below) | User decision; agents land on uneven ground |
| 2026-11-15 | Real terrain from public height maps that are downloaded, not committed | User; see docs/knowledge/apollo17-height-maps.md |
| 2026-11-15 | Later decisions (probe mode, M1b, M2, M3) are kept in `docs/knowledge/decisions-log.md` | One place for decisions |
| 2026-11-15 | Lander default material friction 1.0 ("multiply"), so the surface's 0.4 acts | The default 0.5 halved the friction |
