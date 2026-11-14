# PROGRESS

Last update: 2026-11-14

## Done
- Read the design docs and all diagrams; checked the machine; settled the open decisions (see PLAN.md).
- **Milestone 0 gate passed.** Core package (message types, bus, registry, clock, generators,
  frames stub), Hydra scenario loader, `LanderEnv` with the slot loop. 1,024 envs step without
  error; hook order is tested; 61 CPU tests pass.
- Milestone 1a so far:
  - Apollo LM preset, every number with its source (`configs/landers/apollo_lunar_module.yaml`).
  - Lander articulation built from the preset (body, 4 legs, 4 prismatic spring-damper struts).
  - `engine_rcs_pulse` actuators, `terminal_box` start, `physx_site` plant, `flat` surface.
  - Physics validation tests pass: free fall, hover, RCS torque, propellant (CPU), contact rest.
  - Contact settings chosen: PGS solver, 16 position iterations, 1 velocity iteration.

## In progress
- Touchdown metric, run manifest, episode records. Then the controller, in four steps.

## Blocked
- **CI workflow cannot be pushed.** The GitHub token of `gh` has no `workflow` scope, so GitHub
  rejects `.github/workflows/cpu_tests.yml`. The file is written and waits untracked in the
  working tree. To unblock, run `gh auth refresh -h github.com -s workflow`, then commit and
  push that file.

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
