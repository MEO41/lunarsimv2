---
id: analytic-simulator
title: Analytic simulator
type: component
status: current
updated: 2026-11-20
summary: plant analytic_rigid_body and AnalyticLanderEnvironment - the S1 slot loop in pure torch with an analytic lander, no Isaac Sim; the fast simulator of the transfer study.
---

# Analytic simulator

The fast simulator "A" of the [[analytic-to-physx-transfer-study]]. It runs the S1 scenario with
the same start states, actuators, observation, action, reward and metrics as PhysX ("P"); only
the plant is different. No Isaac Sim is started, so it also runs on a machine without a GPU.

Use: add `plant=analytic_rigid_body` to any flat-ground command line, for example

```bash
python scripts/train_ppo.py scenario=s1_terminal_truth plant=analytic_rigid_body num_envs=4096 seed=0 run_name=ppo_mlp_analytic_seed0
python scripts/evaluate_agent.py scenario=s1_terminal_truth plant=analytic_rigid_body agent=zem_zev num_envs=4096 seed=1000 run_name=zem_zev_analytic
python scripts/benchmark_environment.py scenario=s1_terminal_truth plant=analytic_rigid_body num_envs=4096 steps=500
```

`--device` picks the torch device (default `cuda:0`; `--device cpu` works).

## Files

| File | What it is |
| --- | --- |
| `src/lunarsim/slots/plant/analytic_rigid_body.py` | `plant: analytic_rigid_body` (class attribute `simulator = "analytic"`) |
| `src/lunarsim/env/analytic_lander_environment.py` | `AnalyticLanderEnvironment`: the LanderEnv hooks and Isaac Lab's `DirectRLEnv.step`/`reset` order in plain torch |
| `src/lunarsim/env/slot_loop_parts.py` | Reset order, bus ports and critic truth vector, shared with `LanderEnv` (moved there 2026-11-20, no change of behaviour) |
| `src/lunarsim/env/simulator_choice.py` | Tells from the command line (no torch import) whether to start Isaac Sim; `build_environment` |
| `configs/plant/analytic_rigid_body.yaml` | Model choices of the plant |
| `tests/unit/test_analytic_plant.py`, `test_analytic_environment.py` | Test T1 (CPU) |

## Model

- One rigid body at the vehicle centre of mass; actuator and surface wrench in the body frame;
  lunar gravity; Euler's equations with the gyroscopic term; semi-implicit Euler at the PhysX
  substep (200 Hz), the same order as PhysX's PGS solver (speed first).
- Mass falls by the actuators' propellant flow every substep (PhysX: once per control step).
  The body's share of the inertia scales with the body mass, the legs' share stays (as PhysX).
- Legs (`leg_contact: spring_strut`): massless struts along body z with the preset's stiffness
  (78,800 N/m), damping and stroke (0.813 m); dish footpad = lowest point of its sphere (valid
  below 41° tilt); the strut compresses so the pad stays on flat ground; no pull; a stiff end
  stop past the stroke. The ground carries strut force / cos(tilt) plus the leg's own weight.
- Friction: velocity-level Coulomb (stops sliding within a substep when μN allows, never more
  than μN), μ = the surface's `dynamic_friction` (0.4). The effective mass counts the body's
  turn (1/M + r²/I, smallest I, shared among contacts) so it never overshoots; a first version
  with M/4 per leg made the resting lander chatter and creep 3 cm in 4 s (found by the rest test).
- Body strike: the descent-stage bottom rim against the ground (stiff spring and damper).
- Not modelled: leg-link dynamics, solver iterations, centre-of-mass shift while burning,
  ascent-stage strike, terrain (flat ground only; the environment refuses other surfaces).

## Status (2026-11-20)

- T1 on the CPU: all 13 known-answer checks pass with the PhysX tests' tolerances (free fall,
  hover and mass drop, RCS torque, contact rest with load share and strut compression, sliding
  stop). 11 environment checks pass: shapes, start state = start slot, same seed same flight,
  ZEM/ZEV lands 64/64 nominal starts, truth leak, `end_at_first_contact`, PPO 2 iterations on
  rsl_rl and the export flies, simulator choice. Full CPU suite: 225 passed.
- ZEM/ZEV over the full S1 box, 64 envs on the CPU: 60/64 (2026-11-20, a first look only).
- CPU speed: 20,855 env steps/s at 256 envs (Windows laptop CPU, ZEM/ZEV in the loop).
- **Not yet run on a GPU**; the GPU numbers (T2) and the paired comparison with PhysX (T3) come
  from `noron` ([[workstation-setup-and-machines]]).

- part of: [[analytic-to-physx-transfer-study]]
- mirrors: [[slot-loop-architecture]]
- uses: [[lander-preset-apollo-lm]]
- compared with: [[lander-articulation-model]]
