---
id: slot-loop-architecture
title: Slot loop architecture
type: component
status: current
updated: 2026-11-15
summary: The nine slots, the message bus with its truth guard, the hook order of one control step, timing and message fields.
---

# Slot loop architecture

```
start → plant → sensors → faults → nav → [observation builder] → agent → actuators → surface → plant → metrics
```

- Nine slots: `start, plant, surface, sensors, faults, nav, agent, actuators, metrics`
  (`src/lunarsim/core/binding.py`, `SlotName`). Eight live inside `LanderEnv`. The **agent is
  outside**, because training libraries own it (`tools/episode_runner.py: build_agent`).
- A slot is filled by one **binding**, chosen by name in a Hydra scenario file
  (`configs/scenarios/*.yaml`, one config group per slot under `configs/`). Bindings register
  with `@register_binding(slot, name)` (`core/registry.py`). The full list: [[bindings-catalogue]].
- Binding interface (`core/binding.py`, `BindingBase`): class attributes `slot`, `requires`,
  `provides`, `reads_truth`; methods `setup_scene`, `initialize`, `reset(env_ids, gen)`,
  `begin_control_step`, `step(bus, dt)`. All get a `BindingContext` (env count, device, clock,
  bus, simulation handles, shared data such as the lander preset).

## Message bus and truth guard (goal G3)

- `core/message_bus.py`: holds the latest message per key. Every read and write names the
  binding. A binding may read only its `requires` keys and write only its `provides` keys; only
  `reads_truth=True` bindings may read `TruthState`. Truth reads are logged (`truth_read_log`).
- `core/registry.py` checks before the run that the loop is closed and the truth rule holds.
  `TRUTH_SIDE_SLOTS` = start, plant, surface, sensors, metrics; on the flight side only
  `nav: truth` may read truth. (Deviation from the design text; see [[decisions-log]].)

## One control step: Isaac Lab hooks (`env/lander_env.py`)

1. `_pre_physics_step`: agent output → `Command`; actuators take it.
2. `_apply_action`: actuators + surface → `Wrench` → plant. Repeats **every PhysX substep**.
3. `_get_dones`: plant state read, then metrics.
4. `_get_rewards`: returns zeros (rewards arrive with milestone 1b).
5. `_reset_idx`: start slot, then surface (it may move the site), plant, and the others, only
   for finished environments. Runs in torch inference mode (fix for "Inplace update to inference
   tensor outside InferenceMode").
6. `_get_observations`: plant read **again** (after resets), sensors, faults, nav → `Observation`.

## Timing and sizes

- PhysX step 5 ms (200 Hz); 4 substeps per control step → control at 20 ms (50 Hz).
- **Action: 19 values** = throttle, 2 gimbal angles (accepted, unused), 16 jet commands.
- **Observation: 16 values** = position 3, velocity 3, quaternion 4 (w, x, y, z), body rates 3,
  mass 1, **footpad height above the ground straight below the c.g.** 1 (`NavigationStateLayout`,
  `HEIGHT_ABOVE_GROUND`; added 2026-11-15, user decision). On flat ground it equals position z
  minus 3.62 m. Plus **surface-probe contact** 1 (`SURFACE_PROBE_CONTACT`, 1.0 while a probe
  tip touches the ground: the LM contact light; added 2026-11-15 for the probe-mode cut-off).
  The plant computes it from the ground lookup (`TruthState.surface_probe_contact`) and
  `nav: truth` passes it on; in milestone 2 it becomes a sensor.
- Lateral acceleration comes from tilting the lander (no gimbal model).

## Message types (`core/message_types.py`)

`TruthState` (frame, time, position, velocity, orientation_quaternion, angular_velocity, mass,
contact, footpad_height_above_ground, body_contact, surface_probe_contact), `SurfaceGeometry` (site_origin per env,
height_field; written by the surface, read by plant and metrics), `Measurement`, `NavEstimate`, `Observation`, `Command` (`from_action_tensor`,
`to_action_tensor`), `Wrench` (force, torque and the extra `mass_flow_rate`), `EpisodeEvents`.
Message keys include `start_state` (start slot → plant at reset) and `surface_geometry`
(surface → plant and metrics; written once in `initialize`, site origins updated at reset). Design short names map to
descriptive code names: t→time, r→position, v→velocity, q→orientation_quaternion,
w→angular_velocity, cov→covariance, rcs→reaction_control_jets.

## Conventions

- Frames: SITE frame, z up, origin at the landing target on the surface. Isaac quaternions are
  x, y, z, w; LunarSim uses w, x, y, z; conversion happens **only in the plant**.
- Body frame: origin at the whole-vehicle centre of mass, +z along the thrust axis.
- Forces: Isaac Lab's wrench composer applies a body force at an offset **from the body's own
  centre of mass**. The plant passes minus the root body's c.g. as that offset, so actuator
  forces act at the vehicle c.g. (fixed 2026-11-15; before, zero offsets put them 0.1 m higher).
- SITE frame origin per environment comes from the surface (`SurfaceGeometry.site_origin`), not
  from the clone grid.
- Random numbers: one root seed, one generator per slot (`core/random_generators.py`).

- implements: [[project-goal-and-design-documents]]
- runs on: [[physx-contact-settings]]
- details in: [[bindings-catalogue]]
