---
id: isaac-lab-quirks
title: Isaac Lab quirks
type: environment
status: current
updated: 2026-11-15
summary: Behaviours of this Isaac Sim 6.0.1 / Isaac Lab 3.0 develop build that cost time to find, and how the code handles each.
---

# Isaac Lab quirks (Isaac Sim 6.0.1, Isaac Lab 3.0.0 develop@e86463d)

- **No `--headless` flag.** Headless is the default; `--viz kit` opens a viewer window.
- **Stock tasks default to the Newton backend.** LunarSim forces PhysX; for stock tasks pass `presets=physx`.
- **Asset data are warp arrays**; use `.torch` to get torch tensors.
- **Quaternions:** Isaac uses x, y, z, w; LunarSim uses w, x, y, z. Converted only in the plant
  (`slots/plant/physx_root_body_plant.py`).
- **`set_external_force_and_torque` is deprecated** and forwards to the permanent wrench
  composer; the plant writes the composer every substep.
- **Wrench composer positions are offsets from the body's own centre of mass** (in the body
  frame), not from the link origin. Pass minus the link's c.g. to act at the link origin.
- **Contact sensors** report one force per rigid body, not per collision shape; a separate
  sensor on `body` catches stage strikes. They warn "PhysX does not return a total contact
  force ... Returning net_normal_forces_w" (normal forces only).
- **Material combine modes:** a body without a material gets `sim.physics_material` (default
  friction 0.5, combine "average"). PhysX uses the mode of higher priority (average < min <
  multiply < max), so against a "multiply" ground the friction is the product. LunarSim sets
  the default to 1.0 with "multiply" in `lander_env_cfg.py`.
- **Resets must run in torch inference mode**, or Isaac Lab raises "Inplace update to inference
  tensor outside InferenceMode".
- **TGS solver softens light articulation joints** (strut at 0.78 k); use PGS ([[physx-contact-settings]]).
- **One simulator process holds one environment**, so GPU tests run one process per file
  (`scripts/run_isaac_tests.py`), which also kills a simulator that does not exit after pytest.
- **`isaaclab.sh` is deprecated** ("use `uv run isaaclab`"); still works at this commit.
- The README of this Isaac Lab branch says it targets Isaac Sim 6.1; it runs with 6.0.1 here.
- **Launch stalls:** a simulator launch sometimes stalls. Dead processes leave
  `/dev/shm/carb-RStringInternals-<pid>` and `/dev/shm/sem.carb-RStringInternals-<pid>` files;
  `scripts/run_isaac_tests.py` removes those of dead processes before each launch and restarts a
  launch that prints nothing for 120 s (the child runs with `PYTHONUNBUFFERED=1`, otherwise a
  piped Python child prints nothing until it ends). Same idea as the sibling folder's
  `isaac_run.sh` ([[apollo-lander-modeling-folder]]).
- **Static triangle-mesh ground:** a convex footpad that starts almost fully inside the terrain
  mesh can be pushed out downward. Start landers well above the ground (the terrain test drops
  from 1.5 m; `terminal_box` starts at 5 m or more).
- `TerrainImporter` (`isaaclab.terrains`) supports `generator`, `usd` and `plane`, and has
  `import_mesh` for a trimesh; useful for the real terrain.

- applies to: [[workstation-setup-and-machines]]
