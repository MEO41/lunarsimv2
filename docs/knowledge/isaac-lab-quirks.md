---
id: isaac-lab-quirks
title: Isaac Lab quirks
type: environment
status: current
updated: 2026-11-14
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
- **Resets must run in torch inference mode**, or Isaac Lab raises "Inplace update to inference
  tensor outside InferenceMode".
- **TGS solver softens light articulation joints** (strut at 0.78 k); use PGS ([[physx-contact-settings]]).
- **One simulator process holds one environment**, so GPU tests run one process per file
  (`scripts/run_isaac_tests.py`), which also kills a simulator that does not exit after pytest.
- **`isaaclab.sh` is deprecated** ("use `uv run isaaclab`"); still works at this commit.
- The README of this Isaac Lab branch says it targets Isaac Sim 6.1; it runs with 6.0.1 here.
- Sibling folder note ([[apollo-lander-modeling-folder]]): plain Isaac Sim scripts there sometimes
  stall at launch; their `isaac_run.sh` removes dead `/dev/shm/carb-RStringInternals-*` files and retries.
- `TerrainImporter` (`isaaclab.terrains`) supports `generator`, `usd` and `plane`, and has
  `import_mesh` for a trimesh; useful for the real terrain.

- applies to: [[workstation-setup-and-machines]]
