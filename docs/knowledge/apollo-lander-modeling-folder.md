---
id: apollo-lander-modeling-folder
title: The apollo_lander_modeling folder
type: research
status: current
updated: 2026-11-14
summary: The user's sibling working folder with the Apollo 17 terrain scene and the NASA LM mesh - what it has, what we take from it, and what we do not.
---

# The `apollo_lander_modeling` folder

Location on the RTX 5060 machine: `~/Desktop/MEO/apollo_lander_modeling` (2.2 GB, **not a git
repository**, own `.venv` with rasterio, trimesh, DracoPy, usd-core). Work in progress by the
user on terrain and surface contact. It is **not** part of the old LunarSim repo, so reading it
is allowed. lunarsimv2 does not depend on it.

## What it has

- `nac_dtm.py`: reads the two NAC height maps in a local east/north/up frame ([[apollo17-height-maps]]).
- `build_apollo17_scene.py`: three mesh tiers, all centred on the touchdown point and blended so
  they are watertight: Far (5 m map averaged to 100 m, about 50 × 58 km, 15 % interpolated), Mid
  (5 m map, 3 km), Near (2 m map, 1.2 km; or the 5 m map as a variant). Exact triangle-mesh colliders.
- `glb_to_usd.py`: converts NASA's Apollo LM GLB (science.nasa.gov 3D resources), scale 1.4,
  one rigid body with 331 convex-hull colliders.
- Studies with Isaac Sim: `settle_lander.py` (rest pose, load per pad), `drop_test.py` (1,000
  sites, 5 m drop), `check_collision.py` and `scan_collision_surface.py` (physics surface against
  the map), `force_arrows.py`, renders.
- `reference/nasa_lm_touchdown/`: AAS-26-167 paper (Apollo 11 LM touchdown dynamics, NASA MSFC).

## What lunarsimv2 takes and does not take

- Takes: the method (reading the maps, local frame, tier blending), redone in lunarsimv2 code.
- Does not take: the NASA mesh (user decision: too slow), the rigid lander, the folder's numbers.
  It uses mass 7,063.5 kg, friction 0.8/0.7 ("approx.") and gravity 1.62; lunarsimv2 keeps
  7,327 kg, 0.4 and 1.6242 ([[lander-preset-apollo-lm]]).
- The NASA model's footpads sit about 5 % closer to the axis than the real 4.256 m.
- Later changes in the folder need a new port; it is a snapshot of 2026-11-14.

- source for: [[terrain-integration-plan]]
