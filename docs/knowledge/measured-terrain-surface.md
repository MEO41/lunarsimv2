---
id: measured-terrain-surface
title: Measured terrain surface (Apollo 17)
type: component
status: current
updated: 2026-11-15
summary: The measured_terrain surface binding - real NASA height map, craters and rocks from published laws, one landing target per environment, exact GPU height lookup.
---

# Measured terrain surface (`surface: measured_terrain`)

Built on 2026-11-15 from [[terrain-integration-plan]]. Files:

| File | What it does |
| --- | --- |
| `src/lunarsim/terrain/lunar_height_map_reader.py` | Reads the PDS3 label and memory-maps the raw float32 data of an LROC NAC DTM; local east/north/up frame at a site (numpy only, no GeoTIFF library). Matches the sibling folder's rasterio reader to every digit |
| `src/lunarsim/terrain/surface_detail.py` | Draws craters and rocks from the published laws; crater and rock profiles (model choices) |
| `src/lunarsim/terrain/measured_terrain_builder.py` | Outer grid (measured, 2 m, ±600 m) and detail patch (0.25 m, ±80 m) with craters and rocks fading out over 5 m at the patch edge, so the two meshes meet without a gap |
| `src/lunarsim/terrain/terrain_height_field.py` | Batched torch height and slope lookup that uses the same triangles as the PhysX meshes |
| `src/lunarsim/slots/surface/measured_terrain.py` | The binding: builds the meshes in `setup_scene`, draws a new target per environment at each reset |
| `configs/surface/apollo17_taurus_littrow.yaml` | All values, with sources |
| `configs/scenarios/apollo17_terminal_truth.yaml` | S1 on this surface |
| `scripts/download_height_maps.py` | Downloads the height maps into `assets/terrain/apollo17/` |

## How it fits the slot loop

- The surface publishes `SurfaceGeometry` (new message key `surface_geometry`): the world
  position of every environment's SITE origin (its target on the ground) and the height lookup.
- The SITE frame of an environment has z = 0 at the ground of its target. The plant places the
  lander relative to it and reports `footpad_height_above_ground` (the 15th observation value).
- All environments share the patch; the environment already filters collisions between
  environments, so landers never touch each other. The clone grid spacing is 1 m and the first
  spawn is 40 m up (before the first reset).
- Targets are uniform in ±50 m; the patch is ±80 m, so a footprint never leaves it.

## Numbers (Apollo 17, seed 0)

Build time 0.2 s on the CPU. Outer mesh 707,200 triangles, detail mesh 819,200. In the patch:
863 craters of 1–10 m, 99 rocks of 0.75–2 m (48 above 1 m). Measured relief in the outer grid
−41 m to +20 m. Median local slope 3.1° (95th percentile 13°) on the 0.25 m cells.

## Checks

- CPU: exact triangle lookup, seamless levels, crater and rock counts against the laws, reader on
  a synthetic product (`tests/unit/test_terrain.py`).
- GPU (`tests/smoke/test_measured_terrain.py`): 32 landers dropped from 1.5 m rest with three or
  more dishes on the looked-up ground within 3 cm (measured: within 1.5 mm).

## Limits (model choices, see [[open-issues]])

Crater rim and ejecta profile; rocks are domes in the ground mesh (no rolling, no overhang);
no regolith sinkage; the 2 m map smooths features below about 4 m, which the drawn craters fill.

- part of: [[bindings-catalogue]]
- data: [[apollo17-height-maps]]
- statistics: [[lunar-surface-statistics-research]]
