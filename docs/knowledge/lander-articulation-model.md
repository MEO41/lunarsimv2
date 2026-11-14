---
id: lander-articulation-model
title: Lander articulation model
type: component
status: current
updated: 2026-11-14
summary: How the lander is built in USD from the preset - body, four leg links on prismatic spring struts, sphere footpads - and its simplifications.
---

# Lander articulation model

File: `src/lunarsim/assets/lander_builder.py`. The USD is written as text (`.usda`) from the
preset, so it needs no simulator and is covered by CPU tests. The file name holds a hash of the
content, so a changed preset never reuses an old file.

## Structure

- One articulation. Root link `body` (rigid body, mass, c.g., diagonal inertia, articulation root).
- Body collision: **one cylinder** of radius 2.108 m from 2.092 m below to 3.365 m above the c.g.
- Per leg (4): link `leg_N` with mass 55.2 kg, a **sphere footpad** of radius 0.47 m (so each leg
  touches the ground in one point), and a thin visual strut cylinder (no mass, no collision; for
  pictures only, added after M1a and the GPU tests were rerun).
- Per leg: `PhysicsPrismaticJoint` `strut_N` along body z, joint position = how far the pad moved
  up into the body, limits 0 to 0.813 m, linear force drive with stiffness 78,800 N/m and
  critical damping against a quarter of the vehicle mass.

## Simplifications (also in `M1A_REPORT.md`)

- Struts move along body z; the real primary struts lean about 23° outward (from a CAD-based
  2026 NASA paper only, see [[apollo-lm-geometry-research]]).
- Linear spring-damper; the real strut crushes honeycomb in two stages (20.0 kN for 0.254 m,
  then 42.3 kN up to 0.813 m) and does not spring back. Secondary struts are not modelled.
- Sphere footpads instead of 0.94 m dish pads on a ball joint.
- One cylinder instead of descent stage, ascent stage and engine nozzle.
- The NASA visual model (`apollo_lander_modeling/usd/apollo_lunar_module.usd`) is **not used**:
  331 convex hulls per lander are too slow for thousands of envs (user decision 2026-11-14).

The plan to replace the sphere pads and the cylinder: [[terrain-integration-plan]].

- built from: [[lander-preset-apollo-lm]]
- simulated with: [[physx-contact-settings]]
