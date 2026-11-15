---
id: lander-articulation-model
title: Lander articulation model
type: component
status: current
updated: 2026-11-15
summary: How the lander is built in USD from the preset - two stage shapes, four legs with dish footpads on prismatic spring struts - and its simplifications.
---

# Lander articulation model

File: `src/lunarsim/assets/lander_builder.py`. The USD is written as text (`.usda`) from the
preset, so it needs no simulator and is covered by CPU tests. The file name holds a hash of the
content, so a changed preset never reuses an old file.

## Structure (since 2026-11-15)

- One articulation. Root link `body` (mass, 3-D c.g., diagonal inertia, articulation root) with
  two collision shapes: `descent_stage` (cylinder 4.216 m × 1.749 m) and `ascent_stage` (box
  4.039 × 4.293 × 2.832 m), both on the vehicle axis, which is offset from the c.g.
- Per leg (4): link `leg_N` (55.2 kg) whose origin is the centre of the dish's flat top, a
  **dish footpad**: a spherical cap 0.940 m across and 0.178 m deep (sphere radius 0.71 m),
  curved side down, as a convex-hull mesh of 98 points. Plus a thin visual strut (no mass, no
  collision).
- Per leg: `PhysicsPrismaticJoint` `strut_N` along body z, 0 to 0.813 m, linear force drive,
  78,800 N/m, critically damped against a quarter of the vehicle mass.
- The engine nozzle has **no collision shape**: the real skirt is crushable (28 in). The touchdown
  metric reports its clearance ([[metrics-and-run-records]]).
- A contact sensor on the legs (leg forces) and one on `body` (stage strikes).
- Surface probes (legs 2, 3, 4) have **no shape**: the plant checks their tips against the
  ground lookup and reports the contact light (`TruthState.surface_probe_contact`).

## Simplifications (also in `M1A_REPORT.md`)

- Struts move along body z; the real primary struts lean about 23° outward (CAD-based 2026
  NASA paper only, see [[apollo-lm-geometry-research]]). Decision 2026-11-15: stay straight
  until the honeycomb crush model (milestone 5), so the legs are re-validated once.
- Linear spring-damper; the real strut crushes honeycomb in two stages (20.0 kN for 0.254 m,
  then 42.3 kN to 0.813 m) and does not spring back. Secondary struts are not modelled.
- The dish is fixed to its leg; the real one swivels on a ball joint.
- The descent stage is a cylinder (the real one is an octagon); the ascent stage is a box of its
  envelope.
- The NASA visual model (`apollo_lander_modeling/usd/apollo_lunar_module.usd`) is **not used**:
  331 convex hulls per lander are too slow for thousands of envs (user decision 2026-11-14).

- built from: [[lander-preset-apollo-lm]]
- simulated with: [[physx-contact-settings]]
