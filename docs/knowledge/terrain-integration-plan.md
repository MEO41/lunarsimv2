---
id: terrain-integration-plan
title: Terrain integration plan
type: plan
status: superseded
updated: 2026-11-15
summary: The plan (2026-11-14) for real Apollo 17 terrain and an LM-accurate lander; approved and built on 2026-11-15, kept as the record of what was agreed.
---

# Terrain integration plan (planned 2026-11-14, not started)

**Status (2026-11-15): approved ("proceed", and yes to the 15th value) and built.** What was
built, and where it differs from this plan, is in [[measured-terrain-surface]] and
[[decisions-log]]. Differences: the RCS arm error was found and fixed on the way; the
c.g.-offset change exposed that forces were applied at the body's own c.g. (fixed); the
download addresses are confirmed ([[apollo17-height-maps]]).
The user's requirements behind it are in [[rules-from-the-user]].

## 1. Lander: our model, matched to the real LM (no NASA mesh)

| Part | Now | Planned | Source ([[apollo-lm-geometry-research]]) |
| --- | --- | --- | --- |
| Footpad | sphere, point contact | dish 0.940 m across, 0.178 m deep | TN D-6850 App. C |
| Centre of mass | on the axis | offsets 10 mm and 41 mm | Mission Report mass table |
| Body collision | one cylinder | descent stage 4.216 m wide, 1.749 m high, underside 1.43 m above pads; ascent stage 4.29 × 4.04 × 2.83 m | LM Operations Handbook Fig. 1-3 |
| Engine nozzle | none | collision shape, 1.50 m exit diameter, 0.483 m above the footpad plane | TN D-6850 p. 20; Handbook Table 2.3-1 |
| RCS jets | cluster height assumed | real jet coordinates (assumption confirmed) | MIT R-567 sec. 6.4.1 |

Not changed silently: strut inclination (about 23°, CAD-based source only) stays straight and
is listed as an issue; strut spring stays linear; the inertia question is checked first.

## 2. Terrain: real ground with slopes, craters and rocks

- Base: the 2 m LROC NAC DTM of Apollo 17 ([[apollo17-height-maps]]), 1.2 km around the site;
  coarser tiers from the 5 m DTM if needed (method of [[apollo-lander-modeling-folder]]).
- Added on top, in a fine patch, because a 2 m map cannot show them
  ([[lunar-surface-statistics-research]]):
  - small craters, cumulative 0.0336·D⁻² per m² (Minton et al. 2019, measured mare) as
    default; 0.0794·D⁻² (NASA SLS-SPEC-159 design upper limit) as the hard setting;
  - crater depth from NASA's five degradation classes (SLS-SPEC-159 Table 3.4.1.2-2);
  - rocks 0.00234·D⁻²·¹¹ per m² up to 2 m (Surveyor 1, SLS-SPEC-159 §3.4.1.4); height 0.6·D, σ 0.28.
- **Model choices to mark:** the crater bowl-and-rim profile (no sourced formula found); rocks
  are bumps in the ground mesh (no rolling, no overhang).
- Each environment gets its own target point inside the fine patch, drawn again at every reset;
  the site centre is a parameter. Friction stays 0.4.
- The plant gets a per-environment site origin (east, north, terrain height); landers in
  different envs are filtered from colliding with each other.

## 3. Height above the terrain (needs the user's yes)

Agents assume ground at z = 0 ([[classical-agents]]). Proposal: a **15th observation value,
footpad-plane height above the ground directly below**, from a batched GPU height lookup on the
truth side. On flat ground it equals today's altitude, so S1 stays comparable. This changes a
design number (observation 14 → 15).

## 4. Height maps out of git

Downloaded into `assets/terrain/apollo17/` (git-ignored) with `scripts/download_height_maps.py`;
read directly with numpy (no `rasterio` in the Isaac environment); a missing file gives a clear
error with the link. Download addresses still to be confirmed.

## 5. Files

New: `src/lunarsim/terrain/` (height-map reader, crater and rock generator, GPU height lookup),
`slots/surface/measured_terrain.py`, `configs/surface/apollo17_taurus_littrow.yaml`,
`configs/scenarios/apollo17_terminal_truth.yaml`, `scripts/download_height_maps.py`, CPU and GPU tests.
Changed: lander yaml, `lander_preset.py`, `lander_builder.py`, the plant, `nav: truth`, the
observation, the classical agents, the touchdown metric (body/nozzle strike, ground slope),
README, PROGRESS, M1A_REPORT, `.gitignore`, this knowledge graph.

## 6. Expected effects

The dish footpad changes contact: rerun the contact-rest study ([[physx-contact-settings]]) and
the full S1 evaluation; report the new success rate next to 96.77 % ([[milestone-1a-results]]),
plus the success rate on real terrain. No RL.

- changes: [[lander-preset-apollo-lm]]
- changes: [[lander-articulation-model]]
- changes: [[bindings-catalogue]]
- depends on: [[pending-user-decisions]]
