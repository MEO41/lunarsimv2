---
id: plume-regolith-and-dust
title: Plume, regolith and dust (scenario S8, milestone 5)
type: component
status: current
updated: 2026-11-16
summary: Bekker soil under the footpads, the L1 plume model, the dust fault and the dust metric of S8; first results; the L2 check that does not work in this Isaac Sim (decision pending) and the L3 data plan.
---

# Plume, regolith and dust (S8)

Design S8: "How do plume forces, dust and footpad sinkage change the last metres of the landing?"
Built 2026-11-16 without learning (the design's agent `ppo_gru` is skipped, ZEM/ZEV flies;
[[skipped-learning-work]]).

## Parts

| Part | Binding / file | What it does |
| --- | --- | --- |
| Soil and plume | `surface: regolith_plume_l1`, `slots/surface/regolith_plume_surface.py`, `terrain/regolith_models.py` | Apollo 17 terrain plus, every PhysX substep, Bekker soil forces on each footpad and the L1 plume wrench; writes `regolith_state` |
| Footpads | `plant: physx_site` with `footpad_contact: soil` (set by the surface through the shared data) | Footpad meshes have `physics:collisionEnabled = 0`; the soil forces are added to the leg links (`Wrench.leg_forces`, wrench composer `add_forces_and_torques_index`); `TruthState.contact` is the soil force |
| Dust fault | `faults: dust`, `slots/faults/dust.py` | LiDAR pixel loss (0.9 x density) and short dust returns (0.05 x density); Doppler-lidar beam loss (0.5 x density); radar altimeter untouched |
| Metric | `metrics: dust`, `slots/metrics/dust.py` | `touchdown` plus blinded time, erosion start height, ejecta mass, peak dust, plume force and torque at contact, footpad sinkage |
| Actuators | `Wrench.engine_thrust` from `engine_rcs_pulse` | The plume needs the descent-engine thrust alone |

Soil ([D-6850] Rogers, NASA TN D-6850 p. 3): Apollo 11 "2 to 3 psi/in. lower boundary of bearing
strength" -> Bekker n = 1, k = 2.5 psi/in (default `preset: apollo11`); the LM design soil "1.0
lb/in^2 -> 4 inches, 12 lb/in^2 -> 24 inches" (`preset: lm_design`, n = 1.387). The dish is
integrated over 64 rings; plastic unloading over 2 mm; damping 2,000 N s/m on penetration only;
Coulomb friction 0.4. Static sinkage under a quarter of 7,327 kg: 4.4 cm. For comparison
[D-6850] p. 18: Apollo 11 "maximum footpad penetration of 0.5 to 1.5 inches and a footpad slide
distance of 18 to 22 inches" (postflight simulation of a softer touchdown).

Plume L1: dust onset 100 ft (Armstrong, ALSJ "The First Lunar Landing" after 102:45:17: "I first
noticed that we were, in fact, disturbing the dust on the surface when we were something less
than 100 feet"), full dust at 3 m and the reflected-plume force `0.05 T exp(-h/D)` are model
choices waiting for L3.

## First results (2026-11-16, 16 landers, development run)

ZEM/ZEV with `ekf_fdi` on Apollo 17, soil and plume on: 16 of 16 landings succeed; erosion starts
at a nozzle height of 27 m on average (11 to 30.5 m); sensors are blinded (dust >= 0.5) for 13 s;
26 kg of ejecta; final footpad sinkage 12 cm (peak leg force 28 kN at 1.45 m/s touchdown: the
soil law gives 14 cm at that load); horizontal navigation error at contact 0.35 m. The plume
force at first contact is zero: the engine stops 1 s after the contact light (probe mode, as
Apollo did), so the plume has tailed off before the pads touch.

## L2 check: does not work in this Isaac Sim (decision pending)

The design's L2 is "PhysX particles under the nozzle, few envs, visual checks of L1". A probe on
2026-11-16 (Isaac Sim 6.0.1, Isaac Lab `e86463d`): PBD granular particles can be created
(`omni.physx.scripts.particleUtils`), but the PhysX tensor API has no particle view
(`create_particle_system_view` does not exist) and the particle positions do not come back
through USD, so the plume cannot push the particles. Proposed fix, see [[pending-user-decisions]]:
grains as many small PhysX **rigid bodies** (tensor API rigid-body views work) with a plume
gas-drag force applied each step, in a separate check scene with a few environments.

## L3 data plan (design: "Offline DEM or CFD runs, fitted to a surrogate for L1")

To replace the model choices of L1 (onset height shape, full-dust height, ground-effect
coefficient and decay, ejecta mass rate): (1) Apollo descent films and transcripts for the dust
onset and visibility per mission (ALSJ landing commentaries of Apollo 11 to 17); (2) published
plume-surface interaction measurements and simulations (for example the Metzger, Lane, Immer and
Clements papers on Apollo blowing soil, and NASA plume-surface interaction CFD of the 2020s);
(3) fit `erosion_onset_height_m`, `full_dust_height_m` and `maximum_ejecta_mass_rate_kg_per_s` to
those, and the ground-effect force to CFD pressure on the descent stage base.

- part of: [[milestone-roadmap-and-status]]
- uses: [[measured-terrain-surface]], [[lander-articulation-model]], [[faults-and-detection]]
