---
id: plume-regolith-and-dust
title: Plume, regolith and dust (scenario S8, milestone 5)
type: component
status: current
updated: 2026-11-17
summary: Bekker soil under the footpads, the L1 plume model, the dust fault and the dust metric of S8; the L2 batched torch grain model (Roberts erosion, Apollo ejection angle, ballistic flight) and the L1 vs L2 check; the L3 data plan.
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

## L2: batched torch grains (built 2026-11-17)

PhysX particles cannot be driven in Isaac Sim 6.0.1 (no particle tensor view; positions do not
come back through USD, probe of 2026-11-16). The user chose a torch grain model that stays light
on the GPU ("It has to be efficiently run on the gpu"). `terrain/plume_grain_model.py`, enabled by
the `grains` block of `configs/surface/regolith_plume_l2.yaml`; method of [MLI-08] (Apollo 12):

| Step | Model | Source |
| --- | --- | --- |
| Gas over the ground | mass flow T / v_e turns outward at 3,200 m/s in a layer `delta = 0.537 sqrt(r^2 + h^2)`; density from mass conservation | gas speed [LMI-08] Fig. 4; layer fitted to 2e-5 kg/m^3 at [LMI-08] p. 7's reference point |
| Erosion | Roberts' excess shear `(1/2) alpha u dm/dt = tau - tau_0`, `tau = c_f rho u^2 / 2`, on 40 rings | [MLI-08] Eq. 1; `c_f` = 0.022 fitted to the Apollo 11 eroded volume |
| Soil | 7 octave bins 4-512 um, count distribution `exp(-(36.6/mm D)^0.81)`, grain density 3,100 kg/m^3, threshold `A_N^2 (rho_p g D + gamma/D)` | [MLI-08] Eq. 5; [LMI-08] p. 3; [SL-00] form (constants model choices) |
| Launch | speed `alpha_k u` with `alpha_k` from the drag law over 10 m of gas (6 um: 1.0, 45 um: 0.43, 362 um: 0.06); angle 2.1 deg | [LMI-08] Eqs. 4-5; [IMH-08] Table 1 (mean of Apollo 11, 14, 16, 17) |
| Flight | ballistic under lunar gravity until the height map or 700 m | [MLI-08] p. 4 |
| Dust seen | optical depth on a 45 deg forward-down line of sight from an (r, height) grid of the cloud, extinction efficiency 2; dust = 1 - exp(-optical depth) | model choices |

Each simulated particle carries an equal share of one size bin's eroded mass (4 per bin per
control step, 2,048 per lander, the oldest reused). New grains fly a random part of their first
step, so a 3 km/s grain is not always counted 64 m away. Cost at 16 landers: 587 -> 545 env
steps/s (7 %), 2.4 MB of GPU memory (`scripts/benchmark_environment.py`).

`grains.drives_dust: false` (shadow, the check scenario `s8_plume_check_l2`) keeps L1 on the dust
fault and records both levels on the same landings (`grain_*` columns of `metrics: dust`);
`true` lets the grains drive the dust. Tests: `tests/unit/test_plume_grain_model.py`,
`tests/smoke/test_plume_grain_scenario.py`. Comparison: `scripts/compare_plume_levels.py`.

### L1 vs L2 check (committed run, 2026-11-17)

`results/s8_plume_check_l2_seed0` (64 landers, seed 0, commit `266acbf`, `comparison.yaml`);
95.3 % landed (the grains only watch, so this is S8's own rate).

| | L1 | L2 grains | Apollo reference |
| --- | --- | --- | --- |
| Eroded mass, landings starting above 100 ft (29) | 23.0 kg (15-36) | 57.1 kg (30-86) | 46.8-74.1 kg (Apollo 11, 36-57 L at 1.30 g/cm^3) |
| Time with the sensors blinded (dust >= 0.5) | 14.8 s (2-46) | 0 s | heavy obscuration reported below about 100 ft |
| Peak dust density / optical depth | 1.0 | 0.009 / 0.009 | - |
| Peak lofted number density | - | 3.4e6 per m^3 (1.6e6-8.9e6) | 1e8 to 1e13 per m^3 ([IMH-08]) |

The grains erode from the first step of every landing (S8 starts at 5-50 m; the grain shear
exceeds the threshold well above 100 ft), so their "erosion start height" is the start height,
not an onset; L1's onset is its 100 ft rule.

### Development run (16 landers), kept for the record

| | L1 | L2 grains | Apollo reference |
| --- | --- | --- | --- |
| Eroded mass from 100 ft down | 24.7 kg | 59 kg (fitted) | 47-74 kg (Apollo 11, 36-57 L at 1.30 g/cm^3) |
| Time with the sensors blinded | 13 s | 0 s | crews reported heavy obscuration below about 100 ft (Apollo 11, ALSJ) |
| Lofted number density | (no particles) | about 3e6 per m^3 at the peak | 1e8 to 1e13 per m^3 ([IMH-08]) |

Reading: L1's ejecta mass is about 2.4 times below the Apollo-anchored grains. On blinding the two
disagree in opposite directions from the evidence: L1 saturates; the grains (Roberts erosion, one
thin sheet at 2.1 deg) put 1.5 to 6.5 orders of magnitude fewer grains in the air than the Apollo
videos show, the same gap [MLI-08] p. 6 reports for its model ("at least two orders of magnitude").
So neither level is trusted for sensor blinding; L3 (below) must calibrate the lofted density,
and S8 results must name the plume level.

Known limits of L2: the [MLI-08] size fit puts most mass at 90-360 um, coarser than the "mass
peaked near 60 microns" of its own Fig. 1; Stokes-type drag in a rarefied gas sends the finest
grains to the full gas speed (above lunar escape speed, as [MLI-08] p. 7 also finds); one ejection
angle for all sizes; no grain-grain collisions; the plume force on the vehicle stays L1's.

## L3 data plan (design: "Offline DEM or CFD runs, fitted to a surrogate for L1")

To replace the model choices of L1 (onset height shape, full-dust height, ground-effect
coefficient and decay, ejecta mass rate): (1) Apollo descent films and transcripts for the dust
onset and visibility per mission (ALSJ landing commentaries of Apollo 11 to 17); (2) published
plume-surface interaction measurements and simulations (for example the Metzger, Lane, Immer and
Clements papers on Apollo blowing soil, and NASA plume-surface interaction CFD of the 2020s);
(3) fit `erosion_onset_height_m`, `full_dust_height_m` and `maximum_ejecta_mass_rate_kg_per_s` to
those, and the ground-effect force to CFD pressure on the descent stage base.

- sources: [[sources-bibliography]]
- part of: [[milestone-roadmap-and-status]]
- uses: [[measured-terrain-surface]], [[lander-articulation-model]], [[faults-and-detection]]
