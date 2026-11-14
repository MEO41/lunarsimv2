---
id: lander-preset-apollo-lm
title: Lander preset, Apollo LM
type: component
status: current
updated: 2026-11-14
summary: The Apollo LM-5 preset with its sourced numbers, the body frame, and the values that are model choices or assumptions.
---

# Lander preset: Apollo Lunar Module (LM-5, Apollo 11), terminal phase

File: `configs/landers/apollo_lunar_module.yaml` (every number has its source next to it);
loader `src/lunarsim/assets/lander_preset.py`; builder [[lander-articulation-model]].

## Body frame

Origin at the whole-vehicle centre of mass at lunar landing; +z up along the thrust axis
(LM +X); +x forward (LM +Z, ladder leg); +y left (LM −Y).

## Numbers (35 in total; short list)

| Quantity | Value | Source |
| --- | --- | --- |
| Mass at start | 7,327.0 kg | Apollo 11 Mission Report, mass table, "Lunar landing" (16,153.2 lb) |
| Descent propellant at start | 349.3 kg, of which 43.5 kg unusable | Mission Report 9.13.1 |
| Inertia x, y, z | 21,970 / 18,530 / 17,059 kg m² | Mission Report mass table (see open question below) |
| Centre of mass above footpad plane | 3.62 m | derived: station 213.5 in − 70.84 in |
| Descent engine rated thrust | 46,706 N (10,500 lbf) | NASA TN D-7143 |
| Throttle range | 0.10 to 0.65 (92.5 % fixed point not modelled) | TN D-7143 |
| Engine Isp | 305 s | TN D-7143 |
| Thrust lag / slew | 0.08 s / 0.85 per s | Klumpp, MIT R-695 |
| Engine-off below | 0.05 throttle | **model choice** |
| RCS jet | 444.8 N, Isp 281 s, min pulse 14 ms, valve delay 7 ms, 16 jets in 4 clusters | Grumman LM News Reference |
| RCS cluster arm | 1.676 m per axis | Cox, NASA 1978, Table I |
| RCS cluster height above c.g. | 1.03 m | **ASSUMPTION** in the file; [[apollo-lm-geometry-research]] found MIT R-567 data that confirm it (jet plane X = 254 in) |
| Footpad radius from axis | 4.256 m | NASA TN D-6850 (167.57 in) |
| Footpad diameter | 0.940 m | TN D-6850 App. C |
| Strut stroke | 0.813 m | TN D-6850 Fig. 7 |
| Strut stiffness | 78,800 N/m | **derived model choice**: 20.0 kN at 0.254 m |
| Strut damping ratio | 1.0 | **model choice** |
| Leg mass | 55.2 kg | TN D-6850 Table III (220.7 kg / 4) |
| Touchdown limits | vertical 3.048 m/s; horizontal 1.219 m/s up to 2.134 m/s vertical, then falls to 0; tilt 6°; tilt rate 2°/s | TN D-6850 p. 3 |

Lunar constants (`core/physical_constants.py`): GM 4.902800118e12 m³/s², radius 1,737.4 km,
surface gravity 1.6242 m/s², g0 9.80665 m/s². Simplifications: small c.g. offsets and products
of inertia set to zero (to be changed, see [[terrain-integration-plan]]).

## Open questions

- **Inertia about y:** the preset uses Iyy = 13,667 slug ft² (18,530 kg m²). The research agent
  read 13,867 slug ft² (18,801 kg m²) in the SP-238 edition of the Mission Report. The page
  must be read again before changing the file. Listed in [[pending-user-decisions]].
- Planned changes (dish footpad, c.g. offsets 10 mm / 41 mm, body shapes, nozzle, real jet
  coordinates): [[terrain-integration-plan]], data from [[apollo-lm-geometry-research]].

- sources: [[sources-bibliography]]
- weak points: [[open-issues]]
