---
id: lander-preset-apollo-lm
title: Lander preset, Apollo LM
type: component
status: current
updated: 2026-11-15
summary: The Apollo LM-5 preset with its sourced numbers, the body frame, the corrections of 2026-11-15, and the values that are model choices.
---

# Lander preset: Apollo Lunar Module (LM-5, Apollo 11), terminal phase

File: `configs/landers/apollo_lunar_module.yaml` (every number has its source next to it);
loader `src/lunarsim/assets/lander_preset.py`; builder [[lander-articulation-model]].

## Body frame

Origin at the whole-vehicle centre of mass at lunar landing; +z up along the thrust axis
(LM +X); +x forward (LM +Z, ladder leg); +y left (LM −Y). Since 2026-11-15 the vehicle axis
(thrust axis, legs, stages, jets) sits at (−0.0406, +0.0102) m from the origin, because the
c.g. is off the axis (`vehicle_axis_position`).

## Numbers (short list)

| Quantity | Value | Source |
| --- | --- | --- |
| Mass at start | 7,327.0 kg | Apollo 11 Mission Report, Table A-I, "Lunar landing" (16,153.2 lb) |
| Descent propellant at start | 349.3 kg, of which 43.5 kg unusable | Mission Report 9.13.1 |
| Inertia x, y, z | 21,970 / **18,801** / 17,059 kg m² | Table A-I (Iyy = 13,867 slug ft², read from the page image) |
| c.g. offset from the vehicle axis | x +0.0406 m, y −0.0102 m | Table A-I: Z = 1.6 in, Y = 0.4 in |
| c.g. above footpad plane | 3.62 m | derived: station 213.5 in − 70.84 in |
| Descent stage (cylinder) | 4.216 m wide, 1.749 m high, underside 1.430 m above the pads | LM-3 Operations Handbook Fig. 1-3 |
| Ascent stage (box) | 4.039 × 4.293 × 2.832 m, on the descent stage; top 6.011 m above pads | same figure |
| Nozzle exit | 1.499 m diameter, 0.483 m above the pads | Handbook Table 2.3-1; TN D-6850 p. 20 |
| Descent engine | 46,706 N rated; throttle 0.10–0.65; Isp 305 s; lag 0.08 s; slew 0.85/s | TN D-7143; Klumpp R-695 |
| Engine off below | 0.05 throttle | **model choice** |
| RCS jet | 444.8 N, Isp 281 s, min pulse 14 ms, valve delay 7 ms, 16 jets in 4 clusters | LM News Reference |
| RCS cluster position | **1.679 m along each body axis** (on the diagonals) | MIT R-567 6.4.1.4; agrees with Cox's 5.5 ft |
| Horizontal jet offsets | 1.562 m across, 1.685 m along the firing axis | R-567 6.4.1.4 |
| Jet plane above c.g. | 1.029 m | derived: station 254 in (R-567) − 213.5 in |
| Footpads | 4 at 4.256 m from the axis; dish 0.940 m across, **0.178 m deep** | TN D-6850 p. 6, App. C p. 50 |
| Strut | stroke 0.813 m; 78,800 N/m (**derived model choice**); damping ratio 1.0 (**model choice**) | TN D-6850 Fig. 7 |
| Leg mass | 55.2 kg | TN D-6850 Table III / 4 |
| Touchdown limits | vertical 3.048 m/s; horizontal 1.219 m/s up to 2.134 m/s vertical, then to 0; tilt 6°; rate 2°/s | TN D-6850 p. 3 |

Lunar constants (`core/physical_constants.py`): GM 4.902800118e12 m³/s², radius 1,737.4 km,
surface gravity 1.6242 m/s², g0 9.80665 m/s². Products of inertia are neglected (below 4.5 %).

## Corrections of 2026-11-15

- **Iyy** was 13,667 slug ft² (a text-extraction error); the page image says 13,867.
- **RCS arms:** Cox's "5.5 ft arm" was read as the distance from the axis on the diagonal
  (1.676 m), which gave only 1.185 m along each axis. R-567 gives 1.679 m along each axis, so
  the M1a jet torques were about 29 % too small. Now corrected.
- The RCS cluster height is no longer an assumption (R-567 confirms 254 in).
- Added: c.g. offset, dish footpad depth, stage shapes, nozzle geometry ([[apollo-lm-geometry-research]]).

- sources: [[sources-bibliography]]
- weak points: [[open-issues]]
