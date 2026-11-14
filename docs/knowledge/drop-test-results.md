---
id: drop-test-results
title: Drop test on steep terrain
type: result
status: current
updated: 2026-11-15
summary: 1,000 engine-off drops from 5 m on a massif flank of the Apollo 17 map - where the lander stands, slides and tips over.
---

# Drop test on steep terrain (2026-11-15)

Asked by the user: drop the LM from 5 m at 1,000 places to test collisions and tip-over on steep
slopes. Script `scripts/run_drop_test.py`, scenario `configs/scenarios/apollo17_drop_test.yaml`,
figure `docs/drop_test/drop_test_5m.png`, run `runs/drop_test_5m` (git-ignored).

## Setup

- Terrain: the 2 m Apollo 17 map centred 10.16 km west and 0.62 km south of the LM, a massif
  flank with 337 m of relief, chosen by a search over the whole map for a 1.2 km window with
  full data coverage and both flat and steep ground. Craters and rocks only in the central
  160 m patch, so most drops land on the measured surface alone.
- Sites: footprint slope (plane fitted under the 4.26 m footprint) computed on a 4 m grid; 1,000
  sites drawn evenly from 2.5° bins (about 62 per bin from 0° to 40°, 12 above 40°).
- Each lander: upright, at rest, engine off, random heading, footpad plane 5 m above the ground
  below the c.g.; impact at about 4 m/s (above the 3.05 m/s design limit). Friction 0.4.

## Results (wall time 109 s on the RTX 5060)

| Footprint slope | Drops | At rest on its legs | Tipped over | Median slide |
| --- | --- | --- | --- | --- |
| 0–5° | 124 | 100 % | 0 % | 0.14 m |
| 5–10° | 124 | 100 % | 0 % | 0.67 m |
| 10–15° | 124 | 94 % | 0 % | 2.8 m |
| 15–20° | 124 | 53 % | 0 % | 11 m |
| 20–25° | 124 | 11 % | 2 % | 24 m |
| 25–30° | 124 | 4 % | 2 % | 29 m |
| 30–35° | 122 | 7 % | 17 % | 30 m |
| 35–40° | 122 | 2 % | 40 % | 43 m |
| > 40° | 12 | 0 % | 42 % | 52 m |

Totals: 460 at rest, 460 still sliding when the 15 s settle limit ran out, 80 tipped over
(the episode ends at 45° tilt), 0 body strikes while upright.

## What it means

- **Sliding before tipping.** With a friction coefficient of 0.4 a body can stand still only on
  slopes below tan⁻¹ 0.4 = 21.8°; the impact makes landers slide from about 10° on. Real
  footpads dig into the regolith, which this rigid model cannot do, so slides are probably too
  long; the friction value is the one NASA judged realistic for sliding (TN D-6850).
- **Tip-over starts near 30°**, below the static tip angle of about 40° (arm 3.01 m, c.g. 3.62 m
  high), because the impact and the slide add rotation.
- The LM was designed for at most 6° mean slope and 12° effective slope (TN D-6850), so every
  drop above about 12° is outside its design surface.
- Peak leg forces are about 100 kN (struts bottom out at 4 m/s); the real honeycomb crushes at
  20 and 42 kN ([[open-issues]]).
- During this test the footpad friction was found to be 0.2 instead of 0.4 (material combine
  mode); fixed before this run ([[decisions-log]], `tests/physics/test_footpad_friction.py`).

- measured on: [[measured-terrain-surface]]
- compares with: [[terrain-results]]
