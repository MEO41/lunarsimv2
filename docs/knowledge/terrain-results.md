---
id: terrain-results
title: Results with the LM-accurate lander and real terrain
type: result
status: current
updated: 2026-11-15
summary: ZEM/ZEV success on flat ground with the corrected lander (97.53 %) and on the Apollo 17 terrain with craters and rocks (96.78 %), with failure causes and ground statistics.
---

# Results with the LM-accurate lander and real terrain (2026-11-15, RTX 5060)

Agent `zem_zev`, full `terminal_box` range, 3 seeds × 4,096 episodes per surface, after all
fixes of 2026-11-15 (Iyy, R-567 jet arms, force at the vehicle c.g., footpad friction 0.4).
Runs (git-ignored): `runs/flat_lm_corrected_zem_zev_seed{0,1,2}`, `runs/apollo17_zem_zev_seed{0,1,2}`.

| Surface | Success | Per seed | Starts not low-and-fast | Low-and-fast starts |
| --- | --- | --- | --- | --- |
| M1a: flat, old lander | 96.77 % ± 0.23 | 96.90 / 96.51 / 96.90 | 99.53 % | 83.6 % |
| Flat, corrected lander | **97.53 % ± 0.12** | 97.66 / 97.53 / 97.41 | **100.00 %** | 85.8 % |
| Apollo 17, craters and rocks | **96.78 % ± 0.25** | 96.51 / 97.00 / 96.83 | 99.67 % | 83.0 % |

"Low and fast" = start below 20 m and faster than 3 m/s down ([[metrics-and-run-records]]).

## Failure causes (12,288 episodes each)

| Outcome | Flat | Apollo 17 |
| --- | --- | --- |
| hard_horizontal | 187 | 199 |
| tilt_rate_at_contact | 40 | 79 |
| hard_vertical | 59 | 63 |
| body_contact | 0 | 27 |
| tilt_at_contact | 17 | 22 |
| not_settled | 0 | 6 |

## Findings

- With the corrected lander, **all flat-ground failures come from low-and-fast starts**: the 48
  unexplained M1a failures are gone. They are most likely due to the two M1a errors found on
  2026-11-15 (jet arms 29 % short; forces applied 0.1 m above the c.g.).
- On terrain the success rate drops by 0.75 points. The new causes are **27 body strikes** (a
  stage hit a rock or the ground; median ground relief under the footprint 1.45 m, against
  1.43 m of stage clearance) and 6 landers that did not settle; tilt-rate failures double
  (uneven legs touching one after the other).
- Ground under the footprint at first contact: slope median 2.6°, 95th percentile 6.0°, max
  9.8°; relief median 0.47 m, 95th percentile 1.05 m, max 2.3 m. In 3,748 of 12,288 episodes
  the relief is above the LM design limit of 24 in (0.61 m, TN D-6850); 95.5 % of those still
  land successfully.
- In 116 episodes the nozzle rim reached below the local ground (minimum −0.94 m), always on a
  rock or crater rim; the real skirt crushes (28 in), so this is recorded, not a failure.
- Landing miss: median 0.21 m, max 41 m (as on flat ground).
- The friction fix (0.2 → 0.4) did not change the flat results at all (outcomes are judged at
  contact) and changed the terrain results by less than 0.2 points.
- Wall time per 4,096 episodes: flat about 85–150 s, Apollo 17 about 220–290 s (the terrain
  meshes have 1.5 million triangles; the user's own terrain builder was running at the same time).

- compares with: [[milestone-1a-results]]
- measured on: [[measured-terrain-surface]]
- Video `runs/landing_videos/landings.mp4` (git-ignored; remake with
  `scripts/record_landing_videos.py`): 5 successes and 5 body-contact failures from a 3,072-landing
  run (96.61 %, 17 body strikes). All strikes happen at the two tallest rocks of the patch.
- see also: [[drop-test-results]]
