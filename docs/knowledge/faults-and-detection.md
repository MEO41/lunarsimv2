---
id: faults-and-detection
title: Faults and fault detection (milestone 3)
type: component
status: current
updated: 2026-11-15
summary: The seeded sensor-fault schedule (faults slot) and the ekf_fdi filter with chi-square gate, isolation and loss detection; test results.
---

# Faults and fault detection (milestone 3)

## `faults: schedule` (`slots/faults/fault_schedule.py`, `configs/faults/schedule.yaml`)

One fault per episode on the radar altimeter or the Doppler velocimeter (whole sensor), drawn
from the faults slot's generator (same seed, same faults for every agent); onset when the
footpad height first falls below the onset altitude (5-40 m, design S3), at once if the lander
starts below it. `set_fixed_faults` replays a fixed grid (held-out test set).

| Class | Types |
| --- | --- |
| intermittent | dropout burst, delay spike (old sample with its old stamp), outlier spike |
| persistent | bias, drift, stuck (last value before the onset, reported as new), scale, total loss |

Sizes (**all model choices**, no public fault statistics): bias 0.5-5 m / 0.05-0.5 m/s; outlier
5-20 m / 0.5-3 m/s; drift 0.1-1 m/s / 0.01-0.1 m/s²; scale ±5-20 %; events 0.2-1 per second,
bursts 0.2-1 s, delays 0.1-0.5 s. The binding is truth-side (reads the footpad height) and
writes `fault_status` for the metrics; the navigation never reads it.

## `nav: ekf_fdi` (`slots/navigation/ekf_fdi.py`, `configs/nav/ekf_fdi.yaml`)

The `ekf` plus, per aided sensor: age check (samples older than 0.06 s dropped), chi-square gate
on the normalised innovation squared (99.9 %), failure after 3 rejections in a row, recovery
after 10 passes in a row, loss after 0.5 s without a sample while a return is expected (the
range limits near the ground are not failures). All thresholds are model choices. The
estimate's `health` has two columns per aided sensor: failed flag and data age (both filters).

## Results (CPU tests, synthetic flight, 200 landers)

- No fault: no sensor ever declared failed in 12 s.
- 5 m altimeter bias from 20 m: declared after 0.12-0.14 s (three samples); height error p95
  0.14 m with `ekf_fdi`, 0.90 m with the plain `ekf`.
- Doppler total loss: declared 0.46-0.5 s after the onset.
- A fault present **from the first sample** (onset above the start altitude) can be
  mis-isolated: with a 1 m first-estimate error the filter may reject the good sensor. With the
  design's ranges (start 5-50 m, onset 5-40 m) this is common, so the metric records it.

## `metrics: fault_response` and navigation error records

`slots/metrics/fault_response.py`: the touchdown metric plus `fault_type`, `fault_target`,
`fault_onset_altitude_m`, `fault_onset_time_s`, `fault_present_at_start`, `fault_detected`,
`fault_detection_delay_s` (onset to the filter's failed flag), `false_alarm`,
`fault_corrupting_steps`. Every touchdown record now also has `navigation_position_error_rms_m`,
`navigation_velocity_error_rms_mps` and `navigation_maximum_height_error_m` (before contact).

## First S3 results (2026-11-15, RTX 5060, ZEM/ZEV, 1,024 landings, seed 0, one fault each)

| Fault | Landings | ZEM/ZEV + `ekf_fdi` | ZEM/ZEV + plain `ekf` |
| --- | --- | --- | --- |
| all | 1,024 | **95.41 %** | **75.88 %** |
| stuck | 148 | 93.9 % | 20.9 % |
| drift | 118 | 94.9 % | 59.3 % |
| bias | 137 | 90.5 % | 73.0 % |
| scale | 140 | 91.4 % | 85.0 % |
| outlier spike | 122 | 100.0 % | 91.0 % |
| delay spike | 122 | 95.9 % | 91.0 % |
| dropout burst | 124 | 98.4 % | 98.4 % |
| total loss | 113 | 100.0 % | 100.0 % |

No faults (S2, same seed): 97.46 %. With `ekf_fdi`, mid-flight faults are declared after (median)
0.12 s (bias, scale), 0.17 s (stuck), 0.48 s (total loss), 0.71 s (drift); outliers and delayed
samples are rejected one by one, not declared; dropout bursts only when longer than 0.5 s. 37 %
of the faults were present from the start (onset above the start altitude); 3.5 % of episodes
had a false alarm.

## Late data (scenario S4)

- `faults: delay` (`slots/faults/delayed_delivery.py`, `configs/faults/delay.yaml`): every
  sample of the listed sensors arrives late (design S4: Doppler 50-200 ms per episode, jitter
  ±20 ms per sample), with its true capture stamp; no out-of-order delivery.
- `nav: ekf_late_fusion` (`slots/navigation/ekf_late_fusion.py`): stores inputs and state of the
  last 0.6 s; a late sample is inserted at its capture step and the filter replays to now
  (batched; only replaying environments change their stored past). Without late data it is
  exactly `ekf`.
- CPU test (synthetic flight, Doppler 100-200 ms late), velocity error p95: no delay 4.5 mm/s;
  late data used as new (`ekf`) 27.9 mm/s; dropped (`ekf_fdi` age check) 483 mm/s (velocity
  unobserved); **late fusion 4.5 mm/s**, NEES 9.49.
- Scenario `s4_late_fusion.yaml`; the design's `camera_cheap` sensor is not built (milestone 6).
- **S4 in the simulator** (ZEM/ZEV, Doppler 50-200 ms late, 1,024 landings, seed 0):

  | Late data | Success | Velocity error RMS (median, p95) | Horizontal velocity error at touchdown (median) | Wall time |
  | --- | --- | --- | --- | --- |
  | applied at capture time (`ekf_late_fusion`) | **97.46 %** (as without delay) | 27, 64 mm/s | 1.1 mm/s | 559 s |
  | used as new (`ekf`) | 86.43 % | 49, 298 mm/s | 16.3 mm/s | 84 s |
  | dropped (`ekf_fdi`) | 95.90 % | 194, 500 mm/s | 208 mm/s | 101 s |

  Late fusion is about 6-7 times slower here (a replay of up to 10 steps at almost every step).

## Held-out fault grid (paper 1)

`slots/faults/fault_grid.py`: 8 fault types × 2 targets × onsets 5, 10, 20, 30, 40 m = 80 cells,
environment i gets cell i mod 80 (about 51 landings per cell at 4,096 envs), sizes at the middle
of their ranges, seed 1000. `evaluate_agent.py fault_grid=true`. Queue for `minsky`:
`scripts/minsky/queue_paper_one.sh` (S2 policy evaluation, S3 training of the three encoders
5 seeds each, grid evaluation of everything plus ZEM/ZEV with `ekf_fdi` and `ekf`).

- part of: [[slot-loop-architecture]]
- extends: [[sensors-and-ekf]]
