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

- part of: [[slot-loop-architecture]]
- extends: [[sensors-and-ekf]]
