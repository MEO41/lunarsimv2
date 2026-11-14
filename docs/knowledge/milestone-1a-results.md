---
id: milestone-1a-results
title: Milestone 1a results
type: result
status: current
updated: 2026-11-14
summary: Test counts, physics validation, controller results and the ZEM/ZEV full-box success rate with its failure breakdown (RTX 5060, flat ground).
---

# Milestone 1a results (flat ground, RTX 5060, 2026-11-14)

Full report: `M1A_REPORT.md`. Figures: `docs/m1a/*.png`, shown in the README.

## Tests

- 84 CPU unit tests pass at the M1a gate (`tests/unit`); 89 since the knowledge-graph tests; on GitHub Actions since commit `7e35240`.
- 12 of 12 GPU test files pass (`tests/physics`, `tests/smoke`), rerun after the visual-strut change.

## Physics validation (goal G2)

Free fall, hover, RCS torque (0.5 % error), propellant and rocket equation (CPU), contact rest:
all pass. Details: [[physx-contact-settings]].

## ZEM/ZEV over the full `terminal_box` range

Three seeds × 4,096 episodes: **96.90 %, 96.51 %, 96.90 % → 96.77 % ± 0.23** (11,891 of 12,288).
Run directories (git-ignored): `runs/m1a_zem_zev_full_box_seed{0,1,2}`, only on the RTX 5060 machine.

| Failure outcome | Count |
| --- | --- |
| hard_horizontal | 216 |
| tilt_rate_at_contact | 99 |
| hard_vertical | 61 |
| tilt_at_contact | 21 |
| total | 397 |

- 349 failures are "low and fast" starts (below 20 m and faster than 3 m/s down); 48 are
  "other" and unexplained ([[open-issues]]). No start is strictly impossible (ideal bound at most 2.91 m/s).
- Region A (start at 20 m or higher, or slower than 3 m/s): 99.53 %. Region B: 83.6 %.
- Landing miss: median 0.26 m, worst 42.6 m (when the guided phase is cut short).
- Nominal starts: 256 of 256; miss 0.22 m mean, 1.05 m max.

## Reference value for the new machine

After the move to the RTX 5090, rerun seed 0 with 4,096 envs and compare with **96.90 %**. A
small difference is expected (different GPU, PhysX); a large one needs a look.
After the dish-footpad change ([[terrain-integration-plan]]) these numbers will change; report
old and new side by side.

- measured with: [[classical-agents]]
- judged by: [[metrics-and-run-records]]
- speed: [[speed-benchmarks]]
