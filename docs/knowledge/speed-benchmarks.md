---
id: speed-benchmarks
title: Speed benchmarks
type: result
status: current
updated: 2026-11-14
summary: Environment steps per second and GPU memory against environment count, per GPU (goal G4).
---

# Speed benchmarks (goal G4)

Measured with `scripts/benchmark_environment.py`, scenario S1 with `zem_zev` in the loop.

## RTX 5060, 8 GB (2026-11-14)

| Environments | Env steps per second | GPU memory |
| --- | --- | --- |
| 64 | 5,785 | |
| 256 | 21,433 | |
| 1,024 | 69,646 | |
| 4,096 | 159,733 | |
| 8,192 | 190,096 | |
| 16,384 | 203,702 | |

GPU memory ranged from 2.33 GB to 5.07 GB over these runs. Milestone 0 null loop: about
151,000 env steps/s at 1,024 envs.

## RTX 5090, 32 GB

Not measured yet. To do after the setup ([[workstation-setup-and-machines]]): the same table,
extended to 32,768 and 65,536 environments. No number is promised before it is measured.

- part of: [[milestone-1a-results]]
