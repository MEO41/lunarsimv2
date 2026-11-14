---
id: physx-contact-settings
title: PhysX contact settings
type: component
status: current
updated: 2026-11-15
summary: Solver and contact settings from the contact-rest study, why PGS and not TGS, and the measured residuals.
---

# PhysX contact settings

File: `configs/plant/physx_site.yaml`; applied in `env/lander_env_cfg.py` (scene settings from
the plant's `physx_scene` block plus `env.physx`).

| Setting | Value |
| --- | --- |
| Solver | **PGS** (`solver_type: 0`) |
| Position / velocity iterations | 16 / 1 |
| `enable_external_forces_every_iteration` | false (only supported by TGS) |
| Contact offset / rest offset | 0.02 m / 0 |
| Max depenetration velocity | 1 m/s |
| Body damping | zero |
| Friction / restitution | 0.4 / 0 (surface binding); the lander's default material has 1.0 and "multiply", so 0.4 is what acts (before 2026-11-15 it was 0.2) |
| Bounce threshold | 0.5 m/s |

## Why PGS

The default TGS solver made the strut spring act like 0.78 k at rest. The measured softening
fits k/(1 + c·dt/(n·m_leg)): an artefact of TGS's implicit joint damper with light leg links.
PGS with 16 position iterations gives 0.998 k at the same speed.

## Measured residuals

- About 10 µm strut vibration at rest, up to 5 mm/s velocity noise (matters for the IMU in M2).
- Drift below 0.25 mm in 10 s.
- Free fall: PGS is semi-implicit Euler; 8.1 mm below the parabola after 2 s, 0.1 mm from the
  integrator's own formula.

The new dish footpads of [[terrain-integration-plan]] change the contact, so this study must be
rerun after that change.

- used by: [[lander-articulation-model]]
- recorded in: [[decisions-log]]
