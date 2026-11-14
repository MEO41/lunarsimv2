---
id: decisions-log
title: Decisions log
type: decision
status: current
updated: 2026-11-15
summary: Every decision taken so far, with date and reason; the place to add new ones.
---

# Decisions log

Newest at the bottom. "User" means the user decided; the others were proposed, explained and accepted.

| Date | Decision | Why |
| --- | --- | --- |
| 2026-11-14 | Build v2 from scratch; old LunarSim repo off limits | User ([[rules-from-the-user]]) |
| 2026-11-14 | Repo MEO41/lunarsimv2 (private), package `lunarsim`, BSD-3-Clause | User, open decisions of the design |
| 2026-11-14 | Apollo LM is the first lander preset | User |
| 2026-11-14 | Pin Isaac Sim 6.0.1 (pip) + Isaac Lab 3.0.0 develop@e86463d, PhysX backend | What runs on the machine; stock tasks default to Newton |
| 2026-11-14 | Forces through the permanent wrench composer | `set_external_force_and_torque` is deprecated and only forwards to it |
| 2026-11-14 | Plant state read twice per control step (hook 3 and hook 6) | Metrics must see the state after physics; observations after resets |
| 2026-11-14 | Registry keys and scenario file names as in the design; descriptive Python names and message fields | CLAUDE.md rule 1, user approved |
| 2026-11-14 | Architecture doc wins over the Scenario Bindings doc | User |
| 2026-11-14 | Package in `src/lunarsim`; setuptools build backend | Isaac Lab template layout; `uv` not installed |
| 2026-11-14 | `Wrench` gets `mass_flow_rate` | Plant must know propellant flow |
| 2026-11-14 | Truth access for start, plant, surface, sensors, metrics, and `nav: truth` | Design diagram feeds truth to sensors and surface |
| 2026-11-14 | Message `start_state` from start slot to plant | `reset()` has no bus |
| 2026-11-14 | Body origin = whole-vehicle centre of mass | Found by the RCS torque test |
| 2026-11-14 | Timing 5 ms PhysX / 20 ms control; action 19 values; observation 14 values; gimbal accepted but unused | Design plus M1a needs |
| 2026-11-14 | PGS solver, 16 position iterations | TGS softened the strut by 22 % ([[physx-contact-settings]]) |
| 2026-11-14 | Engine 10–65 % throttle, lag 0.08 s, slew 85 %/s; fixed point not modelled | TN D-7143, Klumpp |
| 2026-11-14 | `zem_zev` flies to a 3 m gate, then vertical descent; waits while horizontal speed is high | Full-box success 81 % → 97 % ([[classical-agents]]) |
| 2026-11-14 | `surface: procedural` smoke-tested only | Agents assume ground at z = 0 |
| 2026-11-14 | Visual struts in the lander USD (no mass, no collision) | Pictures; GPU tests rerun |
| 2026-11-14 | Stop at the M1a gate; M1b not started | User |
| 2026-11-14 | Pinned versions in `requirements-pinned.txt`; `scripts/setup_workstation.sh` and `scripts/check_installation.py` | User asked for the move to the RTX 5090 |
| 2026-11-14 | CPU-test workflow on GitHub Actions | After the user added the `workflow` token scope; first run passed |
| 2026-11-14 | **No NASA lander mesh in simulation**; keep our lander, match LM footpads, c.g. and geometry | User: the mesh is too slow |
| 2026-11-14 | **Ground must be real terrain** with craters, rocks and slopes | User |
| 2026-11-14 | **Height maps are not committed**; users download them into `assets/terrain/apollo17/` | User |
| 2026-11-14 | Keep existing preset numbers; change lander values only with a public source | User |
| 2026-11-14 | Project memory in `docs/knowledge/` (this graph), checked in CI | User asked for context that survives machine changes ([[knowledge-graph-tooling]]) |
| 2026-11-15 | User said "proceed" to [[terrain-integration-plan]] | User |
| 2026-11-15 | **Observation grows to 15 values**: footpad height above the ground straight below the c.g. | User chose it; agents need it on uneven ground |
| 2026-11-15 | Iyy = 13,867 slug ft² (18,801 kg m²), not 13,667 | Page image of Mission Report Table A-I |
| 2026-11-15 | RCS clusters at 1.679 m along each body axis, horizontal jets at R-567 offsets | MIT R-567; the M1a reading of Cox gave 29 % too little torque |
| 2026-11-15 | Lander gets c.g. offset, dish footpads (0.940 × 0.178 m), descent and ascent stage shapes; nozzle without collision (crushable), clearance recorded | User: match the real LM; sources in the preset |
| 2026-11-15 | Plant applies actuator forces at the vehicle c.g. (offset = minus the root body's c.g.) | Isaac Lab's composer measures offsets from the body c.g.; with the c.g. offset the old zero offset tilted the hover |
| 2026-11-15 | New message `SurfaceGeometry`: per-env site origin and ground lookup; surface resets before the plant | Site moves per reset on real terrain; the plant needs the ground for the height |
| 2026-11-15 | Terrain = measured 2 m Apollo 17 map (±600 m) + 0.25 m detail patch (±80 m) with craters (Minton 2019 law) and rocks (Surveyor 1 law); targets uniform ±50 m | [[measured-terrain-surface]] |
| 2026-11-15 | Height maps read with numpy from the raw `.TIF`; downloaded by `scripts/download_height_maps.py` into git-ignored `assets/terrain/apollo17/` | No new package in the Isaac environment; user: not in git |
| 2026-11-15 | Site centre = Apollo 17 LM position 20.19091° N, 30.77168° E | Haase et al. 2019, Table 3 |
| 2026-11-15 | New outcome `body_contact`; records nozzle clearance, footprint slope and relief | Real terrain needs strike and site metrics |
| 2026-11-15 | Hover-hold tilt tolerance 0.6° → 1.0° | Longer R-567 jet arms make the minimum-pulse cycle larger (measured 0.65°) |
| 2026-11-15 | GPU test runner cleans dead-process `/dev/shm` files and restarts a silent launch (unbuffered child output) | Intermittent launch stalls ([[isaac-lab-quirks]]) |

Decisions still open: [[pending-user-decisions]].
