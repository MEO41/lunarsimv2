---
id: decisions-log
title: Decisions log
type: decision
status: current
updated: 2026-11-14
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

Decisions still open: [[pending-user-decisions]].
