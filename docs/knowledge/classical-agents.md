---
id: classical-agents
title: Classical agents
type: component
status: current
updated: 2026-11-15
summary: The four-step classical controller chain ending in ZEM/ZEV guidance, its parameters, and how it uses the observed height above ground.
---

# Classical agents (`src/lunarsim/agents/classical/`)

Built and tested in four steps; each step uses the previous one.

| Step | Agent key | File | Result |
| --- | --- | --- | --- |
| 1 | `attitude_hold` | `attitude_hold_agent.py`, `attitude_controller.py`, `jet_selection.py` | Residual tilt 0.39°; 0.13–1.16 kg RCS propellant. Rate dead band 0.2°/s. `JetCoupleTable` picks jet pairs |
| 2 | `hover_hold` | `landing_agents.py`, `vertical_controller.py` | 0.6 mm altitude error over 60 s |
| 3 | `vertical_descent` | `landing_agents.py` | 16 of 16 land; contact speed 0.95–0.99 m/s; engine cut at 0.25 m footpad height |
| 4 | `zem_zev` | `zero_effort_miss_zero_effort_velocity_agent.py`, `zero_effort_guidance.py` | 256 of 256 nominal starts; 96.77 % over the full box ([[milestone-1a-results]]) |
| — | `zero` | `zero_command_agent.py` | All-zero command (milestone 0) |

## ZEM/ZEV design (`configs/agent/zem_zev.yaml`)

- Zero-effort-miss / zero-effort-velocity guidance to a **gate 3 m above the target**; then
  the vertical-descent controller takes over.
- Time-to-go = 1.25 × shortest feasible time + 2 s; planning tilt limit 10°, guided tilt limit
  15°, thrust margin 0.9; the planned path must not dip below the gate height.
- Final descent waits above ground while horizontal speed is 1.0 m/s or more, descends at full
  rate below 0.3 m/s; held upright below 2 m. These rules raised full-box success from 81 % to 97 %.
- Lateral acceleration comes from tilting the lander.
- It has **no glide-slope constraint**: 2 of 16 logged landings leave a 30° cone
  (`docs/figures/glide_slope_constraint_3d.png`).

## Height above ground (since 2026-11-15)

The agents read the 15th observation value (footpad height above the ground straight below)
for the descent profile, the gate switch, "upright below 2 m" and the engine cut-off at 0.25 m.
They no longer assume ground at z = 0. ZEM/ZEV still steers to a gate 3 m above the target in
the SITE frame (z = 0 at the target's ground). The vertical speed is inertial, not the rate of
change of the height above a sloped ground.

- flies on: [[slot-loop-architecture]]
- judged by: [[metrics-and-run-records]]
