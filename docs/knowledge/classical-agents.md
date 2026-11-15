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
| 3 | `vertical_descent` | `landing_agents.py` | 16 of 16 land; contact speed 0.95–0.99 m/s with the old 0.25 m cut; 1.60–1.63 m/s with the probe-mode cut (2026-11-15) |
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
for the descent profile, the gate switch, "upright below 2 m" and the backup cut-off.
They no longer assume ground at z = 0. ZEM/ZEV still steers to a gate 3 m above the target in
the SITE frame (z = 0 at the target's ground). The vertical speed is inertial, not the rate of
change of the height above a sloped ground.

## Engine cut-off (since 2026-11-15: Apollo probe mode)

`ClassicalLandingAgentBase.update_engine_cutoff`, parameters in `configs/agent/zem_zev.yaml`
and `vertical_descent.yaml`:

- `engine_cutoff_signal: surface_probe_contact` (default): the engine stops
  `engine_stop_delay_after_probe_contact_s` (1.0 s, model choice) after the contact light
  (16th observation value) first comes on, **once the lander is steady** (rate below 1°/s,
  tilt below 3°). A lander that is still turning keeps descending ("pad mode", the Apollo
  backup, Rogers 1973 p. 125) and stops when the height below falls under
  `backup_engine_cutoff_height_m` (0.0 m). The jets close with the engine.
- `engine_cutoff_signal: height_above_ground`: the old rule, stop below
  `vertical.engine_cutoff_altitude_m` (0.25 m).
- Why the steadiness rule: without it, low-and-fast starts reached probe height while still
  manoeuvring and hit the ground turning at about 4°/s (limit 2°/s): 84 % flat success
  instead of 97.7 % (1,024 landings, seed 0).
- Effect: about 85 % of landings stop on the probe and touch down at about 1.6 m/s
  (5.3 ft/s; Apollo 15 was about 6.8 ft/s); the rest hold the engine longer and touch down
  slower.

- flies on: [[slot-loop-architecture]]
- judged by: [[metrics-and-run-records]]
