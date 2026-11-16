---
id: hazard-detection-and-avoidance
title: Hazard detection and avoidance (scenario S6, milestone 4)
type: component
status: current
updated: 2026-11-16
summary: The S6 chain - approach-gate start, pointed flash LiDAR, onboard hazard map, LM footprint criteria, site_pick_divert agent, site_safety metric - and the first results (31 % safe touchdowns with the LiDAR map, 83 % with the true map, 8 % at the nominal targets).
---

# Hazard detection and avoidance (S6)

Design S6: "Can the lander find a safe site from LiDAR data and divert to it with the fuel it has
left?" Built 2026-11-16 without any learning ([[skipped-learning-work]]).

## The chain

| Part | Binding / file | What it does |
| --- | --- | --- |
| Start | `start: approach_gate`, `slots/start/approach_gate.py` | Apollo low gate ([D-6846] Bennett, NASA TN D-6846 p. 10, Table I): 610 m behind the target, 152 m up, 18.3 m/s forward, -4.9 m/s, thrust axis 16 deg back; about +-5 % ranges (model choice) |
| LiDAR | `sensors: terminal_suite` + `lidar.pointing`, `slots/sensors/pointed_flash_lidar.py` | Flash LiDAR (ASC numbers, 32 x 32 of 128 x 128, 6 deg, 10 Hz, 10 cm + 1 % of range) on an ideal pointing mount aimed at the nominal target, 3 x 3 mosaic of 5 deg steps, 0.05 deg pointing error. Writes ranges and the boresight (`measurement.lidar_pointing`) |
| Ray cast | `march_rays_to_ground` | First crossing for low-angle rays: adaptive steps `gap / (down + 2 horizontal)` (ground up to 63 deg trusted), at least 0.5 m, then 12 bisections; compiled on the GPU (7.1 -> 0.7 ms per 16k-ray frame) |
| Map | `nav: hazard_map`, `slots/navigation/hazard_map_navigation.py`, `onboard_hazard_map.py` | `ekf` plus an 80 m x 80 m grid of 0.5 m cells around the nominal target; each return becomes a point through the filter's own pose and is fused with weight 1/sigma^2; a cell is known when its standard error is at most 0.15 m; every 1 s each 2 m candidate is judged |
| Criteria | `terrain/hazard_assessment.py` | LM design surface [D-6850] Rogers, NASA TN D-6850 p. 3: mean slope <= 6 deg, effective slope (plane through the 4 pad heights, worst of 4 headings) <= 12 deg, relief about the mean plane <= 24 in (0.61 m) within the footprint (radius 4.73 m); half the disk known. Same function on the onboard map and on the true ground |
| Agent | `agent: site_pick_divert`, `agents/classical/site_pick_divert_agent.py` | ZEM/ZEV with per-lander targets: picks the best safe candidate within 30 m below 80 m footpad height, may redesignate until 20 m when a site is better by 0.1 margin or the current one becomes unsafe; one decision record per episode |
| Metric | `metrics: site_safety`, `slots/metrics/site_safety.py` | `touchdown` plus the true footprint check at first contact and at the nominal target, and the divert distance |
| Upper limit | `nav: truth_with_hazard_map` | True state and a map filled from the true ground (design: "Set nav: truth with the true hazard map") |

Scenario files: `s6_hazard_avoidance.yaml` (Apollo 17 terrain, outer grid 800 m, rocks rescaled to
cover 5 % of the ground: design `rock_density` in [0, 0.1]; the measured Surveyor law alone covers
0.37 %; episode 180 s; low-gate mass 7,788 kg and 810 kg propellant derived from the Apollo 11
touchdown mass and the landing-phase delta-v of [D-6846] Table I) and
`s6_hazard_avoidance_truth_map.yaml`.

## First results (2026-11-16, 64 landers, seed 0, commit after `f3e2119`)

| Run | Landing success | Safe touchdown (true ground) | Nominal target safe | Mean divert |
| --- | --- | --- | --- | --- |
| `s6_hazard_map_check` (LiDAR map) | 100 % | 31.2 % | 7.8 % | 5.6 m |
| `s6_truth_map_check` (true map) | 100 % | 82.8 % | 7.8 % | 18.7 m |

- Mean landing phase 108 s from low gate, close to the 108 s of the Apollo 11 plan
  ([D-6846] Table I: low gate 10:06, touchdown 11:54).
- With the LiDAR map no candidate is judged safe at 80 m (every first decision found none); sites
  appear as the lander comes closer (2.4 redesignations per landing on average). The 1 %-of-range
  noise makes cells precise only below about 100 m.
- Every unsafe touchdown, with either map, fails on relief (rocks), not on slope.
- The gap between 31 % and 83 % is the onboard map; the gap from 83 % to 100 % is landing accuracy
  and the 2 m candidate grid against the 0.25 m true ground.

Speed: 22 ms per control step at 16 landers on the RTX 5060 after compiling the LiDAR march and
the altimeter and Doppler ray casts (39 ms before; the Newton ray casts alone took 6 ms per sensor
per step on the two-level terrain, which also speeds up S2 and S3 on Apollo 17).

- part of: [[milestone-roadmap-and-status]]
- uses: [[measured-terrain-surface]], [[sensors-and-ekf]], [[classical-agents]]
