---
id: hazard-detection-and-avoidance
title: Hazard detection and avoidance (scenario S6, milestone 4)
type: component
status: current
updated: 2026-11-17
summary: The S6 chain - approach-gate start, pointed flash LiDAR, onboard hazard map, LM footprint criteria, site_pick_divert agent, site_safety metric - the first results (31 % safe touchdowns with the LiDAR map, 83 % with the true map) and the LiDAR comparison (94 % with the ASC LiDAR's measured noise; the 83 % 'upper limit' is set by the agent's landing accuracy, not by the map).
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
- ~~The gap between 31 % and 83 % is the onboard map~~ (corrected 2026-11-17, next section: the
  two runs also fly differently; the true-map landers dive to a site chosen at 80 m and land
  0.8 m off it, which caps them near 83 %).

## LiDAR comparison (2026-11-17, 64 landers, seed 0, code `2c535a7`)

User: "S6: close the LiDAR gap". The LiDAR noise above, 10 cm + 1 % of range, is the ASC
paper's **specification** ("Range Error: < (10 cm + 1% of range)", [ASC-FLASH] p. 4), used as one
sigma. The same paper's Table 1 (p. 5) gives the **measured** RMS range error of the two OSIRIS-REx
flight units: 0.30 / 0.34 m at 354 m and 0.26 / 0.77 m at 1,009 m, where the specification
allows 3.64 m and 10.2 m. New: `range_noise_table_m` in the pointed LiDAR (interpolated measured
errors), the sensor suites `hazard_detection_suite_measured_noise` (worse of the two units per
range) and `hazard_detection_suite_alhat` (ALHAT flash LiDAR flown on Morpheus, [ALHAT-2016]
Table 1: 8 cm 1 sigma, 1 deg, 20 Hz, 150 m to 1.3 km; 8 x 8 mosaic of 1 deg steps, model choice),
scenarios `s6_hazard_avoidance_measured_lidar` and `s6_hazard_avoidance_alhat_lidar`, and
`scripts/compare_hazard_lidar_runs.py` (table in `results/s6_lidar_comparison/`). The agent is
unchanged in every run. The same seed gives every lander the same start in every run, so runs are
also compared lander by lander. `s6_nominal_target_seed0` switches site selection off
(`agent.decision_altitude_m=-100`) to give each lander's propellant without a divert.

| Run | Safe touchdown % (95 % Wilson) | Paired vs specification (+ / -) | Safe site at 80 m | Touchdown s | Horizontal speed at touchdown m/s | Miss from chosen site m | Divert propellant kg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nominal target, no site selection | 7.8 (3.4-17.0); 86 % landed | +0 / -15 | - | 69.3 | 0.42 | 0.76 | 0 |
| ASC specification, 32 x 32 (rerun of the first result, identical) | 31.2 (21.2-43.4) | - | 0 % | 104.8 | 0.07 | 0.18 | 138 |
| ASC specification, 128 x 128 | 53.1 (41.1-64.8) | +21 / -7 | 0 % | 110.0 | 0.07 | 0.20 | 158 |
| **ASC measured, 32 x 32** | **93.8 (85.0-97.5)** | +42 / -2 | 0 % | 101.2 | 0.12 | 0.32 | 124 |
| ASC measured, 128 x 128 | 71.9 (59.9-81.4); 1 tilt-rate failure | +29 / -3 | 95 % | 74.8 | 0.32 | 0.80 | 18 |
| ALHAT, 32 x 32 | 81.2 (70.0-88.9) | +36 / -4 | 100 % | 69.8 | 0.36 | 0.91 | -2 |
| true map and true state | 82.8 (71.8-90.1) | +37 / -4 | 100 % | 69.6 | 0.39 | 0.79 | -3 |

"Divert propellant" is each lander's propellant minus its own propellant in the nominal-target
run. With site selection every unsafe touchdown fails on relief (rocks), one ASC-measured
128 x 128 touchdown also on slope; at the nominal targets 58 fail on relief and 7 on slope. Wall
time at 64 landers: 170 s with 32 x 32 pixels, 236 s with 128 x 128 (RTX 5060).

What the runs show:

- **The noise figure decides the map.** With the measured noise the map judges sites correctly
  (31 % to 94 % safe with the same pixels); all 128 x 128 pixels with the specification noise give
  only 53 %.
- **Two flight patterns, set by when the map first finds a safe site.** When a safe site is known
  at the first decision (80 m footpad height: true map, ALHAT, measured noise with 128 x 128),
  the agent re-plans to it with the shortest feasible time-to-go (1.25 x the minimum + 2 s) and
  lands in about 70 s, 0.3-0.4 m/s sideways and 0.8-0.9 m from the chosen site. The nominal
  flight has the same pattern (69 s, 0.76 m). When sites appear only lower down, the late
  redesignations end in a hover until the lander is steady: about 105 s, 0.07-0.12 m/s,
  0.2-0.3 m from the site, and 120-160 kg more propellant (of 810 kg at the low gate).
- **The 83 % of the true map is therefore not an upper limit of the map.** Its 11 unsafe
  landers all missed their chosen site by 0.5-1.3 m; a 0.8 m miss moves the footpads onto rocks
  next to a site judged safe for an exact touchdown. The unsafe landers with the measured noise
  (32 x 32) missed by only 0.04-0.15 m: those are map errors.
- What limits S6 now is the agent's landing accuracy at the chosen site (and the footprint check,
  which assumes an exact touchdown), not the LiDAR. Fixes are proposed to the user
  ([[pending-user-decisions]]); the default S6 sensor stays the specification noise until the
  user decides.

Speed: 22 ms per control step at 16 landers on the RTX 5060 after compiling the LiDAR march and
the altimeter and Doppler ray casts (39 ms before; the Newton ray casts alone took 6 ms per sensor
per step on the two-level terrain, which also speeds up S2 and S3 on Apollo 17).

- part of: [[milestone-roadmap-and-status]]
- uses: [[measured-terrain-surface]], [[sensors-and-ekf]], [[classical-agents]]
