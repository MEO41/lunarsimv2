---
id: orbit-to-touchdown-mission
title: Orbit to touchdown mission and terrain-relative navigation (S7, S5, milestone 6)
type: component
status: current
updated: 2026-11-16
summary: Milestone 6 - float64 orbit plant, MCI/MCMF/SITE frames, phase gates and gate logs, Apollo-style quadratic guidance (braking and approach), the handoff to PhysX at the low gate, the camera TRN filter; scenarios S7 (full mission) and S5 (TRN descent); the guidance fixes that made the Apollo 11 timeline come out.
---

# Orbit to touchdown (S7) and terrain-relative navigation (S5)

Design M6: "float64 orbit, frame handoff, camera TRN; S5, S7". The S7 question: can the slot loop
fly the whole Apollo descent, from the parking orbit to touchdown, with state handoffs between
phases that do not jump? The S5 question: does a camera terrain fix hold the navigation error
small enough from the high gate to the low gate?

- depends on: [[slot-loop-architecture]]
- depends on: [[hazard-detection-and-avoidance]]
- uses: [[measured-terrain-surface]]
- uses: [[sensors-and-ekf]]
- uses: [[lander-preset-apollo-lm]]
- sources: [[sources-bibliography]]

## How a mission runs

`scripts/run_mission.py mission=<name> num_envs=N run_name=R` reads `configs/missions/<name>.yaml`
and runs its phases one after the other. Every phase is an ordinary scenario. `metrics:
phase_gate` ends each lander at the phase's gate and saves its state (frame, float64 position,
velocity, attitude, rates, mass, navigation error) in `episodes.parquet`. The next phase starts
from that log (`start: from_log`). At the end `runs/R_mission_summary.yaml` lists, per phase, gate
reach, gate dispersion in the site frame, propellant, time, and the landing results.
`from_phase=` plus `log=` restarts at a later phase from a saved log (for development).

| Phase (S7) | Scenario | Plant, frame | Agent | Ends at |
| --- | --- | --- | --- | --- |
| Descent orbit | `s7_phase_descent_orbit` | `orbit_f64`, MCI | `scripted_burns` (DOI, then attitude hold) | perilune (PDI) |
| Braking (P63) | `s7_phase_braking` | `orbit_f64`, MCMF | `quadratic_guidance_braking` | range 8,334 m (4.5 n.mi.) = high gate |
| Approach (P64) | `s7_phase_approach` | `orbit_f64`, MCMF | `quadratic_guidance_approach` | altitude 152.4 m (500 ft) or range 609.6 m (2,000 ft) = low gate |
| Terminal | `s7_phase_terminal` | `physx_site`, SITE (float32) | `site_pick_divert` (the S6 chain) | touchdown |

S5 (`mission=s5_trn`): `s5_trn_descent` (start `high_gate`, plant `local_f64` in SITE float64 over
the Apollo 17 terrain with an 8 m regional grid out to 10 km, sensors `trn_suite`, nav `trn_ekf`,
the approach guidance) to the low gate, then `s5_trn_terminal` in PhysX.

## Bindings added

| Slot | Key | File | What it does |
| --- | --- | --- | --- |
| plant | `orbit_f64` | `slots/plant/orbit_float64.py` | float64 RK4, point-mass lunar gravity, MCI or MCMF (rotating-frame terms); thrust and RCS from the actuators |
| plant | `local_f64` | same file | the same in the SITE frame (Moon centre one site radius below), for S5 over terrain |
| start | `orbit_elements` | `slots/start/orbit_elements.py` | circular parking orbit at the DOI point, 180 deg from PDI, plane through the site at the landing time ([D-6846] p. 3, 8) |
| start | `from_log` | `slots/start/from_log.py` | the previous phase's gate states; `convert_to: SITE` hands MCMF float64 to the float32 site frame and checks the jump |
| start | `high_gate` | config of `approach_gate` | Apollo high gate for S5 ([D-6846] Table I) |
| nav | `truth_noise` | `slots/navigation/truth_noise.py` | truth plus a per-episode bias and white noise; optional separate local-vertical bias |
| nav | `trn_ekf` | `slots/navigation/terrain_relative_navigation.py` | `ekf` plus camera terrain fixes (below) |
| sensors | `trn_suite` | config; camera in `slots/sensors/shaded_relief_camera.py` | terminal suite plus the shaded-relief camera (`camera_cheap`) |
| agent | `scripted_burns` | `agents/classical/powered_descent_guidance.py` | DOI burn to a 50,000 ft perilune, then attitude hold |
| agent | `quadratic_guidance` | same file | Apollo-style quadratic (constant-jerk) guidance in curved coordinates over the sphere, with the LM throttle logic |
| metrics | `phase_gate` | `slots/metrics/phase_gate.py` | gate conditions (one or more, the first one met ends the phase), gate log, navigation error and filter health at the gate |

Frames are in `core/frames.py` ([[slot-loop-architecture]]): MCI, MCMF (Moon rotation from
NSSDCA, sidereal period 655.720 h) and SITE (east-north-up at the site, float32 near the site).

## Terrain-relative navigation (S5)

Each camera frame: every pixel's ray from the **estimated** pose is marched onto the onboard map
(the true map with the design's map error: 5 m resolution, offset (0, 20) m). The brightness of the
camera image goes into a 48 x 48 ground grid centred on the image's mean ground point, cell size
= the image's ground sampling distance. A reference image is rendered from the onboard map over
the same grid plus a 12-cell search margin, with the same sun. Normalised cross-correlation gives
the shift; a peak above 0.5 is a fix (sigma half a cell per axis) for the EKF position update. The
rays are marched onto the map (not onto a level ground under the lander) because the approach
camera looks far ahead from a pitched lander, over hundreds of metres of relief.

## Guidance: what had to change to fly the Apollo 11 timeline

The braking and approach guidance are Apollo-style ([D-6846] p. 4-10). Seven problems were found
by tracing single landers and fixed; each fix follows the Apollo text where it says something:

1. **One burn from PDI to touchdown** ([D-6846] p. 7). The engine first ignites when the thrust
   axis is within `ignite_within_deg` of the wanted direction; after that it keeps burning while
   the attitude follows. Before, a 10 deg rule shut the engine off whenever the commanded
   direction swung, near the end of braking.
2. **Plan at ignition.** The braking plan's clock starts at ignition (about 25 s after the phase
   starts, for the attitude turn), not at the phase start.
3. **Throttle recovery latch** ([D-6846] p. 8: "when the command decreases to 57 percent, a
   3-percent low bias, the DPS is throttled as commanded"). Without it, commands between 60 and
   92.5 % flipped the thrust between the two every step and with it the direction asked for.
   The approach starts throttled (`start_at_fixed_point: false`).
4. **Front-loaded braking.** A constant downrange deceleration from PDI needs about 3.0 m/s^2,
   more than the fixed-throttle-point thrust gives at the PDI mass (2.86 m/s^2). The plan now starts
   at the full fixed-point deceleration and changes linearly to the gate (constant jerk, which the
   quadratic law flies exactly); T and the gate acceleration follow in closed form.
5. **Short time-to-go hold.** Below `minimum_time_to_go_s` the time-to-go is held. Braking had
   20 s, which brakes too gently before the gate; 5 s works.
6. **Fly through the gate.** With a fixed target that has a speed (low gate: 60 ft/s forward,
   -16 ft/s) and the time-to-go held, the quadratic law has a steady hover v_D T / 2 short of the
   gate (46 m short, 12 m high). After the planned time the target now moves on at its target
   rates, so the lander passes through the gate.
7. **Approach duration from the state.** The approach picks its duration with the ZEM/ZEV
   time-to-go chooser of the terminal phase (`time_to_go_mode: feasible`), so a dispersed high gate
   still gets a flyable plan; at least the steady-deceleration time.

Navigation in the orbit phases is `truth_noise` (50 m bias per axis, model choice; the S7 design
has no navigation design). In the approach the vertical bias is 5 m: Apollo's landing radar had
updated the altitude since 39,000 ft ([D-6846] Table I).

## First results (development runs, 4 landers)

| | This model | Apollo 11 plan ([D-6846] Table I, p. 10) |
| --- | --- | --- |
| High gate | 2,081-2,138 m, 151-154 m/s, -39 to -42 m/s, about 8,331 kg | 7,515 ft (2,291 m) in Table I, "approximately 7000 feet" in the text; 506 ft/s (154 m/s); -145 ft/s (-44 m/s); about 8,433 kg (rocket equation from 5,375 ft/s) |
| Approach | 88-92 s | about 100 s |
| Low gate | 152 m, 18-19 m/s, -4.8 m/s, about 7,610 kg | 500 ft, 60 ft/s, -16 ft/s; about 7,788 kg |
| Terminal | 98-113 s, 403-460 kg | 108 s (10:06 to 11:54) |

The committed 16-lander results are in the next section.

## Committed results (2026-11-16, 16 landers, seed 0, commit `fe6ede0`)

S7 (`results/s7_full_mission_seed0_*`, mission summary in the terminal folder):

| Phase | Time | Propellant | Gate state (mean; 1 sigma) |
| --- | --- | --- | --- |
| Descent orbit (DOI to PDI) | 3,432 s | 139 kg | PDI 489 km uprange |
| Braking | 490 s | 6,245 kg | high gate 2,099 m (sigma 49 m), 151.7 m/s, -42.2 m/s; 8,337 kg |
| Approach | 89 s | 725 kg | low gate 162 m (sigma 9 m), 18.4 m/s, -4.6 m/s; 7,612 kg |
| Terminal (PhysX) | 106 s (95-135) | 435 kg | **16/16 landed**, -1.44 m/s, 0.09 m/s horizontal |

Handoff to PhysX at the low gate: position jump 3e-5 m, velocity jump 8e-7 m/s, energy 8e-12.
Crossrange at the gates is about 71 m (1 sigma), the 50 m navigation bias of the orbit phases.
As in S6, 19 % of the touchdowns are on a site that the true ground judges safe (LiDAR map).

S5 (`results/s5_trn_seed0_*`): TRN descent 90 s and 644 kg from the high gate; navigation error
at the low gate 2.95 m mean (1.3 to 12.7 m) from a 50 m start error; terminal **16/16 landed**,
landing error 0.75 m mean (1.5 m at most). The filter's gate rejected 8 to 164 terrain fixes per
lander (health column `fixes_rejected`).

## Mission video

`runs/s7_full_mission_video_seed0_video/mission.mp4` (not in git, 23 MB): lander 0 of the seed 0
run, recorded with `record=0` (the run repeats the committed S7 numbers exactly); see
[[replay-recording-and-rendering]].

## Limits

- The terminal phase is the classical S6 chain; the design's learned `ppo_gru` for S7 and S5 is
  skipped ([[skipped-learning-work]]).
- Each orbit phase draws a new `truth_noise` bias: the estimate jumps at the phase change (the
  true state does not). Good enough to test guidance under a navigation error; not a navigation
  model of the orbit phases.
- The braking burn starts about 25 s after the phase starts (the attitude turn) and the high gate
  comes out about 100 kg lighter than Apollo's plan.
