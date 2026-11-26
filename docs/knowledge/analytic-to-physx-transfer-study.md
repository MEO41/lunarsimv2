---
id: analytic-to-physx-transfer-study
title: Analytic-to-PhysX transfer study
type: plan
status: current
updated: 2026-11-26
summary: Revision of the rejected "Speed versus Fidelity" paper as an RL transfer study on LunarSim v2 - question, tests T1-T9, decisions, machine.
---

# Analytic-to-PhysX transfer study

A study by a second user, with the repository owner's permission (2026-11-20). It revises
a paper that compared an analytic simulator with PhysX for one hand-tuned controller and was
rejected before review. The revision is a reinforcement-learning transfer study on LunarSim v2.

**Question:** Can a landing policy that is trained in a fast analytic simulator land as well in
full rigid-body physics? How much wall-clock time does the analytic simulator save?

"A" = [[analytic-simulator]] (`plant: analytic_rigid_body`), "P" = PhysX (`plant: physx_site`).
Both run S1 (flat ground, true-state navigation, Apollo LM, `engine_rcs_pulse`, `touchdown`).

## Tests

| Test | What | Status |
| --- | --- | --- |
| T1 | Analytic model check: the PhysX validation tests' known answers on A | **Passes** (2026-11-20; CPU and on `noron`) |
| T2 | Throughput of A and P with the same agent on the same GPU, 1 to 16,384 envs (RTX 5060) | Done (second round, below): A 1.6× P at 1 env to 7.1× (ZEM/ZEV) and 9.4× (PPO) at 16,384 |
| T3 | Paired flights: same 4,096 starts in A and P, same agent; McNemar test of the outcomes | ZEM/ZEV on the S1 box: same outcome for all 4,096 starts (2026-11-20). S1 box and wide range done (second round, below); gate policies 99.7-99.8 % same outcome |
| T4 | Parameter sweep (Latin hypercube): mass ±30 %, vehicle cases, altitude, heading, time step | Later |
| T5 | PPO trained in A and in P, 5 seeds each; each tested in A and P on 4,096 held-out starts | **Done, 10 seeds** (2026-11-21, below): no transfer gap (+0.00 points, 95 % CI -0.33 to +0.29) |
| T6 | Domain randomization in A, tested in P | Later |
| T7 | Wall-clock time to 90 % success in P; also "A first, then P" (`resume_from`) | **Done** (2026-11-21, below): 90 % in P after 362 s (P-trained) and 110 s (A-trained), 3.3×; "A first, then P" left out (no gap to close) |
| T8 | Model ablation: spring legs → `metrics.end_at_first_contact=true` → no engine lag / jet limits → point mass | **Done** (b1-b3 2026-11-22, b4-b5 2026-11-23, below): transfer breaks with ideal actuators and with the point mass; lag plus body rate recovers 90.7 %. Body-rate controls, tau sweep and gate: revision round 1 (2026-11-25, below) |
| T9 | All policies in P on the Apollo 17 terrain | Later (height maps must be downloaded on `noron`) |

## Decisions (user, 2026-11-20)

- The main A has spring-strut legs, so both simulators judge landings with the same outcome
  codes (settling, tip-over). "End at first contact" is a step of T8, between the spring legs
  and the point mass (a point mass has no legs, so it needs that rule anyway).
- Code is written on the user's Windows machine and pushed to `noron`; all GPU work and training
  run on `noron`, only on `/media/noron/DISK02` (repository at
  `/media/noron/DISK02/Speed_Versus_Fidelity/lunarsimv2`, branch `analytic-simulator`).

## Heading-keeping attitude controller (2026-11-20)

The classical attitude loop (`agents/classical/attitude_controller.py`, also used by `ppo_mlp`
with `attitude_commands`, by ZEM/ZEV, powered descent guidance and the build-up agents) aimed at
heading 0. With `start.heading_deg=[0,360]` the lander turned back at up to 5 deg/s, and when that
turn was still running at contact the body rate broke the 2 deg/s limit (9 of 64 wide-range
ZEM/ZEV starts on the CPU; about 11 % of the wide-range T3 starts on `noron`). The user decided
(2026-11-20) that the controller keeps the heading the lander has:
`quaternion_for_thrust_direction(direction, current_orientation)` turns the current attitude by
the shortest rotation that brings body +z onto the wanted direction, so no turn about the thrust
axis is commanded. Start states (`orbit_elements`) still use heading 0. Check: ZEM/ZEV, analytic,
CPU, 256 starts of seed 1000: 98.44 % with and without the heading range.

All results below "first round" were made with the old controller and are kept in
`runs/_superseded_heading_zero_controller/` on `noron`; the gate, T3 and T2 were run again
with the new controller (second round, below).

### Second round (heading-keeping controller, commit 121915c, `noron`, 2026-11-20)

- T5 gate, seed 0, 4,096 envs, 600 iterations: training wall time **PhysX 2,390 s, analytic
  801 s** (3.0×). On the 4,096 held-out starts of seed 1000:

| Policy trained in | Flown in P | Flown in A | Same outcome code | Discordant pairs (only A / only P) | McNemar p |
| --- | --- | --- | --- | --- | --- |
| P | 98.02 % | 98.05 % | 99.83 % | 4 / 3 | 1 |
| A | 97.78 % | 97.80 % | 99.73 % | 6 / 5 | 1 |

  The analytic-trained policy flies 97.78 % in PhysX against 98.02 % for the PhysX-trained one
  (0.24 points), after a third of the training time. One seed only: the change from the first
  round (98.32 % to 98.02 %) cannot be told apart from seed noise; T5 with 5 seeds decides.
- T3, ZEM/ZEV, 4,096 starts each:

| Starts | P | A | Same outcome code | McNemar p |
| --- | --- | --- | --- | --- |
| S1 box, seed 1000 | 97.71 % | 97.68 % | 99.98 % | 1 |
| Wide (5-150 m, heading 0-360°), seed 1000 | 99.44 % | 99.44 % | 100 % | 1 |
| Wide, seed 1001 | 99.46 % | 99.46 % | 100 % | 1 |
| Wide, seed 1002 | 99.15 % | 99.15 % | 100 % | 1 |

  With the heading kept, the wide-range success rose from about 88 % (first round) to about
  99.4 %.
- T2, median of 5 repeats, env steps/s (`runs/t2_throughput/benchmark.jsonl`):

| Envs | 1 | 16 | 256 | 1,024 | 4,096 | 8,192 | 16,384 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P, ZEM/ZEV | 96 | 1,417 | 20,815 | 70,199 | 169,125 | 197,515 | 215,328 |
| A, ZEM/ZEV | 153 | 2,404 | 37,233 | 144,271 | 536,656 | 962,504 | 1,532,100 |
| A / P, ZEM/ZEV | 1.6 | 1.7 | 1.8 | 2.1 | 3.2 | 4.9 | 7.1 |
| A / P, PPO policy | 1.6 | 1.7 | 1.8 | 2.2 | 3.5 | 5.5 | 9.4 |

  The analytic repeats spread less than 3 % around the median.

### First round (old controller, heading 0; commit 48b7150 / 64a71db, `noron`)

- T5 gate, seed 0, 4,096 envs, 600 iterations: training wall time **PhysX 2,379 s, analytic
  808 s** (2.9×). On the 4,096 held-out starts of seed 1000:

| Policy trained in | Flown in P | Flown in A | Discordant pairs (only A / only P) | McNemar p |
| --- | --- | --- | --- | --- |
| P | 98.32 % | 98.24 % | 0 / 3 | 0.25 |
| A | 98.27 % | 98.36 % | 4 / 0 | 0.125 |

  So the analytic-trained policy flies 98.27 % in PhysX against 98.32 % for the PhysX-trained
  one; same outcome code for 99.9 % of the pairs.
- T3 wide range (ZEM/ZEV, altitude 5-150 m, heading 0-360°, seeds 1000-1002): A and P
  88.04/87.99 %, 88.55/88.55 %, 87.43/87.40 %; McNemar p 0.625, 1, 1; the failures are mostly
  `tilt_rate_at_contact` from the heading turn (fixed above).
- T2, median of 5 repeats, env steps/s, ZEM/ZEV in the loop (PPO in the loop is 2-36 % faster
  for A and 1-8 % for P):

| Envs | 1 | 16 | 256 | 1,024 | 4,096 | 8,192 | 16,384 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P | 94 | 1,394 | 20,339 | 66,821 | 156,275 | 185,136 | 205,879 |
| A | 154 | 2,409 | 37,102 | 145,069 | 538,700 | 966,726 | 1,527,010 |
| A / P | 1.6 | 1.7 | 1.8 | 2.2 | 3.4 | 5.2 | 7.4 |

## T5 over 10 seeds (heading-keeping controller, commit 23a9fba, `noron`, 2026-11-21)

`scripts/noron/transfer_study_seeds.sh` ran the gate for seeds 1-9 overnight (seed 0 from the
second round); all 20 trainings reached 600 iterations, no warnings. Every policy flew the same
4,096 held-out starts (seed 1000) in both simulators. Analysis: `runs/_analysis/t5_seed_summary.py`
and `.txt` on `noron`.

| Trained in → flown in | Mean success, 10 seeds | sd | 95 % CI (t) |
| --- | --- | --- | --- |
| P → P | 97.93 % | 0.25 | 97.75-98.12 |
| P → A | 97.94 % | 0.26 | 97.75-98.12 |
| A → P | 97.94 % | 0.47 | 97.60-98.27 |
| A → A | 97.93 % | 0.45 | 97.61-98.26 |

- Transfer gap in P (A-trained minus P-trained): **+0.00 points**; Welch p = 0.99,
  Mann-Whitney p = 0.68; bootstrap 95 % CI -0.33 to +0.29 points; equivalent within ±1 point
  (TOST p < 0.0001). The A-trained policies spread more between seeds (sd 0.47 against 0.25;
  seed 2 is the low one at 96.78 %).
- Simulator gap (same policy in A and P): mean +0.002 and -0.005 points; same outcome code for
  99.81 % and 99.84 % of the pairs; discordant pairs over 10 × 4,096 balanced (38/37 and 29/31).
- Training wall time: P 2,318 s (sd 28), A 798 s (sd 3), 2.9×. 90 % success in the training
  simulator after 403 s (P) and 128 s (A), 3.1× (first iteration at or above 90 %, mean of 10
  seeds; corrected 2026-11-21 from 393/125 s, which counted iterations from the first logged line); reached in 10/10 for both. (T7 still needs the
  checkpoints flown in P.)
- Reading: on flat ground with this action space the analytic model is enough: policies trained
  in it lose nothing in PhysX. The speed-up of training (2.9×) is below the environment speed-up
  at 4,096 envs (3.4×) because the PPO update does not get faster.

## T8 design (2026-11-21)

PPO is trained in ever simpler analytic models (seeds 0-4) and every policy is flown in full
PhysX with the standard success rule. Step 0 is T5.

| Step | Training model (all analytic) | Overrides |
| --- | --- | --- |
| b1_contact | spring-leg rigid body, episode ends at the first contact | `metrics.end_at_first_contact=true` |
| b2_ideal | b1 + ideal actuators: no thrust lag, throttle slew limit, shutdown tail-off, jet minimum pulse or valve delay (throttle range, thrust and propellant use stay) | `+ actuators.ideal_response=true` |
| b3_point_mass | point mass: thrust along the commanded direction at once (heading kept), no rotation dynamics, no legs, stops on the ground; ideal actuators; ends at contact | `plant=analytic_point_mass` + the two above |

- Point-mass attitude: user decision 2026-11-21, option A (at once). A first-order lag of the
  thrust direction (option B) is kept as an extra step if b3 breaks the transfer.
- The Command carries jet duties, not the direction, so the PPO action adapter keeps its last
  commanded thrust direction and the training wrapper hands it to an environment that asks for
  it (`accepts_commanded_thrust_direction`). Only training does this; the point-mass plant
  stops with an error in any other use. PhysX behaviour is unchanged (the T7 queue, started on
  commit d0044b6, ran on f8123f6 from about 15:50; the PhysX path is the same).
- Tests: `tests/unit/test_analytic_model_ablation.py` (8, CPU).

## T7 results: checkpoints flown in PhysX (commit f8123f6, `noron`, done 2026-11-21 23:38)

Every 25th checkpoint (0-575) of the 20 T5 trainings was flown in P on the same 4,096 held-out
starts (480 evaluations, no warnings; iteration 600 is the T5 result). Time = learning wall clock
of the first checkpoint at or above the threshold (`checkpoint_wall_clock.csv`); the 25-iteration
grid is about 97 s in P and 33 s in A, so the times are quantized. Table:
`runs/_analysis/export/t7_success_in_physx_by_checkpoint.csv`.

| Success in P reached | P-trained (10 seeds) | A-trained (10 seeds) | Ratio P/A (bootstrap 95 % CI) | Welch p |
| --- | --- | --- | --- | --- |
| 50 % | 278 s (CI 218-337), iteration 75 | 97 s (85-110), iteration 78 | 2.85 (2.32-3.51) | 6e-5 |
| 90 % | 362 s (300-425), iteration 98 | 110 s (91-130), iteration 88 | 3.28 (2.68-4.00) | 4e-6 |
| 95 % | 411 s (310-512), iteration 110 | 121 s (92-149), iteration 95 | 3.41 (2.58-4.49) | 8e-5 |

- Reached by 10/10 seeds in both simulators at every threshold.
- The A-trained policies are as good in P at every stage of training, not only at the end: per
  iteration they reach the thresholds at about the same iteration (88 against 98 for 90 %), so the
  wall-clock saving is the per-iteration speed-up. Median success in P at iteration 75: A 96.0 %,
  P 80.4 %; from iteration 100 on both are within about 1 point (96-98 %).

## T8 results: model ablation flown in PhysX (commit f8123f6, `noron`, done 2026-11-22 03:05)

All 15 trainings reached 600 iterations (training success 97.8-99.9 %); each policy flew the same
4,096 held-out starts in P. "Judged at first contact" counts `not_settled` as a landing (inside
every limit at contact, still moving when the settle time ran out). Per seed with outcome shares:
`runs/_analysis/export/t8_ablation_in_physx.csv`; script `runs/_analysis/t8_summary.py`.

| Step (5 seeds) | Success in P, full rule | Judged at first contact | Main failures |
| --- | --- | --- | --- |
| T5 full analytic (seeds 0-4) | 97.8 % (CI 97.0-98.6) | 97.8 % | hard horizontal 1.3 % |
| b1_contact | 88.2 % (76.7-99.8) | 97.5 % (96.7-98.4) | not settled 0.3-20.8 % |
| b2_ideal | 55.8 % (35.9-75.8) | 77.1 % (67.0-87.3) | hard vertical 5-16 %, hard horizontal 5-14 %, not settled 6-58 % |
| b3_point_mass | 0.0 % (all seeds) | 4.5 % | propellant depleted / out of bounds (seeds 0, 1, 3: 90-96 %), hard vertical (seed 2: 53 %) |

- **b1: the loss is after contact, not at it.** The policy never saw a post-contact state, so it
  throttles up at contact (trajectory: throttle 0.65 at first leg force) and hops; judged at the
  first contact it is as good as T5 (97.5 against 97.8 %). The episode design, not the touchdown
  physics, is what b1 changes.
- **b2: actuator dynamics matter.** With instant throttle and jets the policy descends faster
  (median contact speed 1.41 against 1.10 m/s for b1) and the lagging PhysX engine cannot brake in
  time: hard landings rise from about 2 % to about 23 %. Judged at contact, -20 points against b1.
- **b3: rotation dynamics are essential.** The policy expects its thrust direction at once; in P
  the body lags, the tilt grows (to 15° in the example), it arrives at about 3.7 m/s, then throttles
  to full at contact and flies off (out of bounds) or hovers until the propellant runs out.
- Reading for RQ3: the analytic model must keep the actuator response and the rotation dynamics;
  the leg model matters only because it lets the episode continue after contact.

### T8 second part: point mass with standard actuators (approved 2026-11-23, done 13:43)

b3 changed two things at once (ideal actuators and attitude at once). Two more steps, both with
standard actuators and ending at contact, separate them (`scripts/noron/transfer_study_t8_lag.sh`,
seeds 0-4, same PhysX evaluation):

| Step | Training model | Overrides |
| --- | --- | --- |
| b4_point_mass_actuators | point mass, attitude at once | `plant=analytic_point_mass metrics.end_at_first_contact=true` |
| b5_point_mass_lag | b4 + first-order lag of the thrust axis, tau = 1.3 s | `+ plant.attitude_time_constant_s=1.3` |

- tau is measured, not chosen: the analytic rigid-body lander, hovering, flown by the classical
  attitude controller (the one behind the policy's attitude commands), reaches 63.2 % of a 2 or
  5 degree tilt step after 1.29 s (10-90 % rise 2.59 s at 5 degrees, close to a first-order lag
  with tau 1.29 s, which gives 2.84 s); larger steps are slower (1.61 s at 10, 2.20 s at 15
  degrees), so the lag is a little fast for large turns. `scripts/measure_attitude_time_constant.py`.
- The lagged plant reports the turn rate as the body rate, so the policy sees a rate and the
  touchdown metric a tilt rate. `attitude_time_constant_s: 0` (the default) is option A unchanged.
- Tests: `tests/unit/test_analytic_model_ablation.py` (12, CPU).

Results (commit 06861be, all 10 trainings reached 600 iterations, training success 97.7-99.5 %):

| Step (5 seeds) | Success in P, full rule | Judged at first contact | Main failures (mean share) |
| --- | --- | --- | --- |
| b4_point_mass_actuators | 0.0 % (seeds 0-3: 0, seed 4: 0.1 %) | 4.8 % | propellant depleted 44.6 %, hard vertical 18.5 %, tilt at contact 12.8 %, hard horizontal 11.2 % |
| b5_point_mass_lag | **90.7 % (CI 87.2-94.3)** | 90.8 % | tilt rate at contact 3.3 %, tilt at contact 3.2 %, hard horizontal 2.5 % |

- **The attitude-at-once assumption is what breaks the point mass, not the ideal actuators:**
  real actuators alone (b4) leave it at 0 %; adding a first-order attitude lag with the measured
  time constant (b5) brings it to 90.7 %. b5 also beats b2 (rigid body, ideal actuators, 55.8 %).
- b5 is still 7 points below the full model (97.8 %, seeds 0-4); its failures are tilt and tilt
  rate at contact: the lag is a little fast for large turns (measured tau 1.6-2.2 s above 10
  degrees) and has no angular momentum to overshoot.
- b5 has no `not_settled` landings although its episodes also end at first contact (b1: 0.3-20.8 %).
  Median contact values: 1.09 m/s vertical, 0.40 m/s horizontal, 2.8 degrees tilt (b1: 1.10, 0.56,
  1.9).
- Reading for RQ3, revised: the analytic model needs attitude dynamics that take time; a single
  measured time constant recovers most of the transfer. Actuator response matters next (b2).

## Revision round 1 experiments (simulated Acta Astronautica review, 2026-11-25)

Run names start with `t9_`, but they are not test T9 (terrain); they answer the simulated review
(decision package in `paper/acta_astronautica/paper/ars_review_round1/`). Queue:
`scripts/noron/transfer_study_revision_round1.sh` (commit f3c172d; follow-up evaluation added
afterwards), summary: `runs/_analysis/t9_summary.py` → `runs/_analysis/export/t9_revision_in_physx.csv`.
Same training (600 iterations, 4,096 envs) and PhysX evaluation (4,096 starts, seed 1000), seeds 0-4.

New options: `plant.body_rate_report` (`physical` default, `zero`, `held_turn_rate`: the latest
attitude change divided by `held_turn_rate_period_s`, held until the next change) for the point
mass, and `agent.zero_body_rate_in_policy_input` (evaluation only: the policy sees a zero rate,
the attitude controller still gets the real one).

| Model (5 seeds) | Success in P | Seeds | Note |
| --- | --- | --- | --- |
| b4 flown with zero rate input | 2.3 % | 0, 0, 10.3, 0.4, 0.8 | evaluation only |
| b4_held_rate (instant, rate reported) | 26.8 % (13.6-40.0) | 23.6, 25.4, 40.5, 32.4, 12.0 | +26.8 vs b4, p = 0.005 |
| b5_zero_rate (lag 1.3 s, no rate) | 0.0 % | 0, 0.2, 0, 0, 0 | -90.7 vs b5 |
| b5_zero_rate flown with zero rate input | 0.8 % (0.1-1.4) | 0.4, 0.6, 0.3, 1.4, 1.3 | evaluation only |
| b5 tau 0.3 s | 68.0 % (47.4-88.6) | 84.7, 63.4, 74.9, 41.6, 75.6 | -22.7 vs b5 (CI 2.2-43.2) |
| b5 tau 0.65 s | 90.0 % (88.3-91.7) | 89.5, 89.4, 88.5, 91.9, 91.0 | -0.7 vs b5 (CI -4.2 to +2.8) |
| b5 tau 2.2 s | 47.9 % (31.8-64.0) | 59.0, 55.0, 45.8, 53.3, 26.4 | -42.8 vs b5; tilt rate at contact 33 % |
| full model, 10 ms step (unmatched) | 98.2 % (97.9-98.5) | 98.3, 97.8, 98.1, 98.4, 98.3 | +0.4 vs full model (CI -0.4 to +1.1) |

Agreement gate (ZEM/ZEV in the training model against ZEM/ZEV in P, thresholds fixed before:
same outcome >= 99.5 %, touchdown RMS <= 0.05 m/s vertical and horizontal, <= 0.25 degrees tilt):
b1 passes (99.98 %; 0.024 / 0.014 m/s, 0.11 degrees), b2 fails (98.58 %; 0.272 / 0.125 m/s,
0.46 degrees; 42 starts land only in the analytic model), 10 ms model passes (99.98 %; 0.032 /
0.017 m/s, 0.11 degrees). The gate misses b1's loss, which comes from the episode design.

Reading: the zero-rate failure is not an input artefact (masked 0.8 %). The point mass needs both a lag and a body rate consistent with it. Rate without lag
(b4_held_rate) reaches 26.8 %, lag without rate (b5_zero_rate) 0 %, both (b5) 90.7 %. The
earlier reading "a single time constant recovers the transfer" is incomplete: the lagged plant
also gave the policy a rate. tau must be measured (half is fine, a quarter or 1.7 times is not).
Q1 does not hinge on the 5 ms step.

## Revision round 2 experiments (simulated Acta Astronautica review, 2026-11-26)

Run names start with `t10_`. Decision package and goals (with the reading rules written before
the runs): `paper/acta_astronautica/paper/ars_review_round2/`. Queue:
`scripts/noron/transfer_study_revision_round2.sh`. Same training and PhysX evaluation, seeds 0-4.

Wiring found in the code (review item R8): with `body_rate_report=zero` the zero rate goes into
`TruthState`, so the policy input, the body-rate reward term, the critic and the tilt-rate check at
contact all see zero; `held_turn_rate` feeds the held rate to all four. The round-1 controls
therefore do not separate the observation from the training signal.

New options (`src/lunarsim/agents/rsl_rl/policy_input_options.py`, training and evaluation):
`agent.zero_body_rate_in_policy_input` (now also a training option: only the policy input is
zero) and `agent.last_action_in_policy_input` (the previous clipped action appended to the
policy input, zero at the first decision). A policy trained with an option is flown with it.

| Run | What | Answers |
| --- | --- | --- |
| `t10_b5_rate_hidden_from_policy` | b5, rate hidden from the policy only | R8 |
| `t10_b5_zero_rate_last_action` | b5_zero_rate + previous action in the input, flown with zero rate input | R9 |
| `t10_refined_physx_{analytic,physx}_policy` | final T5 policies, 10 seeds each, PhysX at 2.5 ms and 32 position iterations | S24 |

Results: pending.

## Scope (proposal 2026-11-21, after the user's RL framing)

The paper answers "Can a landing policy trained in a fast analytic simulator land as well in full
rigid-body physics, and how much wall-clock time does this save?" Core: T5 and T7; T8 as the
third question (which simplification breaks the transfer); T6 only if T8 shows a gap; T1-T3 as
short validation. Future work (user, 2026-11-23): T4 (parameter sweep) and T9 (terrain), which
compare simulators rather than answer the RL question.

## Target journal (user, 2026-11-23)

Acta Astronautica (Elsevier; subscription route, no author fee); fallback Advances in Space
Research. Write to the Elsevier template.

## Running on noron

The Isaac environment `/media/noron/DISK02/isaac-lab-native/env_isaaclab` has an editable
`lunarsim` install of the other clone (`~/Desktop/MEO/lunarsimv2`); it is not changed. Every
command sets `PYTHONPATH=<this clone>/src` so this clone's code is imported (checked
2026-11-20). The DISK02 clone accepts pushes to its checked-out branch
(`receive.denyCurrentBranch=updateInstead`). Jobs and their logs live in `runs/_jobs/`.
`scripts/push_run_results.sh` is not used here: it pushes to `main` under the owner's name.

- uses: [[analytic-simulator]]
- compares with: [[ppo-landing-agent]]
- baseline: [[milestone-1b-results]]
