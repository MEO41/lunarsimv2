---
id: analytic-to-physx-transfer-study
title: Analytic-to-PhysX transfer study
type: plan
status: current
updated: 2026-11-20
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
| T2 | Throughput of A and P with the same agent on the same GPU, 1 to 16,384 envs (RTX 5060) | Full table done (first round, below): A 1.6× P at 1 env to 7.4× at 16,384; to repeat with the new controller |
| T3 | Paired flights: same 4,096 starts in A and P, same agent; McNemar test of the outcomes | ZEM/ZEV on the S1 box: same outcome for all 4,096 starts (2026-11-20). Wide range and the gate policies: first round below; to repeat with the new controller |
| T4 | Parameter sweep (Latin hypercube): mass ±30 %, vehicle cases, altitude, heading, time step | Later |
| T5 | PPO trained in A and in P, 5 seeds each; each tested in A and P on 4,096 held-out starts | One-seed gate `scripts/noron/transfer_study_gate.sh`: first round passed (below); to repeat with the new controller, then 5 seeds |
| T6 | Domain randomization in A, tested in P | Later |
| T7 | Wall-clock time to 90 % success in P; also "A first, then P" (`resume_from`) | `train_ppo.py` writes `checkpoint_wall_clock.csv`; first-round gate runs have it |
| T8 | Model ablation: spring legs → `metrics.end_at_first_contact=true` → no engine lag / jet limits → point mass | `end_at_first_contact` built and tested |
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
`runs/_superseded_heading_zero_controller/` on `noron`; the gate, T3 and T2 run again with the
new controller.

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
