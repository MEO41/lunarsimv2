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
| T2 | Throughput of A and P with the same agent on the same GPU, 1 to 16,384 envs (RTX 5060) | First numbers: A 2.1× P at 1,024 envs, 7.6× at 16,384 ([[analytic-simulator]]); the full table (5 repeats, every count, both in one session) is still to do |
| T3 | Paired flights: same 4,096 starts in A and P, same agent; McNemar test of the outcomes | ZEM/ZEV on the S1 box: same outcome for all 4,096 starts (2026-11-20). Still to do: the wide range (5-150 m, `start.heading_deg=[0,360]`) and trained policies |
| T4 | Parameter sweep (Latin hypercube): mass ±30 %, vehicle cases, altitude, heading, time step | Later |
| T5 | PPO trained in A and in P, 5 seeds each; each tested in A and P on 4,096 held-out starts | One-seed gate: `scripts/noron/transfer_study_gate.sh` (P must be retrained on the RTX 5060; the 97.95 % runs were on `minsky`) |
| T6 | Domain randomization in A, tested in P | Later |
| T7 | Wall-clock time to 90 % success in P; also "A first, then P" (`resume_from`) | `train_ppo.py` writes `checkpoint_wall_clock.csv` |
| T8 | Model ablation: spring legs → `metrics.end_at_first_contact=true` → no engine lag / jet limits → point mass | `end_at_first_contact` built and tested |
| T9 | All policies in P on the Apollo 17 terrain | Later (height maps must be downloaded on `noron`) |

## Decisions (user, 2026-11-20)

- The main A has spring-strut legs, so both simulators judge landings with the same outcome
  codes (settling, tip-over). "End at first contact" is a step of T8, between the spring legs
  and the point mass (a point mass has no legs, so it needs that rule anyway).
- Code is written on the user's Windows machine and pushed to `noron`; all GPU work and training
  run on `noron`, only on `/media/noron/DISK02` (repository at
  `/media/noron/DISK02/Speed_Versus_Fidelity/lunarsimv2`, branch `analytic-simulator`).

## Open questions

- **Heading and the attitude controller (found 2026-11-20).** The classical attitude loop
  (`agents/classical/attitude_controller.py`, also used by `ppo_mlp` with `attitude_commands`)
  always aims at heading 0. With `start.heading_deg=[0,360]` the lander turns back at up to
  5 deg/s, and when that turn is still running at contact the body rate breaks the 2 deg/s limit:
  9 of 64 wide-range ZEM/ZEV starts failed as `tilt_rate_at_contact` (analytic, CPU check), about
  0.3 % without the heading range. It affects both simulators alike, so T3 stays a fair paired
  test, but it lowers the baseline of R1. Changing the controller to hold the current heading is
  a design change of the owner's code: the user decides.

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
