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
| T1 | Analytic model check: the PhysX validation tests' known answers on A | **CPU part passes** (2026-11-20); GPU run on `noron` next |
| T2 | Throughput of A and P with the same agent on the same GPU, 1 to 16,384 envs (RTX 5060) | Script ready (`benchmark_environment.py` takes `plant=analytic_rigid_body`) |
| T3 | Paired flights: same 4,096 starts in A and P, same agent; McNemar test of the outcomes | Next |
| T4 | Parameter sweep (Latin hypercube): mass ±30 %, vehicle cases, altitude, heading, time step | Later |
| T5 | PPO trained in A and in P, 5 seeds each; each tested in A and P on 4,096 held-out starts | Later; P must be retrained on the RTX 5060 (the 97.95 % runs were on `minsky`) |
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

- uses: [[analytic-simulator]]
- compares with: [[ppo-landing-agent]]
- baseline: [[milestone-1b-results]]
