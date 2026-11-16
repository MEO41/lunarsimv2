---
id: pending-user-decisions
title: Pending user decisions
type: plan
status: waiting-for-user
updated: 2026-11-16
summary: Questions that wait for the user's answer; work that depends on them must not start before.
---

# Pending user decisions (as of 2026-11-16)

| # | Question | Default if the user agrees | Blocks |
| --- | --- | --- | --- |
| 1 | **Paper 1, plain-`ekf` test of memory (planned, waits for a free `minsky`).** User 2026-11-16: "we will run that study later when minsky is less crowded with other projects". MLP, GRU and attention, 5 seeds, **plain `ekf`** (no fault detection) and **no health flags**, so only a history of readings can reveal a fault ([[paper-one-results]]) | Copy `scripts/minsky/queue_paper_one_ablation.sh` with `nav=ekf` added to the training and grid commands and run names `*_s3_plainekf_*`; MLP 600, GRU and attention 1,500 iterations; about 11 h on the RTX 5090. Compare with `grid_zem_zev_ekf` (78.6 %) and the `noflags` runs | Only the memory question of paper 1; the draft can start without it |

Answered and closed (2026-11-15, later): engine cut-off **probe contact** ("go with the
probe"); nozzle clearance **not a failure by default, but switchable**; struts **left to
Claude** (kept straight until milestone 5). See [[decisions-log]].

Answered and closed (2026-11-15): "Proceed" with [[terrain-integration-plan]] (user said
"proceed"); 15th observation value **yes** (user chose it); inertia about y is **13,867**
slug ft² (page image of the Mission Report checked, preset corrected). Earlier: GitHub
`workflow` token scope, setup script, knowledge graph.

When an answer comes, move it to [[decisions-log]], update this table and
[[milestone-roadmap-and-status]] in the same commit.
