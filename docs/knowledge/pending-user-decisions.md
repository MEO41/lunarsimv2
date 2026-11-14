---
id: pending-user-decisions
title: Pending user decisions
type: plan
status: waiting-for-user
updated: 2026-11-15
summary: Questions that wait for the user's answer; work that depends on them must not start before.
---

# Pending user decisions (as of 2026-11-14)

| # | Question | Default if the user agrees | Blocks |
| --- | --- | --- | --- |
| 1 | Keep the strut straight (not inclined 23°) for now? | Yes, listed as an open issue | Strut geometry |
| 2 | Open milestone 1b (RL)? | No; only when the user says so | All RL work |
| 3 | Should the nozzle clearance (recorded, not a failure) ever count as a failure? | No: the real skirt is crushable (28 in) | Outcome rules |
| 4 | Change the agents' engine cut-off from "0.25 m below the c.g." to a probe-contact signal like Apollo (1.7 m probes) or the lowest-footpad height? | Probe contact (closer to the real LM) | Body strikes over rocks ([[open-issues]]) |

Answered and closed (2026-11-15): "Proceed" with [[terrain-integration-plan]] (user said
"proceed"); 15th observation value **yes** (user chose it); inertia about y is **13,867**
slug ft² (page image of the Mission Report checked, preset corrected). Earlier: GitHub
`workflow` token scope, setup script, knowledge graph.

When an answer comes, move it to [[decisions-log]], update this table and
[[milestone-roadmap-and-status]] in the same commit.
