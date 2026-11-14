---
id: pending-user-decisions
title: Pending user decisions
type: plan
status: waiting-for-user
updated: 2026-11-14
summary: Questions that wait for the user's answer; work that depends on them must not start before.
---

# Pending user decisions (as of 2026-11-14)

| # | Question | Default if the user agrees | Blocks |
| --- | --- | --- | --- |
| 1 | "Proceed" with [[terrain-integration-plan]]? | Start in this order: preset and builder, contact rerun, terrain reader, surface binding, scenario | All terrain work |
| 2 | Add a **15th observation value**, footpad height above the ground below? | Yes, so agents can land on uneven ground | Agents on terrain; observation size |
| 3 | Inertia about y: 13,667 or 13,867 slug ft²? | Re-read the Mission Report page; correct only with the page in hand | Preset change |
| 4 | Keep the strut straight (not inclined 23°) for now? | Yes, listed as an open issue | Strut geometry |
| 5 | Open milestone 1b (RL)? | No; only when the user says so | All RL work |

Answered and closed: GitHub `workflow` token scope (done; CI runs), setup script (written),
knowledge graph (this folder).

When an answer comes, move it to [[decisions-log]], update this table and
[[milestone-roadmap-and-status]] in the same commit.
