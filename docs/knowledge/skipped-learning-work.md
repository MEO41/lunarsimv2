---
id: skipped-learning-work
title: Learning work skipped until more compute is available
type: plan
status: open
updated: 2026-11-16
summary: The user's rule of 2026-11-16 (build through M7, no training longer than 20 minutes) and the list of every learning step left out on the way, to come back to.
---

# Learning work skipped until more compute is available

User, 2026-11-16: "reach the M7 do not do anything that requries training more than 20min lets
see the end of the project later we can come back and do the training when there we have more
available compute. Also note that we skipped learning part."

So milestones 4 to 7 are built with their classical agents and with every learned agent left
out. A training of at most 20 minutes is allowed (for example a short check that an adapter
learns at all). Each entry below names what was skipped and what is ready for it.

| Milestone | Skipped learning work | What is ready | Where it was planned |
| --- | --- | --- | --- |
| M3 | Plain-`ekf` memory study of paper 1 (MLP, GRU, attention without health flags) | queue recipe | [[pending-user-decisions]] item 1, [[paper-one-results]] |
| M3 | PPO policies in S4 (late Doppler data) | S4, `ekf_late_fusion`, the PPO agents | [[faults-and-detection]] |
| M4 | Agent adapters `sac`, `dreamer`, `tdmpc2`, the equal-budget sweep runner, scenario S9 and paper 2 | the S3 environment (S9 reuses it), `ppo_mlp`/`ppo_gru`/`ppo_attention` | design M4, scenario catalogue S9 |
| M4 | A learned site-selection or divert agent in S6 | S6 with the classical `site_pick_divert` baseline | [[hazard-detection-and-avoidance]] |

| M5 | The design's S8 agent `ppo_gru` (learned); ZEM/ZEV flies S8 | S8, `ppo_gru` | [[plume-regolith-and-dust]] |

Rows are added as milestones 6 and 7 are built.

- part of: [[milestone-roadmap-and-status]]
- follows: [[rules-from-the-user]]
