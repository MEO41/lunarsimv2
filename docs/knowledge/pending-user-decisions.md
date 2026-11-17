---
id: pending-user-decisions
title: Pending user decisions
type: plan
status: waiting-for-user
updated: 2026-11-17
summary: Questions that wait for the user's answer; work that depends on them must not start before.
---

# Pending user decisions (as of 2026-11-16)

| # | Question | Default if the user agrees | Blocks |
| --- | --- | --- | --- |
| 1 | **Paper 1, plain-`ekf` test of memory (planned, waits for a free `minsky`).** User 2026-11-16: "we will run that study later when minsky is less crowded with other projects". MLP, GRU and attention, 5 seeds, **plain `ekf`** (no fault detection) and **no health flags**, so only a history of readings can reveal a fault ([[paper-one-results]]) | Copy `scripts/minsky/queue_paper_one_ablation.sh` with `nav=ekf` added to the training and grid commands and run names `*_s3_plainekf_*`; MLP 600, GRU and attention 1,500 iterations; about 11 h on the RTX 5090. Compare with `grid_zem_zev_ekf` (78.6 %) and the `noflags` runs | Only the memory question of paper 1; the draft can start without it |

| 3 | **Install ROS 2 for the S10 bridge?** The bridge runs over a loopback transport now; the ROS 2 transport (`transport: ros2`) is written but cannot be tested, because ROS 2 is not installed and installing it needs `sudo` ([[software-in-the-loop-bridge]]) | ROS 2 Jazzy from apt (Ubuntu 24.04, Python 3.12 like the Isaac environment) on one machine; then run S10 with `nav.transport=ros2 agent.transport=ros2` and two `python -m lunarsim.bridge.serve_node --transport ros2` nodes | Only the ROS 2 transport check; the M7 parity gate runs on loopback |
| 5 | **S6 after the LiDAR comparison (2026-11-17).** (a) Default S6/S7 LiDAR noise: keep the ASC specification (31 % safe) or switch to the ASC measured errors (94 %)? (b) The agent lands 0.8-0.9 m from its chosen site when it re-plans early, which caps a good map near 83 %: add a terminal position hold over the chosen site (`site_pick_divert` only), a landing-miss allowance in the footprint check, or both ([[hazard-detection-and-avoidance]]) | (a) measured errors, the specification kept as a variant; (b) both as `site_pick_divert` options, switched on in S6 after a check run; `zem_zev` in S1-S4 untouched | S6 and S7 baseline numbers |
| 4 | **How should the M7 parity gate be stated?** The tolerance set before the run (1 m along the trajectory, 0.1 m/s at touchdown) fails for the flight-like CPU/ONNX nodes (5.2 m, 0.7 m/s, same outcome in 20/20). The bridge is exact with the simulator's arithmetic (0.000 m), and the in-loop code alone differs by 3.1 m when only compilation is switched off, so no change of arithmetic can meet 1 m in this loop ([[software-in-the-loop-bridge]]) | Gate = (a) exact match with the same arithmetic (passed) and (b) the CPU/ONNX nodes within the in-loop code's own rounding spread: same outcome rate, touchdown metrics within the `reference_eager` differences. Under that rule the gate passes | Only the M7 gate wording |

Answered and closed (2026-11-17): **M5 L2** with batched torch grains, not PhysX rigid bodies (user: "It has to be efficiently run on the gpu"; [[plume-regolith-and-dust]]).

Answered and closed (2026-11-15, later): engine cut-off **probe contact** ("go with the
probe"); nozzle clearance **not a failure by default, but switchable**; struts **left to
Claude** (kept straight until milestone 5). See [[decisions-log]].

Answered and closed (2026-11-15): "Proceed" with [[terrain-integration-plan]] (user said
"proceed"); 15th observation value **yes** (user chose it); inertia about y is **13,867**
slug ft² (page image of the Mission Report checked, preset corrected). Earlier: GitHub
`workflow` token scope, setup script, knowledge graph.

When an answer comes, move it to [[decisions-log]], update this table and
[[milestone-roadmap-and-status]] in the same commit.
