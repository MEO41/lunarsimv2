---
id: metrics-and-run-records
title: Touchdown metric and run records
type: component
status: current
updated: 2026-11-14
summary: How a landing is judged (outcome codes, detection settings) and what every run writes to disk, plus the figure and summary tools.
---

# Touchdown metric and run records

## `metrics: touchdown` (`slots/metrics/touchdown.py`, `configs/metrics/touchdown.yaml`)

- Touchdown values (speeds, tilt, tilt rate) come from the last control step **before** the
  first footpad contact, so they describe the approach, not the impact.
- Contact: a leg force above 100 N. At rest: slower than 0.02 m/s and 0.5°/s for 1 s; give up
  10 s after contact. Tip-over at 45°. Out of bounds: 300 m horizontal or 200 m altitude.
- Outcomes, the first matching one wins: `success`, `out_of_bounds`, `timeout_no_contact`,
  `propellant_depleted`, `hard_vertical`, `hard_horizontal`, `tilt_at_contact`,
  `tilt_rate_at_contact`, `tip_over`, `not_settled`. Limits from [[lander-preset-apollo-lm]].
- One record per episode, collected with `drain_episode_records`.

## Run directory (git-ignored `runs/`)

- `manifest.yaml`: scenario, seeds of every slot, versions (`tools/run_manifest.py`).
- `episodes.parquet`: one row per episode (`tools/episode_records.py`).
- Trajectories for a few environments (`tools/trajectory_log.py`).

## Tools (no simulator needed)

- `lunarsim figures <run_dir>`: outcomes, success by start, touchdown envelope, trajectories
  (`tools/figures.py`).
- `lunarsim summary <run_dir>`: outcome table and failure grouping (`tools/failure_analysis.py`:
  "low and fast" = start below 20 m and faster than 3 m/s down; ideal-braking bound).
- `scripts/evaluate_agent.py`: runs `collect_episodes` (`tools/episode_runner.py`).

Planned additions with the terrain: body or nozzle strike, ground slope at the landing point
([[terrain-integration-plan]]).

- produces: [[milestone-1a-results]]
- commands: [[testing-and-commands]]
