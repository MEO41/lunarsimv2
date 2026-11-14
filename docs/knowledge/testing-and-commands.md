---
id: testing-and-commands
title: Testing and commands
type: environment
status: current
updated: 2026-11-14
summary: How to run the tests, scenarios, evaluations, benchmarks, figures and the knowledge-graph check.
---

# Testing and commands

Activate the Isaac environment first ([[workstation-setup-and-machines]]).

## Tests

| Command | What | Needs GPU |
| --- | --- | --- |
| `python -m pytest tests/unit -q` | 89 CPU tests (core, preset, controllers, records, figures, knowledge graph) | no |
| `python scripts/run_isaac_tests.py` | `tests/physics` + `tests/smoke`, one simulator process per file | yes |
| `python scripts/run_isaac_tests.py physics` | one folder | yes |
| `python scripts/run_isaac_tests.py tests/smoke/test_null_loop_environment.py` | one file | yes |
| `python scripts/check_installation.py` | versions against `requirements-pinned.txt`, CUDA check | yes |

GPU tests carry the pytest marker `isaac`. GitHub Actions runs the CPU tests on every push.

## Running things

```bash
python scripts/run_scenario.py scenario=m0_null_loop num_envs=1024 steps=100      # add --viz kit for a window
python scripts/evaluate_agent.py scenario=s1_terminal_truth num_envs=4096 run_name=zem_zev_seed0 seed=0
python scripts/benchmark_environment.py scenario=s1_terminal_truth num_envs=1024 steps=500
python scripts/render_lander_image.py --enable_cameras output=docs/figures/lander_at_rest.png
lunarsim figures runs/<run_name>
lunarsim summary runs/<run_name>
lunarsim knowledge-graph            # rebuild graph.json and graph_overview.md
lunarsim knowledge-graph --check    # broken links, bad headers, stale files
```

Every `key=value` after the script name is a Hydra override of the scenario. One process
measures one environment count in the benchmark.

- runs: [[metrics-and-run-records]]
- checks: [[knowledge-graph-tooling]]
