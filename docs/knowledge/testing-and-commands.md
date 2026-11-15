---
id: testing-and-commands
title: Testing and commands
type: environment
status: current
updated: 2026-11-15
summary: How to run the tests, scenarios, evaluations, benchmarks, figures and the knowledge-graph check.
---

# Testing and commands

Activate the Isaac environment first ([[workstation-setup-and-machines]]).

## Tests

| Command | What | Needs GPU |
| --- | --- | --- |
| `python -m pytest tests/unit -q` | 128 CPU tests (core, preset, controllers, records, figures, terrain, knowledge graph, reward, action adapter, engine cut-off, EKF on a synthetic flight) | no |
| `python scripts/run_isaac_tests.py` | `tests/physics` + `tests/smoke` (17 files, including the probe test, the EKF gate test and the PPO smoke test), one simulator process per file; the terrain test is skipped without the height maps | yes |
| `python scripts/run_isaac_tests.py physics` | one folder | yes |
| `python scripts/run_isaac_tests.py tests/smoke/test_null_loop_environment.py` | one file | yes |
| `python scripts/check_installation.py` | versions against `requirements-pinned.txt`, CUDA check | yes |

GPU tests carry the pytest marker `isaac`. GitHub Actions runs the CPU tests on every push.

## Running things

```bash
python scripts/download_height_maps.py                                            # once per machine, 952 MB
python scripts/run_scenario.py scenario=m0_null_loop num_envs=1024 steps=100      # add --viz kit for a window
python scripts/evaluate_agent.py scenario=apollo17_terminal_truth num_envs=4096 run_name=terrain_seed0 seed=0
python scripts/evaluate_agent.py scenario=s1_terminal_truth num_envs=4096 run_name=zem_zev_seed0 seed=0
python scripts/evaluate_agent.py scenario=s2_terminal_ekf num_envs=4096 run_name=s2_zem_zev_seed0 seed=0   # S2: sensors + EKF
python scripts/train_ppo.py scenario=s1_terminal_truth num_envs=4096 seed=0 run_name=ppo_mlp_s1_seed0    # M1b, on minsky
python scripts/evaluate_agent.py scenario=s1_terminal_truth agent=ppo_mlp agent.policy_path=runs/ppo_mlp_s1_seed0/policy.pt num_envs=4096 seed=1000 run_name=...
lunarsim compare zem_zev=runs/a ppo=runs/b,runs/c,runs/d                          # same held-out starts, low-and-fast split
python scripts/export_policy.py run=runs/ppo_mlp_s1_seed0 checkpoint=model_500.pt   # policy.pt from any checkpoint, no Isaac
python scripts/benchmark_environment.py scenario=s1_terminal_truth num_envs=1024 steps=500
python scripts/render_lander_image.py --enable_cameras output=docs/figures/lander_at_rest.png
python scripts/run_drop_test.py drops=1000 drop_height_m=5 run_name=drop_test_5m
python scripts/record_landing_videos.py landings=3072                             # then:
python scripts/render_replay.py replay=runs/landing_videos/landings.npz mode=video
python scripts/contact_check.py                                                   # then:
python scripts/render_replay.py replay=runs/contact_check/contacts.npz mode=stills
python scripts/render_replay.py replay=runs/drop_test_5m/flips.npz mode=video
lunarsim figures runs/<run_name>
lunarsim summary runs/<run_name>
lunarsim knowledge-graph            # rebuild graph.json and graph_overview.md
lunarsim knowledge-graph --check    # broken links, bad headers, stale files
```

Every `key=value` after the script name is a Hydra override of the scenario. One process
measures one environment count in the benchmark.

- runs: [[metrics-and-run-records]]
- checks: [[knowledge-graph-tooling]]
