---
id: testing-and-commands
title: Testing and commands
type: environment
status: current
updated: 2026-11-17
summary: How to run the tests, scenarios, evaluations, benchmarks, figures and the knowledge-graph check.
---

# Testing and commands

Activate the Isaac environment first ([[workstation-setup-and-machines]]).

## Tests

| Command | What | Needs GPU |
| --- | --- | --- |
| `python -m pytest tests/unit -q` | 201 CPU tests (core, preset, controllers, records, figures, terrain, knowledge graph, reward, action adapter, engine cut-off, EKF on a synthetic flight, hazards, regolith, frames, orbit plant, TRN, dust streaks, LiDAR noise table) | no |
| `python scripts/run_isaac_tests.py` | `tests/physics` + `tests/smoke` (27 files, including the probe test, the EKF gate test, the PPO smoke tests, S6, S8, the S7/S5 mission phases, S10 and the L2 grains), one simulator process per file; the terrain test is skipped without the height maps | yes |
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
python scripts/sensor_check.py && python scripts/plot_sensor_check.py && python scripts/render_sensor_check.py   # visual sensor check
lunarsim compare zem_zev=runs/a ppo=runs/b,runs/c,runs/d                          # same held-out starts, low-and-fast split
python scripts/export_policy.py run=runs/ppo_mlp_s1_seed0 checkpoint=model_500.pt   # policy.pt from any checkpoint, no Isaac
python scripts/benchmark_environment.py scenario=s1_terminal_truth num_envs=1024 steps=500
python scripts/run_mission.py mission=s7_full_mission num_envs=16 run_name=s7_full_mission_seed0 seed=0   # S7 orbit to touchdown, phase by phase
python scripts/run_mission.py mission=s5_trn num_envs=16 run_name=s5_trn_seed0 seed=0                     # S5 TRN descent
python scripts/run_mission.py mission=s7_full_mission num_envs=16 seed=0 run_name=s7_full_mission_video_seed0 record=0 && python scripts/render_mission_video.py run=s7_full_mission_video_seed0 lander=0   # S7 video
python scripts/evaluate_agent.py scenario=s8_plume_check_l2 num_envs=64 seed=0 run_name=s8_plume_check_l2_seed0 && python scripts/compare_plume_levels.py run=runs/s8_plume_check_l2_seed0   # M5 L1 vs L2
python scripts/export_policy_onnx.py policy=runs/ppo_gru_s3_seed0_to1500/policy.pt kind=gru hidden_size=128   # S10 policy node
python scripts/run_software_in_the_loop.py episodes=20 seed=1000 run_name=s10_seed1000                       # S10 parity and timing
python scripts/render_lander_image.py --enable_cameras replay=runs/contact_check/contacts.npz clip=1 output=docs/figures/lander_at_rest.png   # on the Apollo 17 terrain (needs the contact-check recording); without replay= on flat ground
python scripts/run_drop_test.py drops=1000 drop_height_m=5 run_name=drop_test_5m
python scripts/record_landing_videos.py landings=3072                             # then:
python scripts/render_replay.py replay=runs/landing_videos/landings.npz mode=video
python scripts/contact_check.py                                                   # then:
python scripts/render_replay.py replay=runs/contact_check/contacts.npz mode=stills
python scripts/render_replay.py replay=runs/drop_test_5m/flips.npz mode=video
python scripts/record_plume_grain_video.py   # M5 L2 grains: 64 landers, 2 recorded, chase + window views (about 25 min on the RTX 5060)
python scripts/record_plume_grain_video.py fly=0 gain=100   # render the saved replay again, dust opacity x100 (labelled)
bash scripts/push_run_results.sh                  # results of every finished run -> results/ on GitHub (user rule)
python scripts/archive_run_results.py runs/<run_name>   # only copy into results/ (no commit)
lunarsim figures runs/<run_name>
lunarsim summary runs/<run_name>
lunarsim knowledge-graph            # rebuild graph.json and graph_overview.md
lunarsim knowledge-graph --check    # broken links, bad headers, stale files
```

Every `key=value` after the script name is a Hydra override of the scenario. One process
measures one environment count in the benchmark.

- runs: [[metrics-and-run-records]]
- checks: [[knowledge-graph-tooling]]
