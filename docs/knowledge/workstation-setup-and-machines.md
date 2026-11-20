---
id: workstation-setup-and-machines
title: Workstation setup and machines
type: environment
status: current
updated: 2026-11-15
summary: The two machines, where the Python environment lives on each, how to set up a new one, and what does not travel with git.
---

# Workstation setup and machines

## Machines

| Machine | GPU | Status | Isaac Python environment |
| --- | --- | --- | --- |
| Development machine (until 2026-11-14) | RTX 5060, 8 GB, driver 595.99.02, Ubuntu 24.04 | All M0/M1a results | `/media/noron/DISK02/isaac-lab-native/env_isaaclab`, Isaac Lab source at `/media/noron/DISK02/isaac-lab-native/IsaacLab` |
| `minsky` (from 2026-11-15), heavy GPU runs | RTX 5090, 32 GB; 32 CPU threads; 60 GB RAM; 1.9 TB disk | Set up 2026-11-15 by hand (not by the setup script) | **`~/lunarsim-workstation/env_isaaclab`** (pinned setup, user `minsky`); the user's `~/isaac-env` (Isaac Sim 6.2) is not used. Repo at `~/lunarsimv2`, height maps copied into `assets/terrain/apollo17/` |

Repository on the development machine: `/home/noron/Desktop/MEO/lunarsimv2`. The
[[analytic-to-physx-transfer-study]] uses its own clone on the same machine,
`/media/noron/DISK02/Speed_Versus_Fidelity/lunarsimv2` (user rule: that study works only on DISK02). On a new
machine, wherever it is cloned.

## Using `minsky` from the development machine

The two machines share the local network and Tailscale. `minsky` is at `minsky.local`
(10.9.25.45) and at Tailscale address 100.104.128.23 (`minsky.tailc4da8d.ts.net`). Log in as
user `minsky` with the development machine's key `~/.ssh/id_ed25519` (installed 2026-11-15;
no password needed, and none is written in this repository):

```bash
ssh minsky@100.104.128.23
source ~/lunarsim-workstation/env_isaaclab/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES && cd ~/lunarsimv2 && git pull
python scripts/evaluate_agent.py scenario=apollo17_terminal_truth num_envs=4096 run_name=... seed=0
```

Work flow: edit and commit on the development machine, push, `git pull` on `minsky`, run
there, then copy the run directories back with `rsync -a minsky@100.104.128.23:lunarsimv2/runs/<name> runs/`.
`gh` on `minsky` is logged in as MEO41. The user's own environment `~/isaac-env` has Isaac Sim 6.2.0 and a newer Isaac Lab whose cloner
API differs, so LunarSim does not run in it. LunarSim uses a second environment made by
`scripts/setup_workstation.sh` (2026-11-15): `~/lunarsim-workstation/env_isaaclab`, Isaac Sim
6.0.1, Isaac Lab `e86463d`, rsl_rl 5.4.1. 18 of 20 checks agree: torch and torchvision are the
CUDA 12.8 builds (Isaac Lab's install step replaced the cu130 wheels). The first Isaac launch in
a new environment asks for the NVIDIA EULA on stdin, which blocks scripts: set
`OMNI_KIT_ACCEPT_EULA=YES` (the user had accepted it in `~/isaac-env`). All 15 GPU test files pass there.

```bash
source ~/lunarsim-workstation/env_isaaclab/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES
```

## Speed on `minsky` (benchmark 2026-11-15, `scripts/minsky/benchmark_5090.sh`)

PPO training in S3 (`ppo_mlp`, 24 decisions per environment per iteration):

| Setting | Seconds per iteration, each run | Env steps per second, all runs |
|---|---|---|
| 1 run, 4,096 landers | 4.0 | 122k |
| 1 run, 16,384 landers | 9.4 | 208k |
| 1 run, 32,768 landers | 27.2 | 145k |
| 5 runs × 4,096, no MPS | 12.0–16.4 | 172k |
| **5 runs × 4,096, CUDA MPS** | **6.3–7.5** | **350k** |
| 2 runs × 16,384, CUDA MPS | 16.9–17.3 | 230k |

Without MPS the runs take turns on the GPU (time-slicing); with MPS their kernels run at the
same time. Environment only (S1, `zem_zev`): 126k env steps/s at 4,096 landers, 148k at 16,384,
155k at 32,768, 248k at 65,536 (13.5 GB). Paper 1 trains with 5 × 4,096 under MPS (user
decision 2026-11-15); start the daemon with `nvidia-cuda-mps-control -d` and the variables
`CUDA_MPS_PIPE_DIRECTORY`, `CUDA_MPS_LOG_DIRECTORY` (see `scripts/minsky/queue_paper_one.sh`),
stop it with `echo quit | nvidia-cuda-mps-control`.

## Pinned software (`requirements-pinned.txt`)

Isaac Sim 6.0.1.0 (pip, `isaacsim[all,extscache]`, from pypi.nvidia.com); Isaac Lab 3.0.0,
branch develop, commit `e86463df8a8d8dff275e50fdcfd892e10ba32ec1`; Python 3.12.3; torch 2.11.0
(+cu130, CUDA 13.0); numpy 2.3.1; scipy 1.17.0; pandas 3.0.5; pyarrow 26.0.1; matplotlib
3.10.8; hydra-core 1.3.7; omegaconf 2.3.1; trimesh 4.11.1; pytest 9.2.0; warp-lang 1.19.0.

## Setting up a new machine

```bash
gh auth login                                   # repo is private
git clone https://github.com/MEO41/lunarsimv2.git && cd lunarsimv2
scripts/setup_workstation.sh                    # default root ~/lunarsim-workstation
source ~/lunarsim-workstation/env_isaaclab/bin/activate
python scripts/check_installation.py            # 20 checks against the pinned setup
python -m pytest tests/unit -q                  # 89 CPU tests
python scripts/download_height_maps.py          # Apollo 17 height maps, 952 MB, not in git
python scripts/run_isaac_tests.py               # 13 GPU test files
```

Then rerun the speed table ([[speed-benchmarks]]) and one full evaluation (seed 0, 4,096 envs;
compare with 96.90 %, [[milestone-1a-results]]).

The setup script has not yet run on a clean machine. Its guesses: `./isaaclab.sh --install
core` (the original install option is unknown) and a minimum driver of 580. If a step fails,
fix the script and record it here.

## What does not travel with git

- `runs/` (evaluation runs; figures are already in `docs/`). Copy only to rebuild figures.
- `~/Desktop/MEO/apollo_lander_modeling/` ([[apollo-lander-modeling-folder]]).
- Height maps ([[apollo17-height-maps]]); download them again.
- Local Claude memory (`~/.claude/projects/.../memory/`). **This knowledge graph replaces it**
  for project context.

## GitHub

Account MEO41, `gh` token scopes include `repo` and `workflow` (added 2026-11-14 so the CI
workflow could be pushed). CPU tests run on every push (`.github/workflows/cpu_tests.yml`).

- quirks: [[isaac-lab-quirks]]
- commands: [[testing-and-commands]]
