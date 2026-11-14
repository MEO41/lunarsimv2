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
| New machine (from 2026-11-14) | RTX 5090, 32 GB | Being set up | Default from the setup script: `~/lunarsim-workstation/env_isaaclab` |

Repository on the development machine: `/home/noron/Desktop/MEO/lunarsimv2`. On a new
machine, wherever it is cloned.

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
