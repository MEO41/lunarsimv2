# LunarSim v2

A modular lunar-landing research simulator on Isaac Lab, written from scratch.
Every part of the control loop is a *slot* with typed inputs and outputs, so a study can swap
sensors, faults, estimators, agents or physics in one line of a scenario file.

- Design: [docs/design](docs/design) (architecture and build plan, scenario catalogue).
- Plan for the current milestones: [PLAN.md](PLAN.md). Status: [PROGRESS.md](PROGRESS.md).

## Pinned versions

| Component | Version |
| --- | --- |
| Isaac Sim | 6.0.1 (pip) |
| Isaac Lab | 3.0.0, branch `develop`, commit `e86463d` |
| Physics backend | PhysX (selected explicitly) |
| Python | 3.12 |
| torch | 2.11.0 |

## Install

```bash
# inside the Python environment that has Isaac Sim and Isaac Lab
pip install -e ".[dev]"
```

The `lunarsim.core` package and the pure-torch models need only `torch`, so the CPU tests run
without Isaac:

```bash
pytest tests/unit
```

Tests that need Isaac Lab and a GPU each start their own simulator process:

```bash
python scripts/run_isaac_tests.py            # all GPU tests
python scripts/run_isaac_tests.py smoke      # one folder
```

## Run a scenario

```bash
python scripts/run_scenario.py scenario=m0_null_loop num_envs=1024 steps=100
```

## Layout

```
src/lunarsim/core       message types, bus, registry, clock, random generators, frames
src/lunarsim/slots      one folder per slot, one file per binding
src/lunarsim/env        LanderEnv (DirectRLEnv), config, scenario loader
src/lunarsim/agents     classical agents (learning agents come in milestone 1b)
src/lunarsim/assets     lander builder from a preset
src/lunarsim/tools      run manifest, episode records, figures
configs/scenarios       one Hydra file per scenario
configs/landers         lander presets, every number with its public source
tests/unit              CPU tests (run on every push)
tests/physics, smoke    GPU tests (need Isaac Lab)
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
