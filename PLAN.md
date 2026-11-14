# PLAN: Milestone 0 (Skeleton) and Milestone 1a (Lander and classical S1)

Status: **waiting for your OK**. No code is written yet.
Scope: M0 and M1a only. No rewards, no training code, no rsl_rl setup. Work stops at the M1a gate.

## 1. Settled decisions (14 Nov 2026)

| Topic | Decision |
| --- | --- |
| Repo and package | `https://github.com/MEO41/lunarsimv2.git` (private, empty, push works), package `lunarsim` |
| License | BSD-3-Clause |
| First lander | Apollo Lunar Module, every number with its public source in the preset file |
| Versions (pinned) | Isaac Sim 6.0.1 (pip), Isaac Lab 3.0.0 `develop` @ `e86463d`, Python 3.12.3, torch 2.11.0, warp 1.19.0 |
| Physics backend | PhysX, selected explicitly (stock tasks in this Isaac Lab default to Newton) |
| Force call | The articulation's permanent wrench composer, written every PhysX substep. It is the new name of `set_external_force_and_torque`, which is deprecated here |
| Hook order | As in "One control step", plus one extra plant read at the start of hook 3 so Metrics see the state after physics |
| Naming | Registry keys and scenario files as in the design (`zem_zev`, `physx_site`, `s1_terminal_truth.yaml`). Python files, classes, variables and message fields are descriptive |
| Doc conflict | The two docs give different `Binding` and `TruthState` listings. The architecture doc wins: `step(bus, dt) -> None`, `slot`, `reads_truth`, `frozenset`, `TruthState.contact`, `Frame` enum |

Message field names (design name → code name): `t` → `time`, `r` → `position`, `v` → `velocity`, `q` → `orientation_quaternion` (wxyz), `w` → `angular_velocity`, `cov` → `covariance`, `rcs` → `reaction_control_jets`.

## 2. Machine

| Item | Found |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060, 8 GB, driver 595.99.02, CUDA 13.2 |
| CPU, RAM, OS | Intel i7-14700 (28 threads), 31 GB, Ubuntu 24.04.4, kernel 7.0.0 |
| Isaac install | Native pip install in `/media/noron/DISK02/isaac-lab-native/env_isaaclab`; no Docker images |
| Proof run | Stock `Isaac-Cartpole-Direct` (`DirectRLEnv`), 1,024 envs, 300 steps: PhysX 377k env steps/s, Newton 611k env steps/s, no errors |
| Disk | `/` has 43 GB free (90% full); DISK02 has 408 GB free. Run outputs go to `runs/` (git-ignored); I will move it to DISK02 if it grows |
| Git | `gh` is logged in as MEO41 with `repo` scope; the remote is reachable and empty |

Note: the tutorial `run_cartpole_rl_env.py` rejects `--headless` in this version (headless is the default now), so I ran the same env through a short script.

## 3. Things I need your OK on

1. **Install into the shared Isaac env.** `pip install -e .` of `lunarsim` into `env_isaaclab`, plus `pyarrow` and `pytest` (both missing there). Easy to undo with `pip uninstall`.
2. **This directory becomes the repo root.** `git init` here, branch `main`, remote `origin` = MEO41/lunarsimv2. The four design files move to `docs/design/`. `CLAUDE.md` is committed.
3. **Timing.** PhysX step 5 ms (200 Hz), control step 20 ms (50 Hz, 4 substeps). The RCS pulse logic runs per substep, so the shortest pulse is 5 ms. I will change this only if the contact test needs it, and I will tell you.
4. **Action and observation layout** (the same interface RL uses in M1b): action = throttle (1) + gimbal (2) + jets (16) = 19 values; policy observation = position (3), velocity (3), quaternion (4), body rates (3), mass (1) = 14 values, built from `NavEstimate`. Gimbal is accepted but not used in M1a (the `gimbal` actuator is a later binding).
5. **Lateral control.** The Apollo descent engine cannot push sideways, so ZEM/ZEV lateral acceleration is made by tilting the lander with the RCS attitude loop (as the real vehicle did).

## 4. Milestone 0: files in order

Each row is one commit (or a few small ones), pushed after its test passes.

| # | Files | Content | Test |
| --- | --- | --- | --- |
| 0.1 | `pyproject.toml`, `LICENSE`, `README.md`, `.gitignore`, `.pre-commit-config.yaml`, `PROGRESS.md`, `docs/design/*`, `source/lunarsim/lunarsim/__init__.py` | Repo from the Isaac Lab external-project template (generator in `IsaacLab/tools/template`), trimmed: no RL library configs | `pip install -e .` works; `import lunarsim` works without Isaac |
| 0.2 | `core/message_types.py` | `Frame`, `TruthState`, `Measurement`, `NavEstimate`, `Command`, `Wrench`, `Observation` | `tests/unit/test_message_types.py`: shapes, batch dimension first, dtype checks |
| 0.3 | `core/binding.py`, `core/message_bus.py` | `Binding` protocol and base class; `MessageBus` with the runtime truth guard (a binding without `reads_truth` gets an error when it reads `TruthState`) | `test_message_bus.py`: write/read, missing key, truth guard |
| 0.4 | `core/registry.py` | Name → binding class per slot; requires/provides check in step order; truth-leak check | `test_registry.py`: good loop passes; missing provider fails; illegal truth reader fails |
| 0.5 | `core/random_generators.py`, `core/simulation_clock.py`, `core/frames.py` | One `torch.Generator` per slot from root seed + slot name; one clock; frames stub (SITE identity, others raise "Milestone 6") | `test_random_generators.py`: same seed → same draws, a change in one slot does not move another; `test_simulation_clock.py`; `test_frames.py` |
| 0.6 | `slots/<slot>/null_binding.py` for the 8 env slots, `slots/__init__.py` | Null bindings and their registry entries | `test_null_bindings.py`: the null loop passes the registry check |
| 0.7 | `configs/scenarios/m0_null_loop.yaml`, `configs/<slot>/*.yaml`, `env/scenario_loader.py`, `env/lander_env_cfg.py` | Hydra compose → `LanderEnvCfg` (`@configclass`) | `test_scenario_loader.py`: load, command-line override changes one slot, unknown binding name fails |
| 0.8 | `env/lander_env.py`, `scripts/run_scenario.py` | `LanderEnv(DirectRLEnv)`: slots called in the hook order; a box on flat ground; N envs | `tests/smoke/test_null_loop_environment.py` (GPU): 1,024 envs, 100 steps, no error; hook-order test with recording bindings |
| 0.9 | `.github/workflows/cpu_tests.yml` | CPU tests on every push (torch CPU wheel, no Isaac) | The workflow is green on GitHub |

**M0 gate:** 1,024 envs step without error; core tests pass. I report and continue to M1a.

## 5. Milestone 1a: files in order

| # | Files | Content | Test |
| --- | --- | --- | --- |
| 1.1 | `configs/landers/apollo_lunar_module.yaml`, `assets/lander_preset.py`, `core/physical_constants.py` | Mass, inertia, geometry, descent engine (thrust range, Isp, lag), RCS (thrust, Isp, positions, minimum impulse, valve delay), Moon GM and radius. Source next to each number (NASA reports, Apollo press kits, Grumman news reference, NASA fact sheet) | `test_lander_preset.py`: loads, units, every number has a `source` field |
| 1.2 | `assets/lander_builder.py` | Builds one `Articulation` USD from the preset: body, 4 legs, a prismatic spring-damper strut per leg, footpads | `tests/physics/test_lander_articulation.py` (GPU): 5 bodies, 4 prismatic joints, total mass and inertia match the preset |
| 1.3 | `slots/plant/physx_site.py`, `slots/surface/flat.py`, `slots/start/terminal_box.py`, `slots/navigation/truth.py` | Plant writes `TruthState` (SITE frame) and applies the summed `Wrench`; flat ground; start sampler over altitude, speeds and tilt; truth navigation | Unit tests for sampling ranges and seeding; smoke run |
| 1.4 | `slots/actuators/engine_rcs_pulse.py`, `slots/actuators/propulsion_models.py` (pure torch) | Throttle range, first-order lag, RCS minimum impulse and valve delay, propellant use, mass update | `tests/unit/test_propellant.py` (CPU): mass flow = thrust / (Isp g0); Δv matches the rocket equation; pulse timing tests |
| 1.5 | `tests/physics/` | **Validation tests, written before any controller.** Each states its tolerance and the reason for it: free fall, hover while mass drops, RCS torque per axis, contact rest (sum of leg forces = m g, no bounce, no jitter) | The tests themselves; all must pass |
| 1.6 | `env/lander_env_cfg.py`, strut values in the preset | Contact tuning: solver iterations, contact offsets, strut stiffness and damping. I report the chosen values | Contact-rest test passes |
| 1.7 | `slots/metrics/touchdown.py`, `tools/run_manifest.py`, `tools/episode_records.py` | Touchdown detection; vertical and horizontal speed, tilt, tilt rate, tip-over, fuel used, success and failure cause; `manifest.yaml`, `episodes.parquet` | `test_touchdown_metric.py`, `test_run_manifest.py`, `test_episode_records.py` |
| 1.8 | `agents/classical/attitude_controller.py` | **Controller step 1:** RCS attitude hold with jet selection | `tests/physics/test_attitude_hold.py`: recovers from a small tilt, stays inside a stated band |
| 1.9 | `agents/classical/vertical_controller.py` | **Step 2:** hover loop that holds altitude while mass drops | `test_hover_hold.py` |
| 1.10 | same file | **Step 3:** vertical descent to a soft touchdown from straight above | `test_vertical_descent.py`: touchdown speed under the limit |
| 1.11 | `agents/classical/zero_effort_miss_zero_effort_velocity_agent.py` (registry key `zem_zev`) | **Step 4:** full ZEM/ZEV, lateral and vertical, from the whole `terminal_box` range | `test_zem_zev_nominal_landing.py`; guidance law unit test on CPU against the closed-form answer |
| 1.12 | `configs/scenarios/s1_terminal_truth.yaml`, `scripts/evaluate_classical_agent.py` | Scenario S1 with `agent: zem_zev`; evaluation over the full start range | S1 smoke test: 64 envs, 100 steps |
| 1.13 | `slots/surface/procedural.py` | Procedural height field through `TerrainImporter` | Smoke test on the procedural surface |
| 1.14 | `tools/figures.py`, `tools/command_line.py` | `lunarsim figures <run_dir>`: figures only from `manifest.yaml` and `episodes.parquet` (plus a small trajectory log for a few envs) | `test_figures.py` on a recorded run |
| 1.15 | `scripts/benchmark_environment.py` | Env steps per second and GPU memory for S1 at several env counts (G4 baseline) | Numbers go into the report |
| 1.16 | `M1A_REPORT.md`, `PROGRESS.md` | Final report as you listed it, with the M1b task list | n/a |

**M1a gate:** all validation tests pass; ZEM/ZEV lands from nominal starts; success rate over the full `terminal_box` range is reported with failures grouped by cause; speed baseline is measured.

## 6. Rules I will keep

- All env code is batched torch on the GPU, `N` first. No Python loop over envs.
- No code or numbers from the old LunarSim repo. Constants come from cited public sources.
- `lunarsim.core` and the pure-torch models import without Isaac, so CPU tests run in CI.
- Only `nav: truth` and metrics read `TruthState`; the registry check and the bus guard enforce this.
- Every new function has a docstring. Commits are small and pushed. `PROGRESS.md` stays current.
- If something in the design does not work in Isaac Lab, I stop, explain it in simple words and propose a fix.

## 7. Known risks before starting

- **Start range against engine limits.** A start at 5 m altitude with −6 m/s may be impossible to stop softly with the real Apollo thrust-to-weight ratio. I will not hide this: the report groups such cases under their cause.
- **Develop-branch API.** Isaac Lab 3.0 is not released yet. The asset data are warp arrays; I convert them to torch at one place (the plant binding).
- **8 GB GPU.** Fine for S1 without cameras. The benchmark will show the largest env count that fits.
