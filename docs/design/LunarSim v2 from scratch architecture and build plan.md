# LunarSim v2 from scratch: architecture and build plan

Nov 14, 2026 · @Muhammed Emin Onay

LunarSim v2 is a new Isaac Lab project, written from zero. It uses no code from the earlier LunarSim repo. Every part of the loop is a slot with typed inputs and outputs, so a study can swap sensors, faults, estimators, agents or physics in one config line. The scenario catalogue is [Scenario Bindings](https://claude.ai/artifact/N2L2XMPYfVQSHewLbynBAQ).

## Key decisions

v2 is built for GPU-batched training in Isaac Lab from the first commit, with GNC research (sensors, faults, estimation, agents) as its main use.

| Topic | Choice | Why |
| --- | --- | --- |
| Env base | Isaac Lab `DirectRLEnv` subclass | Its hooks map one to one onto the slot loop, and we control the order of every slot |
| Data | `torch` tensors on the GPU, batch dimension `N` first | No Python loop over envs, anywhere |
| Precision | float32 for the PhysX site frame; float64 for the orbit plant | PhysX is float32; orbit distances need float64 |
| Config | Hydra scenario files that resolve into Isaac Lab `@configclass` objects | One file per scenario, command-line overrides, sweeps |
| RL libraries | rsl\_rl first (PPO, recurrent policies); skrl for SAC and attention; adapters for Dreamer and TD-MPC2 | rsl\_rl is Isaac Lab's default and supports GRU and LSTM |
| Lander | Parametric lander config; first preset from public Apollo LM data | Other landers are new presets, not new code |
| Terrain | Procedural height fields first; public DEM tiles (LROC NAC DTM) later | Training starts fast; real sites come when hazard studies need them |
| Cheap sensors | Isaac Lab `Imu`, `RayCaster` (altimeter, Doppler beams, LiDAR), `ContactSensor` | Batched on the GPU, fast enough for thousands of envs |
| Expensive sensors | `TiledCamera`, RTX LiDAR, for tests and final results | Too slow for most training |
| Random numbers | One root seed; one `torch.Generator` per slot | A change in one slot does not change the draws of another |
| ROS 2 | A bridge binding outside the training loop | ROS 2 is too slow for thousands of envs |
| Old repo | Not used: no code, no numbers | A clean start; every constant comes from a cited public source |

## Architecture

&#91;embedded content: architecture · five layers; eight slots in the env, the agent outside\]

The agent slot sits outside the env, because training libraries own the agent. Every other slot lives inside `LanderEnv`. Agents never see `TruthState`, except the critic during training.

## One control step

&#91;embedded content: one control step · 6 hooks mapped to slots\]

The coloured row is the GNC chain that most studies change. It runs last, after resets, so a reset env starts its new episode with a fresh measurement and estimate.

## Interfaces

Slots exchange only these message types. All tensors are batched: the first dimension is the env count `N`.

```python
@dataclass
class TruthState:
    frame: Frame            # MCI, MCMF or SITE
    t: Tensor               # (N,)
    r: Tensor; v: Tensor    # (N, 3) float64 in orbit, float32 at the site
    q: Tensor; w: Tensor    # (N, 4) wxyz, (N, 3) body rates
    mass: Tensor            # (N,)
    contact: Tensor         # (N, n_legs) leg contact forces

@dataclass
class Measurement:          # one per sensor
    value: Tensor           # (N, d)
    valid: Tensor           # (N,) bool, False when a fault drops it
    stamp: Tensor           # (N,) capture time
    cov: Tensor             # (N, d, d) or (N, d)

@dataclass
class NavEstimate:
    mean: Tensor; cov: Tensor
    health: Tensor          # (N, n_sensors) fault flags

@dataclass
class Command:
    throttle: Tensor        # (N,) 0..1
    gimbal: Tensor          # (N, 2) rad
    rcs: Tensor             # (N, n_jets) on/off or duty

@dataclass
class Wrench:
    force: Tensor; torque: Tensor   # (N, 3) body frame
```

Every binding has the same interface:

```python
class Binding(Protocol):
    slot: str                    # one of the 9 slot names
    requires: frozenset[str]     # message keys it reads
    provides: frozenset[str]     # message keys it writes
    reads_truth: bool = False    # only nav.truth and metrics may set this
    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None: ...
    def step(self, bus: MessageBus, dt: float) -> None: ...
```

The `MessageBus` holds the latest message of each key for all envs. Before a run, the registry checks that every `requires` key has a provider earlier in the step, and that only bindings with `reads_truth=True` read `TruthState`. The critic observation group reads `TruthState` through a separate, logged path.

**Rules**

- Each slot reads only its neighbour's output, except the truth bypass.
- One root seed. Each slot gets its own `torch.Generator`, seeded from the root seed and the slot name.
- One clock. `stamp` is the capture time, not the arrival time.
- Every `TruthState` carries its frame. Only `core/frames.py` changes frames.
- Every run saves its resolved scenario config, git hash and seeds next to its results.

## Package layout

&#91;embedded content: package layout · package tree and repo root\]

A new binding is one new file in its slot folder plus one registry entry. The registry name is the name used in scenario files, for example `nav: ekf_fdi`.

## Slots

Each slot starts with the simplest binding that lets S1 run. Later milestones add bindings; they never change the slot interface.

| Slot | First binding (M1) | Later bindings | Isaac Lab or torch feature |
| --- | --- | --- | --- |
| Start | `terminal_box` | `approach_gate`, `orbit_elements`, `from_log` | `_reset_idx`, per-slot generator |
| Plant | `physx_site` | `orbit_f64`, `hybrid` (handoff) | `Articulation` lander; torch float64 integrator |
| Surface | `flat`, `procedural` | `dem_tile`, `plume_l1`, `bekker_pads` | `TerrainImporter` height fields; wrench on the body |
| Sensors | none (truth) | `imu`, `radar_alt`, `doppler_vel`, `lidar_cheap`, `camera_cheap`, RTX versions | `Imu`, `RayCaster`, `TiledCamera`, RTX LiDAR |
| Faults | `none` | `intermittent`, `persistent`, `delay`, `dust` | torch masks and ring buffers |
| Navigation | `truth` | `ekf`, `ekf_fdi`, `ekf_late_fusion`, `trn_ekf`, `hazard_map` | batched torch EKF |
| Agent | `zem_zev``  (M1a), ppo_mlp (M1b) ` | `ppo_gru`, `ppo_attention`, `sac`, `dreamer`, `tdmpc2`, `mpc` | rsl\_rl, skrl, adapters |
| Actuators | `engine_rcs_pulse` (throttle range, lag, minimum impulse, valve delay) | `gimbal`, `engine_out` | `set_external_force_and_torque` per substep |
| Metrics | `touchdown` | `landing_error`, `nav_error`, `fault_response`, `dust`, `latency` | `ContactSensor`, episode records |

The lander is one `Articulation`: a body, four legs, and a prismatic spring-damper strut in each leg for stroke. Mass, inertia, engine and RCS data come from a lander preset file. The first preset uses public Apollo LM numbers, each with its source in the file.

## Testing and reproducibility

With no old repo to match, every physics part is checked against a known answer.

| Test | Known answer | Runs on |
| --- | --- | --- |
| Free fall | z(t) = z₀ − ½ g t², g = 1.62 m/s² | GPU (Isaac) |
| Hover | Thrust = m g keeps altitude constant while mass drops | GPU (Isaac) |
| RCS torque | Angular acceleration = torque / inertia, per axis | GPU (Isaac) |
| Propellant | Mass flow = thrust / (Isp g₀); Δv matches the rocket equation | CPU (torch) |
| Contact rest | The lander settles; the sum of leg forces = m g | GPU (Isaac) |
| Orbit | Two-body energy and angular momentum stay constant; circular period = 2π√(a³/μ) | CPU (torch float64) |
| Frames | MCI → MCMF → SITE → back gives the start state | CPU |
| Slots | Same seed gives the same output; requires/provides check; truth-leak check | CPU |
| Smoke | Each scenario runs 100 steps with 64 envs | GPU (Isaac) |

CPU tests run on every push. GPU tests need Isaac Lab, so they run on a workstation or a self-hosted runner, at least once a day and before every merge to `main`.

GPU PhysX does not give bit-identical runs. Thus, results are statistics over seeds, never single runs. Each run writes:

- `manifest.yaml`: the resolved scenario config, git hash, root seed, Isaac Lab and Isaac Sim versions, GPU name.
- `episodes.parquet`: one row per episode with every metric.
- Figures come only from these files, through one command (`lunarsim figures <run_dir>`).

## Goals

v2 is done when all six goals pass.

| # | Goal | How we check it | Target |
| --- | --- | --- | --- |
| G1 | **Swappable modules.** A slot change is one line in a scenario file. | CI builds all 10 scenarios and runs each for 100 steps. | 10 of 10 pass |
| G2 | **Correct physics.** Every physics part matches a known answer. | The validation tests in the table above. | All pass, with tolerances written in the test |
| G3 | **No truth leaks.** Only `nav: truth`, metrics and the critic read `TruthState`. | Registry check plus a runtime guard. | Passes in CI |
| G4 | **Known speed.** Every scenario has a measured cost. | Env steps per second and GPU memory, per scenario, per GPU. | Target set after the M1 baseline |
| G5 | **Research ready.** The first study runs end to end. | S3: 3 encoders + classical baseline, 5 seeds. | Paper 1 results table filled |
| G6 | **Reproducible.** Any number in a paper can be rebuilt. | Manifest and episode records for every run; one figure command. | 100% of paper figures |

## Roadmap

&#91;embedded content: roadmap · 8 milestones with scenarios and gates\]

No training starts before the M1 physics tests pass and ZEM/ZEV lands. A failed gate stops the line, because every later milestone builds on it.

## Milestones

Estimates are first guesses for one developer working full time. Every new binding must also pass its unit test, the scenario smoke test, the truth-leak check and a speed log before you tick it.

### M0 Skeleton

**Goal:** an empty slot loop runs many envs in Isaac Lab. **Estimate:** 1 week.

- [ ] New repo from the Isaac Lab external-project template; `pyproject.toml`, license, CI for CPU tests.
- [ ] `core/`: message types, `Binding`, `MessageBus`, registry with the requires/provides check, clock, per-slot generators, frames stub.
- [ ] Hydra scenario loader that builds Isaac Lab config classes.
- [ ] `LanderEnv(DirectRLEnv)` with null bindings, a box on flat ground, N envs.

**Gate:** 1,024 envs step without error; core tests pass.

### M1a Lander and classical S1

**Goal:** land from 50 m with true state and a classical controller, on physics that is proven correct. No RL yet. **Estimate:** 3 weeks.

- [ ] Lander preset from public Apollo LM data, each number with its source; `Articulation` built from the preset (body, 4 legs, struts).
- [ ] `engine_rcs_pulse` actuators: throttle range, lag, minimum impulse, valve delay, propellant use.
- [ ] `flat` and `procedural` surfaces; `terminal_box` start; `truth` navigation; `touchdown` metric.
- [ ] Physics validation tests (free fall, hover, RCS torque, propellant, contact rest), written before the controller.
- [ ] `zem_zev` agent (vertical and lateral guidance plus an attitude loop on the RCS).
- [ ] Run manifest, episode records, figure command.
- [ ] Scenario S1 with `agent: zem_zev`.

**Gate:** G2 tests pass; ZEM/ZEV lands from nominal starts, and its success rate over the full `terminal_box` range is reported; G4 speed baseline.

### M1b RL on S1

**Goal:** the first learned agent on the same S1 env. Only the agent and reward config change. **Estimate:** 1 week.

- [ ] `ppo_mlp` with rsl\_rl; reward terms set in the scenario file.
- [ ] Train and evaluate scripts; 5 seeds.
- [ ] PPO vs ZEM/ZEV on the same held-out start states.

**Gate:** PPO results for 5 seeds next to ZEM/ZEV, with no change to env code between M1a and M1b.

### M2 Sensors and EKF

**Goal:** the agent flies from an estimate. **Estimate:** 3 weeks.

- [ ] `imu`, `radar_alt`, `doppler_vel`, `lidar_cheap` sensor bindings, each with rate, noise, `valid` and `stamp`.
- [ ] Clock with a rate per sensor.
- [ ] `ekf` navigation; a consistency test (NEES) on recorded truth.
- [ ] Critic observation group with `TruthState`.
- [ ] Scenario S2; S1 policy vs S2 policy on the same seeds.

**Gate:** the EKF consistency test passes; navigation error is logged.

### M3 Faults and history agents

**Goal:** run the sensor-failure study (paper 1). **Estimate:** 4 weeks.

- [ ] `intermittent`, `persistent` and `delay` faults, seeded schedules, held-out test set.
- [ ] `ekf_fdi` (chi-square gate) and `ekf_late_fusion` (state buffer and replay).
- [ ] History buffer and validity mask; `ppo_gru`, `ppo_attention`.
- [ ] Scenarios S3 and S4; `fault_response` and `nav_error` metrics.
- [ ] Paper 1 results: 3 encoders + classical baseline, 5 seeds.

**Gate:** G5 passes; paper 1 draft is complete.

### M4 Real terrain, hazards and agents

**Goal:** land on safe sites on real terrain, and compare agent families (paper 2). **Estimate:** 5 weeks.

- [ ] `dem_tile` surface: download and tile public LROC NAC DTMs; true hazard maps (slope, roughness, footpad height difference).
- [ ] `hazard_map` navigation from LiDAR scans; `site_pick_divert` agent.
- [ ] Agent adapters: `sac`, `dreamer`, `tdmpc2`; equal-budget sweep runner.
- [ ] Scenarios S6 and S9; `site_safety` metric.

**Gate:** the agent sweep is done at equal budget; paper 2 results table is filled.

### M5 Plume and footpads

**Goal:** model the last metres with plume forces, dust and sinkage. **Estimate:** 4 weeks.

- [ ] `plume_l1` surface: force and torque against altitude and thrust, ejecta density, erosion flag.
- [ ] `bekker_pads`: pressure-sinkage under each footpad.
- [ ] `dust` fault driven by ejecta density.
- [ ] L2 check scene with PhysX particles, a few envs; L3 data plan.
- [ ] Scenario S8; `dust` metric.

**Gate:** the L1 vs L2 check is written up.

### M6 Orbit, phases and TRN

**Goal:** fly from orbit to touchdown, and navigate by terrain. **Estimate:** 6 weeks.

- [ ] `orbit_f64` torch propagator; orbit tests.
- [ ] Frames MCI, MCMF, SITE; handoff with a jump check; `hybrid` plant with a per-env mode mask.
- [ ] Phase manager: `phases:` list, `until:` gates, saved states; `from_log` start.
- [ ] `camera_cheap` sensor, `trn_ekf` navigation, map-error model.
- [ ] Scenarios S5 and S7.

**Gate:** the handoff check passes; S7 runs from orbit to touchdown.

### M7 ROS 2 bridge

**Goal:** run the same GNC code as ROS 2 nodes. **Estimate:** 2 weeks.

- [ ] `ros2_bridge` binding for navigation and agent; sim clock topic.
- [ ] EKF node and policy node (ONNX export).
- [ ] Parity and timing tests; scenario S10.

**Gate:** the parity test passes.

The estimates add up to 29 weeks for one person in order. M5, M6 and M7 do not depend on each other, so they can run in parallel.

## Risks and mitigations

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Lander contact is unstable in PhysX (bounce, jitter at rest) | Touchdown metrics are noise | Contact-rest test in M1; tune solver iterations, contact offsets and strut damping before any training |
| New physics has bugs that no old result can reveal | Wrong papers | G2 validation tests against known answers; ZEM/ZEV must land before RL starts |
| Isaac Lab API changes between versions | Code breaks on upgrade | Pin Isaac Sim and Isaac Lab versions in the manifest and in CI; upgrade in one planned step |
| Camera and RTX sensors are slow | S5 and S6 training is slow | Train with `RayCaster` and `camera_cheap`; use RTX for final tests only |
| Model-based libraries do not fit the batched env | S9 slips | Adapters see only `Observation` and `Command`; test each on S1 first |
| Plume L1 is not trusted | S8 claims are weak | L2 check, L3 fit to published data, model level named in every result |
| Scope is too large for one person | Late milestones never finish | Papers at M3 and M4; M5–M7 can wait |

## Open decisions

Settle these at M0. The first option is the recommendation.

- [ ] **Repo name and host:** a new repo (for example `lunarsim2`), or a fresh `main` in a new repo under the lab's account.
- [ ] **License:** BSD-3 (the same as Isaac Lab), or Apache-2.0.
- [ ] **Isaac versions:** the newest Isaac Sim and Isaac Lab release that runs on your GPUs; pin them.
- [ ] **Attention encoder:** a small custom transformer in rsl\_rl's actor-critic, or skrl's model classes.
- [ ] **Lander preset:** Apollo LM first (well documented), or a smaller generic lander first (faster tuning).
- [ ] **Paper order:** S3 first, then S9; or both on the S3 env together.
