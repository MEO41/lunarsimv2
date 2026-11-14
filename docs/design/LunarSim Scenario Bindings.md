# LunarSim: Scenario Bindings

Nov 14, 2026 · @Muhammed Emin Onay

## How bindings work

A scenario binds one implementation to each of nine slots. Slots exchange only typed messages. Thus, you can replace one binding and its neighbours do not change.

&#91;embedded content: the slot loop · 9 slots, 1 truth bypass\]

The loop runs once per control step. The dashed line is the truth bypass. The critic always gets the true state. The actor gets it only when the navigation slot is set to `truth`.

Every scenario diagram below uses this same layout. A coloured box is the slot that the scenario studies. A dashed box is off or bypassed.

| Slot | Bindings |
| --- | --- |
| Start | `orbit_elements`, `pdi_state`, `terminal_box`, `from_log` |
| Plant | `orbit_f64` (GPU propagator), `local_f64`, `physx_site`, `hybrid` (handoff between them) |
| Surface | terrain: `flat`, `dem_tile`, `procedural`; contact: `rigid`, `bekker`; plume: `none`, `L1`, `L2` |
| Sensors | `imu`, `radar_alt`, `doppler_vel`, `star_tracker`, `lidar_cheap`, `lidar_rtx`, `camera_cheap`, `camera_rtx` |
| Faults | `none`, `intermittent`, `persistent`, `delay`, `scripted`, `dust` (driven by the plume) |
| Navigation | `truth`, `truth_noise`, `ekf`, `ekf_fdi`, `ekf_late_fusion`, `trn_ekf`, `hazard_map`, `ros2_bridge` |
| Agent | `zem_zev`, `lqr`, `mpc`, `ppo` (MLP, GRU, attention), `sac`, `dreamer`, `tdmpc2`, `ros2_bridge` |
| Actuators | `ideal`, `engine` (throttle limits, lag), `rcs_pulse` (minimum impulse, valve delay) |
| Metrics | `touchdown`, `landing_error`, `nav_error`, `fault_response`, `dust`, `latency` |

### Message types

All tensors have the environment count `N` as the first dimension.

```python
@dataclass
class TruthState:
    frame: str            # "MCI", "MCMF" or "SITE"
    t: Tensor             # (N,)    float64 in orbit, float32 in PhysX
    r: Tensor; v: Tensor  # (N, 3)
    q: Tensor; w: Tensor  # (N, 4), (N, 3)
    mass: Tensor          # (N,)

@dataclass
class Measurement:
    sensor: str
    value: Tensor         # (N, d)
    valid: Tensor         # (N,) bool, set to False by a fault
    stamp: Tensor         # (N,) capture time, older than now if delayed
    cov: Tensor           # (N, d, d)

@dataclass
class NavEstimate:
    mean: Tensor; cov: Tensor
    health: Tensor        # (N, n_sensors) fault flags

@dataclass
class Command:
    throttle: Tensor; gimbal: Tensor
    rcs: Tensor           # (N, n_jets) pulse on or off

@dataclass
class Wrench:
    force: Tensor; torque: Tensor   # body frame
```

The agent does not read `NavEstimate` directly. The Isaac Lab observation manager builds `Observation` from it (or from `TruthState` in truth mode), with a history buffer and a validity mask. A second observation group gives `TruthState` to the critic.

### One interface for every binding

```python
class Binding(Protocol):
    requires: set[str]    # message types it reads
    provides: set[str]    # message types it writes
    def reset(self, env_ids: Tensor, rng: Generator) -> None: ...
    def step(self, msgs: dict[str, Any], dt: float) -> dict[str, Any]: ...
```

Before a run, a check walks the loop. It stops the run if a binding requires a type that no slot provides.

### Scenario files

A scenario file (Hydra) lists one binding per slot. The command line can change any slot:

```bash
python train.py scenario=s3_faults agent=ppo_attention faults=persistent seed=3
```

The numbers in the config examples below are example values. Set them from your mission reference.

## Binding matrix

Ten scenarios use the same nine slots. Each row changes only the slots in bold. The other slots stay fixed, so you can compare results between rows.

| Scenario | Start | Plant | Surface | Sensors | Faults | Navigation | Agent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 Terminal, truth | terminal box | PhysX | flat or DEM | off | off | **truth** | **PPO vs ZEM/ZEV** |
| S2 Terminal, EKF | terminal box | PhysX | DEM | IMU, altimeter, velocimeter | off | **EKF** | PPO |
| S3 Sensor failures | terminal box | PhysX | DEM | IMU, altimeter, velocimeter | **fault schedule** | EKF + fault check | **PPO: MLP, GRU, attention** |
| S4 Late fusion | terminal box | PhysX | DEM | multi-rate | delay, jitter | **late-fusion EKF** | PPO, GRU |
| S5 TRN descent | descent gate | f64, site frame | DEM, rendered | camera, altimeter, IMU | image faults | **TRN + EKF** | ZEM/ZEV or PPO |
| S6 Hazard avoidance | approach gate | PhysX | procedural hazards | flash LiDAR | optional | **hazard map** | **site pick + divert** |
| S7 Full mission | orbit | **f64 then PhysX** | DEM | per phase | optional | per phase | per phase |
| S8 Plume, regolith | terminal box | PhysX, footpads | **regolith + plume L1** | LiDAR, camera, altimeter | dust (from plume) | EKF | PPO, GRU |
| S9 MB vs MF | as S3 | as S3 | as S3 | as S3 | as S3 (held-out set) | as S3 | **PPO, SAC, Dreamer, TD-MPC2, MPC** |
| S10 ROS 2 | terminal box | PhysX, 1 env | DEM | IMU, altimeter | fault schedule | **ROS 2 EKF node** | **ROS 2 policy node** |

Actuators are `engine` + `rcs_pulse` in every row. Metrics change with the question and are named in each section.

## S1 Terminal touchdown with truth state

**Question:** Can the agent make a soft contact without a tip-over when it knows the exact state? This is the first scenario to build, and every later one starts from it.

&#91;embedded content: S1 bindings · sensor chain off, truth to the actor\]

The sensor chain is not built. The truth bypass feeds the actor directly, so all errors come from the agent, the actuators and the contact.

```yaml
# scenarios/s1_terminal_truth.yaml
defaults:
  - start: terminal_box
  - plant: physx_site
  - surface: dem_tile        # or flat
  - sensors: none
  - faults: none
  - nav: truth
  - agent: ppo_mlp           # baseline run: zem_zev
  - actuators: engine_rcs
  - metrics: touchdown
start:
  altitude_m: [5, 50]
  v_vertical_mps: [-6, 0]
  v_horizontal_mps: [-2, 2]
  tilt_deg: [0, 10]
```

**Measure:** vertical and horizontal speed at contact, tilt and tilt rate, tip-over rate, fuel used, success rate.

**Variants:**

- Change `surface` from `flat` to `dem_tile` to add slope at the footpads.
- Change `actuators` from `ideal` to `engine_rcs` to see the cost of lag and minimum impulse.
- Change `contact` from `rigid` to `bekker` to add footpad sinkage.

## S2 Terminal touchdown with sensors and EKF

**Question:** How much touchdown quality does the agent lose when it sees an estimate and not the truth? S2 is S1 with three slots changed.

&#91;embedded content: S2 bindings · full sensor chain, no faults\]

The sensors, the EKF and the agent's observation change. The truth bypass now goes to the critic only (asymmetric actor-critic). The faults slot passes data through with no change.

```yaml
# scenarios/s2_terminal_ekf.yaml
defaults:
  - s1_terminal_truth         # start from S1
  - override sensors: [imu, radar_alt, doppler_vel]
  - override nav: ekf
  - override agent: ppo_mlp
start:
  nav_error:                  # error in the first estimate
    position_m: 1.0
    velocity_mps: 0.2
```

**Measure:** the S1 touchdown metrics, plus position and velocity error at contact. Compare one S1 policy and one S2 policy on the same seeds.

**Variants:**

- Change `nav` to `truth_noise` (truth plus white noise) to separate noise from filter lag.
- Train the policy in S1 and test it in S2 to measure the sim-to-sensor gap.

## S3 Sensor failures: intermittent and persistent

**Question:** What is the safe landing performance under intermittent and persistent sensor failures, and which history encoder keeps it highest? This is the first paper scenario.

&#91;embedded content: S3 bindings · faults varied, encoder compared\]

Two slots are studied. The faults slot is the factor that you vary. The agent slot is the method that you compare. A classical row (`agent: zem_zev`, `nav: ekf_fdi`) gives the baseline.

| Fault class | Fault types | Parameters |
| --- | --- | --- |
| Intermittent | dropout burst, delay spike, outlier spike | rate, burst length |
| Persistent | bias, drift, stuck value, scale error, total loss | onset altitude, size |

Each measurement keeps its `valid` flag and `stamp`. The agent gets the flag and the data age in its observation, so it can learn to trust or ignore a sensor.

```yaml
# scenarios/s3_faults.yaml
defaults:
  - s2_terminal_ekf
  - override faults: schedule
  - override nav: ekf_fdi
  - override agent: ppo_gru     # sweep: ppo_mlp, ppo_gru, ppo_attention
  - override metrics: fault_response
faults:
  seed: ${seed}                 # same faults for every agent
  classes: [intermittent, persistent]
  targets: [radar_alt, doppler_vel]
  onset_altitude_m: [5, 40]
agent:
  history_len: 32               # steps
```

**Measure:** success rate per fault type and onset altitude, touchdown speed, time to detect, and the success drop compared with no faults.

**Variants:**

- Train on intermittent faults only, then test on persistent faults (generalisation).
- Remove the validity flag from the observation, to see if the encoder can find faults by itself.

## S4 Late fusion with asynchronous, delayed sensors

**Question:** Does late fusion keep the estimate and the landing correct when sensors run at different rates and their data arrives late or out of order?

&#91;embedded content: S4 bindings · multi-rate sensors, late-fusion EKF\]

"Late fusion" has two meanings. S4 supports both, in two different slots:

| Meaning | Slot | Binding | What it does |
| --- | --- | --- | --- |
| Estimator late fusion | Navigation | `ekf_late_fusion` | Keeps a buffer of past states. A late measurement is applied at its capture time, then the filter replays to now. |
| Network late fusion | Agent | `ppo_late_fusion` | One encoder per sensor. The features join only at the policy head. Compare with one early-fusion encoder. |

The delay comes from the `stamp` field. The faults slot can set a delay per sensor without any change to the sensor binding.

```yaml
# scenarios/s4_late_fusion.yaml
defaults:
  - s2_terminal_ekf
  - override sensors: [imu, radar_alt, doppler_vel, camera_cheap]
  - override faults: delay
  - override nav: ekf_late_fusion   # compare: ekf (drops late data)
  - override agent: ppo_gru
  - override metrics: nav_error
sensors:
  rate_hz: {imu: 200, radar_alt: 20, doppler_vel: 10, camera_cheap: 5}
faults:
  delay_ms: {doppler_vel: [50, 200], camera_cheap: [100, 300]}
  jitter_ms: 20
```

**Measure:** position and velocity error over time, the error at contact, and touchdown success. Compare three filters: drop late data, use late data as if it is new, and late fusion.

**Variants:**

- Combine with the S3 fault schedule: a sensor that is late and also faulty.
- Swap `nav: truth` and `agent: ppo_late_fusion` to test network fusion with no filter.

## S5 Terrain-relative navigation descent

**Question:** How accurately can the lander reach its target when position comes from matching camera images to an onboard map, and not from truth?

&#91;embedded content: S5 bindings · TRN in the navigation slot, map from the surface\]

The surface slot gives two things: the true terrain for the camera, and an onboard map with added error for TRN. Thus, the map error is a parameter and not a separate module.

The camera is slow. Train with `camera_cheap` (a shaded height map rendered on the GPU). Run the final tests with `camera_rtx`. Use the same seeds for both, to measure the render gap.

```yaml
# scenarios/s5_trn.yaml
defaults:
  - start: descent_gate          # or from_log: states saved by S7
  - plant: local_f64
  - surface: dem_tile_rendered
  - sensors: [imu, radar_alt, camera_cheap]
  - faults: image                # blur, glare, low sun
  - nav: trn_ekf
  - agent: zem_zev               # or ppo_gru
  - actuators: engine_rcs
  - metrics: [landing_error, nav_error]
surface:
  dem: lola_site_tile
  map_error: {offset_m: [0, 20], resolution_m: 5}
  sun_elevation_deg: [3, 30]
```

**Measure:** TRN position error against altitude, landing error at contact, the number of TRN fixes rejected by the EKF, and the rate of lost lock.

**Variants:**

- Change `nav` to `truth_noise` with the same error level as TRN, to see if TRN error structure matters or only its size.
- Sweep sun elevation to find where TRN fails, then add `faults: image` on top.

## S6 Hazard detection and avoidance with retargeting

**Question:** Can the lander find a safe site from LiDAR data and divert to it with the fuel it has left?

&#91;embedded content: S6 bindings · LiDAR hazard map and divert agent\]

The navigation slot now outputs a hazard map as well as the state estimate. The agent picks a site from that map and flies to it. The surface keeps the true hazard map, so the metric can check the site that the lander touched.

```yaml
# scenarios/s6_hda.yaml
defaults:
  - start: approach_gate
  - plant: physx_site
  - surface: procedural_hazards
  - sensors: [imu, radar_alt, lidar_cheap]   # final tests: lidar_rtx
  - faults: none                             # or intermittent on lidar
  - nav: hazard_map
  - agent: site_pick_divert                  # baseline: rule_based_hda
  - actuators: engine_rcs
  - metrics: [touchdown, site_safety]
surface:
  rock_density: [0.0, 0.1]      # fraction of area
  max_safe_slope_deg: 10
```

**Measure:** fraction of landings on a safe site, slope and rock height under the footpads, divert distance, fuel for the divert, and time to pick the site.

**Variants:**

- Set `nav: truth` with the true hazard map. This gives an upper limit for the agent alone.
- Add LiDAR dropout from S3 during the divert.

## S7 Full mission from orbit with phase handoffs

**Question:** How do errors from orbit, braking and approach add up at touchdown, and which phase causes the most landing error?

&#91;embedded content: S7 phases · 5 phases, 4 gates, 1 frame handoff\]

A phase is a list of slot overrides and a gate condition. A phase manager checks the gate each step. When the gate is met, it saves the state and swaps the bindings that the next phase lists. Slots that a phase does not list keep their binding.

At the low gate, the plant changes from the f64 propagator to PhysX. The handoff converts the state from the Moon-fixed frame to the site frame and from float64 to float32. A check stops the run if position, velocity or energy jump at the handoff.

```yaml
# scenarios/s7_full_mission.yaml
start: orbit_elements
phases:
  - name: coast
    plant: orbit_f64              # Moon-inertial
    nav: imu_star_tracker
    agent: scripted_burns
    until: {event: doi_done}
  - name: descent_orbit
    until: {event: pdi_time}
  - name: braking
    plant: orbit_f64              # Moon-fixed
    nav: trn_ekf
    agent: braking_guidance
    until: {altitude_m: 2000}     # high gate
  - name: approach
    nav: [trn_ekf, hazard_map]
    agent: divert_guidance
    until: {altitude_m: 100}      # low gate
  - name: terminal
    plant: physx_site
    nav: ekf
    agent: ppo_gru
    until: {event: touchdown}
handoff:
  at: low_gate
  convert: mcmf_to_site
  check: [position, velocity, energy]
save_states_at: [pdi, high_gate, low_gate]
```

The saved states connect S7 to the other scenarios. S5 can start from the states saved at PDI. S1, S3 and S8 can start from the states saved at the low gate. Thus, short scenarios use the true dispersion of the full mission, not a guessed box.

**Measure:** state dispersion at each gate, fuel used per phase, landing error, and touchdown metrics. Repeat with one phase set to `nav: truth` to find the share of error from each phase.

**Training note:** do not train one policy over the full mission. Train each phase agent in its own short scenario, then run S7 to test the chain.

## S8 Plume and regolith at final contact

**Question:** How do plume forces, dust and footpad sinkage change the last metres of the landing, and how do RCS pulses change the dust?

&#91;embedded content: S8 bindings · plume drives the dust fault\]

This scenario has one link that the others do not have: the surface sends dust density to the faults slot. The dust fault then degrades LiDAR and camera data. Thus, a physical effect becomes a sensor fault with no change to the sensor bindings.

| Plume level | What it models | Cost | Use |
| --- | --- | --- | --- |
| L1 | Force and torque as a function of altitude and thrust; ejecta density; Bekker sinkage | Low, all envs | Training and main results |
| L2 | PhysX particles under the nozzle | High, few envs | Visual checks of L1 |
| L3 | Offline DEM or CFD runs, fitted to a surrogate for L1 | Offline | Calibrate L1 |

```yaml
# scenarios/s8_plume.yaml
defaults:
  - s3_faults
  - override surface: regolith_plume_l1
  - override sensors: [imu, radar_alt, lidar_cheap, camera_cheap]
  - override faults: dust
  - override agent: ppo_gru
  - override metrics: [touchdown, dust]
surface:
  contact: bekker
  plume:
    model: l1_surrogate
    erosion: true
actuators:
  rcs_min_impulse_Ns: 0.5
  valve_delay_ms: 20
```

**Measure:** touchdown metrics, footpad sinkage, time with sensors blinded by dust, ejecta mass, and the plume torque at contact.

**Variants:**

- Turn `plume` off and compare with L1, using the same seeds. The difference is the plume effect.
- Change the RCS pulse width and the engine cut-off altitude, then compare dust and tilt at contact.

## S9 Model-based vs model-free agent comparison

**Question:** Do model-based agents learn a safe landing with fewer steps, and do they stay safe on faults that they did not see in training?

&#91;embedded content: S9 bindings · agent swept, all else frozen\]

S9 freezes every slot from S3 and sweeps only the agent. This is the cleanest use of the slot design: any difference in the results comes from the agent.

| Agent binding | Type | History encoder | Uses truth |
| --- | --- | --- | --- |
| `ppo` | model-free, on-policy | MLP, GRU or attention | critic only |
| `sac` | model-free, off-policy | GRU or attention | critic only |
| `dreamer` | model-based, world model (RSSM) | recurrent latent | not in the actor |
| `tdmpc2` | model-based, latent planning | latent | not in the actor |
| `mpc` | classical, known model | EKF estimate | no |

All agents read the same `Observation` and write the same `Command`. A wrapper converts the Isaac Lab vector env to each library's interface, so the env code does not change.

```yaml
# scenarios/s9_mb_vs_mf.yaml
defaults:
  - s3_faults
  - override agent: dreamer    # sweep: ppo_gru, sac_gru, dreamer, tdmpc2, mpc
budget:
  env_steps: 5e7               # same for every agent
faults:
  train_set: [dropout, bias]
  test_set: [drift, stuck, delay]   # held out
seeds: [0, 1, 2, 3, 4]
```

**Measure:** env steps to reach a target success rate, final success rate, success on held-out faults, wall-clock time, and GPU memory.

**Variants:**

- Run the sweep in S1 (truth) and in S3 (faults). The gap shows how much each agent depends on clean data.
- Probe the world model: compare its predicted altitude with the truth during a fault.

## S10 ROS 2 software-in-the-loop

**Question:** Does the GNC code give the same landing when it runs as ROS 2 nodes, with real message timing, as it gives inside the training loop?

&#91;embedded content: S10 bindings · navigation and agent run as ROS 2 nodes\]

The `ros2_bridge` binding has the same interface as every other binding. It publishes the slot's input as a topic and waits for the node's output. Thus, the EKF and the policy move out of the simulator with no change to the other slots.

ROS 2 is not in the training loop. It runs one env in real time, so use it for tests only:

1. **Parity test:** run S3 seeds with `nav: ekf_fdi` and `agent: ppo_gru`, then with the bridge. The trajectories must match within a set tolerance.
2. **Timing test:** add the real message latency and executor jitter. Measure the change in touchdown metrics.
3. **Hardware-in-the-loop (later):** run the nodes on the flight computer. The simulator side does not change.

```yaml
# scenarios/s10_ros2_sil.yaml
defaults:
  - s3_faults
  - override nav: ros2_bridge
  - override agent: ros2_bridge
  - override metrics: [touchdown, latency]
env:
  num_envs: 1
  clock: sim_time               # /clock topic, so nodes use sim time
ros2_bridge:
  nav_node: lunarsim_ekf
  agent_node: lunarsim_policy   # loads the ONNX export of ppo_gru
  timeout_ms: 20
```

**Measure:** position difference against the Python run, message latency, deadline misses, and touchdown metrics.

## Rules that keep the modules swappable

These ten rules apply to every binding. If a new binding breaks one, it will break the scenarios that use it.

1. **One input type and one output type per slot.** A binding cannot read a message from a slot that is not its neighbour. The truth bypass is the only exception.
2. **Declare `requires` and `provides`.** The check before the run uses them to find a missing link.
3. **Only two bindings read `TruthState`:** `nav: truth` and the critic observation group. Metrics read it after the step. Any other read is a bug, because it leaks truth into a test of sensors.
4. **Every slot has its own random stream** from one root seed. Then a change to the agent does not change the faults or the start states.
5. **One clock.** Each sensor has its own rate and delay, but all use the simulation time. Store capture time in `stamp`, not arrival time.
6. **All tensors are batched over `N` envs** and stay on the GPU. A binding that needs a CPU loop is for tests only (like `ros2_bridge`).
7. **Frames are explicit.** Every `TruthState` carries its frame. Only the handoff function changes frames.
8. **A cheap and an expensive version for slow parts:** `camera_cheap` and `camera_rtx`, `lidar_cheap` and `lidar_rtx`, plume L1 and L2. Train with cheap ones. Test with both on the same seeds.
9. **Log every slot output** for a small set of envs. Then any scenario can replay a run and swap one slot (for example, test a new EKF on logged sensor data).
10. **Scenario files are versioned with the results.** A result table names the scenario file and its git hash.

### Build order for the scenarios

S1 → S2 → S3 gives the first paper with only the terminal phase. S9 reuses S3 with no new env code. S4 and S8 add one slot each. S5, S6 and S7 need the f64 plant, rendering and the phase manager, so they come later. S10 needs only the bridge binding and can come at any time after S3.
