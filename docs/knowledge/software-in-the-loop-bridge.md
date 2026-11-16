---
id: software-in-the-loop-bridge
title: Software-in-the-loop bridge (S10, milestone 7)
type: component
status: current
updated: 2026-11-16
summary: Milestone 7 - nav and agent ros2_bridge bindings that run the unchanged ekf_fdi and ppo_gru (ONNX export) as separate node processes; loopback transport (tested) and ROS 2 transport (written but untested, as ROS 2 is not installed); deadline model; parity and timing tests of S10.
---

# Software-in-the-loop bridge (S10)

Design M7: "run the same GNC code as ROS 2 nodes"; gate: "the parity test passes". Design S10:
"Does the GNC code give the same landing when it runs as ROS 2 nodes, with real message timing,
as it gives inside the training loop?"

- depends on: [[slot-loop-architecture]]
- uses: [[faults-and-detection]]
- uses: [[ppo-landing-agent]]
- results: [[paper-one-results]]

## How it works

| Part | File | What it does |
| --- | --- | --- |
| `nav: ros2_bridge` | `slots/navigation/ros2_bridge.py` | Declares the inner binding's `requires` and `provides` (so the slot-loop and truth-leak checks see the real data flow; a binding that reads truth cannot be bridged). Each step sends the input messages that changed (the node keeps the rest), waits for the estimate, writes it on the bus. Resets carry the changed messages too (a filter reads its start state at reset) |
| `agent: ros2_bridge` | `agents/bridge_agent.py` | Sends the observation, flies the node's command. `policy_format: onnx` makes the node's agent use the ONNX export instead of TorchScript; observation scaling, action adapter and attitude controller are the same code |
| Node programs | `bridge/node_programs.py` | Rebuild the inner binding from a picklable spec on the CPU, on the node's own message bus; never import Isaac Sim |
| Transports | `bridge/message_transport.py`, `bridge/serve_node.py` | `loopback`: the node is a separate interpreter (`python -m lunarsim.bridge.serve_node`) on a local authenticated socket, payload = pickled bytes. `ros2`: the same bytes in `std_msgs/UInt8MultiArray` on `/lunarsim/<node>/request` and `/reply`, sim time on `/clock` |
| Deadline model | `bridge/deadline_model.py` | Delay = measured round trip + added latency + seeded jitter; a reply above `timeout_ms` is not used and the binding keeps its previous output for that step. `timeout_ms: null` uses every reply (parity) |
| ONNX export | `bridge/onnx_policy.py`, `scripts/export_policy_onnx.py` | `policy.pt` -> `policy.onnx` (opset 17, any batch size), checked against TorchScript; run with onnxruntime if installed, else the `onnx` reference evaluator (no new install needed) |
| Scenario | `configs/scenarios/s10_ros2_sil.yaml` | S3 with both bridges; the inner configs are the real ones (`- /nav@nav.node: ekf_fdi`, `- /agent@agent.node: ppo_gru`) |
| Runner | `scripts/run_software_in_the_loop.py` | Reference, parity and two timing variants on the same S3 seeds; trajectories, touchdown metrics, timing |

Why a loopback transport: ROS 2 is not installed on either machine (installing it needs `sudo`;
see [[pending-user-decisions]]). The loopback node is a real separate process with real message
timing, and the payload is the same bytes the ROS 2 transport would send, so the ROS 2 path
differs only in the transport class.

Deviation: the design lists `metrics: [touchdown, latency]`. One metrics binding per slot exists
here, so S10 keeps the S3 metric and the bridges add per-episode timing columns
(`bridge_navigation_*`, `bridge_agent_*`: replies, deadline misses, longest round trip).

## Tests

- CPU (`tests/unit/test_software_in_the_loop_bridge.py`): `ekf_fdi` in a loopback node gives
  **exactly** the in-process estimates on the synthetic flight (200 landers, 60 steps); late
  replies keep the previous estimate; a bridged `zem_zev` gives exactly the in-process commands;
  a GRU policy's ONNX export matches TorchScript to 1e-5.
- GPU (`tests/smoke/test_software_in_the_loop.py`, needs the paper 1 GRU policy in `runs/`): the
  bridged run stays within 5 cm of the in-loop run for 3 s from the same S3 start.

## Results (2026-11-16, 20 S3 episodes, seed 1000, one lander, paper 1 GRU seed 0)

Runs `results/s10_seed1000_*` (commit `50d92e8`) and `results/s10_diagnosis_seed1000_*` (commit
`9b77e95`); summaries `s10_parity_and_timing.yaml` and `s10_parity_diagnosis.yaml` there.

Parity, each run against the in-loop reference (`ekf_fdi` compiled on the GPU, TorchScript policy):

| Run | Largest position difference | Median of the largest | Touchdown speed difference | Same outcome |
| --- | --- | --- | --- | --- |
| Reference run again | 0.000 m | 0 | 0 | 20/20 |
| Bridge with the simulator's arithmetic (`parity_same_numerics`: nodes on the GPU, filter compiled, TorchScript) | **0.000 m** | 0 | 0 | 20/20 |
| In-loop, only the filter not compiled (`reference_eager`) | 3.1 m | 1.4 m | 0.6 m/s | 18/20 |
| Bridge, flight-like nodes (`parity`: CPU, eager filter, ONNX policy) | 5.2 m | 1.5 m | 0.7 m/s | 20/20 |

The bridge itself is exact: with the same arithmetic all 20 episodes match bit for bit. Any
change of float rounding (even switching off compilation inside the simulator) starts a 1 mm
difference within 0.2 to 3.5 s of flight, which the closed loop (fault switching, policy
decisions) grows to metres; landings stay equally good (success 90 % in both). The tolerance set
before the run (1 m, 0.1 m/s) therefore fails for the CPU/ONNX nodes; how to state the gate is
[[pending-user-decisions]] item 4.

Timing (loopback, this machine): navigation round trip median 3.9 ms (99 % below 4.6 ms, largest
40 ms), policy 1.0 ms (1.8 ms, 3.2 ms). With `timeout_ms: 20` and the measured times, 3 of 17,669
navigation replies missed (the rare slow ones) and no agent reply. Stress test (15 ms added
latency, 5 ms jitter): 16 % of navigation and 15 % of agent replies missed, the bridge held its
last output each time; success 100 % (reference 90 %), mean touchdown speed 0.80 m/s (reference
0.75), landing error 8.2 m (8.7): no loss for this loop at this rate of misses.
