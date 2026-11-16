# SPDX-License-Identifier: BSD-3-Clause
"""S10 parity and timing tests (milestone 7): the GNC code in the loop and as separate nodes.

Example::

    python scripts/run_software_in_the_loop.py episodes=20 seed=1000 run_name=s10_seed1000

Design S10: "Parity test: run S3 seeds with nav: ekf_fdi and agent: ppo_gru, then with the bridge.
The trajectories must match within a set tolerance. Timing test: add the real message latency
and executor jitter. Measure the change in touchdown metrics."

One environment flies ``episodes`` S3 episodes (same seed, so the same starts and faults) in four
variants:

- ``reference``: scenario ``s3_faults`` with ``agent: ppo_gru`` (TorchScript), all in the loop;
- ``parity``: ``s10_ros2_sil``: ``ekf_fdi`` and ``ppo_gru`` (ONNX) as loopback nodes, every reply used;
- ``timing_measured``: as parity with ``timeout_ms: 20``, the real measured round trips decide;
- ``timing_stress``: ``timeout_ms: 20`` plus 15 ms added latency and 5 ms jitter (model choices;
  repeatable: the measured round trips are recorded but do not decide).

Parity tolerance (set before the run, model choices): every episode has the same outcome, the
largest position difference along the trajectory is below 1.0 m and the touchdown speeds differ by
less than 0.1 m/s. The reference runs the filter compiled on the GPU and the TorchScript policy;
the nodes run it eagerly on the CPU and the ONNX policy, so the results are close, not equal (the
CPU unit test shows the bridge itself changes nothing: equality there is exact).

Writes ``runs/<run_name>_<variant>/`` (manifest, ``episodes.parquet``, ``trajectories.npz``) and
``runs/<run_name>_parity/summary.yaml`` with the parity and timing results.
"""

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="S10 parity and timing tests.")
AppLauncher.add_app_launcher_args(parser)
arguments, script_arguments = parser.parse_known_args()
simulation_app = AppLauncher(arguments).app

import numpy  # noqa: E402
import pandas  # noqa: E402
import yaml  # noqa: E402

import lunarsim.agents.bridge_agent  # noqa: E402, F401  (registers agent: ros2_bridge)
import lunarsim.slots.navigation.ros2_bridge  # noqa: E402, F401  (registers nav: ros2_bridge)
from lunarsim.core.message_types import MessageKey  # noqa: E402
from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario  # noqa: E402
from lunarsim.tools.episode_records import EPISODES_FILE_NAME  # noqa: E402
from lunarsim.tools.episode_runner import build_agent, collect_episodes  # noqa: E402
from lunarsim.tools.mission_phases import merge_agent_decision_records  # noqa: E402
from lunarsim.tools.run_manifest import write_manifest  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY = "runs/ppo_gru_s3_seed0_to1500/policy.pt"
ONNX = "runs/ppo_gru_s3_seed0_to1500/policy.onnx"
POSITION_TOLERANCE_M = 1.0
TOUCHDOWN_SPEED_TOLERANCE_MPS = 0.1
DEADLINE = ["nav.timeout_ms=20", "agent.timeout_ms=20"]
VARIANTS = {
    "reference": ("s3_faults", ["agent=ppo_gru", f"agent.policy_path={POLICY}"]),
    "parity": ("s10_ros2_sil", [f"agent.node.policy_path={POLICY}", f"agent.onnx_path={ONNX}"]),
    "timing_measured": ("s10_ros2_sil", [f"agent.node.policy_path={POLICY}", f"agent.onnx_path={ONNX}"] + DEADLINE),
    "timing_stress": (
        "s10_ros2_sil",
        [f"agent.node.policy_path={POLICY}", f"agent.onnx_path={ONNX}"] + DEADLINE
        + [f"{slot}.{name}={value}" for slot in ("nav", "agent") for name, value in (("added_latency_ms", 15.0), ("jitter_ms", 5.0), ("count_measured_round_trip", "false"))],
    ),
}


class _TruthReader:
    """Reader of the true state for the trajectory record (a metrics-side reader)."""

    requires = frozenset({MessageKey.TRUTH_STATE})
    provides = frozenset()
    reads_truth = True


def run_variant(name: str, scenario_name: str, overrides: list[str], episodes: int, seed: int, run_name: str) -> tuple[pandas.DataFrame, list[numpy.ndarray], dict]:
    """Fly ``episodes`` episodes of one variant; write its run directory; return episodes, trajectories and timing."""
    scenario = load_scenario(scenario_name, ["env.num_envs=1", f"seed={seed}"] + overrides)
    environment = LanderEnv(build_lander_env_cfg(scenario, device=arguments.device))
    agent = build_agent(environment)
    reader = _TruthReader()
    trajectories: list[list[list[float]]] = [[]]
    previous_time = {"value": -1.0}

    def record(step_index: int, command) -> None:
        """Append the true position and velocity; a time that went back means a new episode."""
        truth = environment.bus.read(MessageKey.TRUTH_STATE, reader)
        now = float(truth.time[0])
        if now < previous_time["value"]:
            trajectories.append([])
        previous_time["value"] = now
        trajectories[-1].append([now] + truth.position[0].tolist() + truth.velocity[0].tolist())

    began = time.perf_counter()
    table = collect_episodes(environment, agent, episodes_per_environment=episodes, maximum_steps=episodes * (int(environment.max_episode_length) + 10), after_step=record)
    elapsed = time.perf_counter() - began
    table = merge_agent_decision_records(table, agent)
    navigation = environment.bindings["nav"]
    table = merge_agent_decision_records(table, navigation)
    timing = {}
    for label, owner in (("navigation", navigation), ("agent", agent)):
        if hasattr(owner, "deadline"):
            timing[label] = owner.deadline.summary()
    directory = REPOSITORY_ROOT / "runs" / f"{run_name}_{name}"
    write_manifest(directory, scenario, environment.random_generators.seeds(), extra={"variant": name, "wall_time_s": round(elapsed, 1), "timing": timing})
    table.to_parquet(directory / EPISODES_FILE_NAME, index=False)
    arrays = [numpy.asarray(trajectory, dtype=numpy.float64) for trajectory in trajectories if trajectory][:episodes]
    numpy.savez_compressed(directory / "trajectories.npz", *arrays)
    for owner in (agent, navigation):
        if hasattr(owner, "close"):
            owner.close()
    environment.close()
    print(f"[s10] {name}: {len(table)} episodes in {elapsed:.0f} s, success {table['success'].mean():.3f}, timing {timing}", flush=True)
    return table, arrays, timing


def compare(reference: tuple, other: tuple) -> dict:
    """Return the parity measures of ``other`` against ``reference``."""
    table_a, paths_a, _ = reference
    table_b, paths_b, _ = other
    table_a = table_a.sort_values("episode_number").reset_index(drop=True)
    table_b = table_b.sort_values("episode_number").reset_index(drop=True)
    largest = []
    for path_a, path_b in zip(paths_a, paths_b):
        steps = min(len(path_a), len(path_b))
        largest.append(float(numpy.linalg.norm(path_a[:steps, 1:4] - path_b[:steps, 1:4], axis=1).max()))
    speed_a = numpy.hypot(table_a["touchdown_vertical_speed_mps"], table_a["touchdown_horizontal_speed_mps"])
    speed_b = numpy.hypot(table_b["touchdown_vertical_speed_mps"], table_b["touchdown_horizontal_speed_mps"])
    same_outcome = (table_a["outcome"].to_numpy() == table_b["outcome"].to_numpy())
    return {
        "episodes": int(len(table_b)),
        "same_outcome_fraction": float(same_outcome.mean()),
        "largest_position_difference_m": float(max(largest) if largest else float("nan")),
        "median_of_largest_position_difference_m": float(numpy.median(largest) if largest else float("nan")),
        "largest_touchdown_speed_difference_mps": float(numpy.abs(speed_a - speed_b).max()),
        "success_reference": float(table_a["success"].mean()),
        "success": float(table_b["success"].mean()),
        "touchdown_speed_mean_mps": float(speed_b.mean()),
        "touchdown_speed_mean_reference_mps": float(speed_a.mean()),
        "landing_error_mean_m": float(numpy.hypot(table_b["touchdown_position_x_m"], table_b["touchdown_position_y_m"]).mean()),
        "landing_error_mean_reference_m": float(numpy.hypot(table_a["touchdown_position_x_m"], table_a["touchdown_position_y_m"]).mean()),
    }


def main() -> None:
    """Run the four variants and write the parity and timing summary."""
    options = {"episodes": "20", "seed": "1000", "run_name": None, "variants": ",".join(VARIANTS)}
    for argument in script_arguments:
        key, _, value = argument.partition("=")
        if key not in options:
            raise SystemExit(f"unknown argument '{argument}' (episodes=, seed=, run_name=, variants=)")
        options[key] = value
    run_name = options["run_name"] or f"s10_{time.strftime('%Y%m%d_%H%M%S')}"
    episodes, seed = int(options["episodes"]), int(options["seed"])
    if not (REPOSITORY_ROOT / ONNX).exists():
        raise SystemExit(f"{ONNX} is missing: python scripts/export_policy_onnx.py policy={POLICY} kind=gru hidden_size=128")
    results = {}
    for name in options["variants"].split(","):
        scenario_name, overrides = VARIANTS[name]
        results[name] = run_variant(name, scenario_name, overrides, episodes, seed, run_name)
    summary = {"run_name": run_name, "seed": seed, "episodes": episodes, "tolerance": {"position_m": POSITION_TOLERANCE_M, "touchdown_speed_mps": TOUCHDOWN_SPEED_TOLERANCE_MPS}}
    if "reference" in results:
        for name, result in results.items():
            if name == "reference":
                continue
            measures = compare(results["reference"], result)
            measures["timing"] = result[2]
            if name == "parity":
                measures["passed"] = bool(
                    measures["same_outcome_fraction"] == 1.0
                    and measures["largest_position_difference_m"] < POSITION_TOLERANCE_M
                    and measures["largest_touchdown_speed_difference_mps"] < TOUCHDOWN_SPEED_TOLERANCE_MPS
                )
            summary[name] = measures
    directory = REPOSITORY_ROOT / "runs" / f"{run_name}_parity"
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "summary.yaml", "w") as file:
        yaml.safe_dump(summary, file, sort_keys=False)
    print(f"[s10] summary: {yaml.safe_dump(summary, sort_keys=False)}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
