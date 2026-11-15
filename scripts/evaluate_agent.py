# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate the agent of a scenario and write a run directory.

Example::

    python scripts/evaluate_agent.py scenario=s1_terminal_truth num_envs=2048 run_name=zem_zev_full_box

Writes ``runs/<run_name>/manifest.yaml``, ``episodes.parquet`` (one row per episode) and
``trajectories.parquet`` (a few environments, every control step). Figures are made from
these files only: ``lunarsim figures runs/<run_name>``.

Script arguments: ``run_name=``, ``episodes_per_env=`` (default 1), ``logged_envs=`` (default 16),
``fault_grid=true`` (S3: the held-out fault test grid of ``slots/faults/fault_grid.py``).
Every other ``key=value`` is a Hydra override of the scenario.
"""

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a LunarSim agent.")
AppLauncher.add_app_launcher_args(parser)
arguments, scenario_arguments = parser.parse_known_args()
simulation_app = AppLauncher(arguments).app

from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario, split_command_line  # noqa: E402
from lunarsim.tools.episode_records import EPISODES_FILE_NAME  # noqa: E402
from lunarsim.tools.episode_runner import build_agent, collect_episodes  # noqa: E402
from lunarsim.tools.run_manifest import write_manifest  # noqa: E402
from lunarsim.tools.trajectory_log import TrajectoryLog  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run the evaluation and write the run files."""
    script_options = {"run_name": None, "episodes_per_env": "1", "logged_envs": "16", "fault_grid": "false"}
    remaining_arguments = []
    for argument in scenario_arguments:
        key = argument.split("=", 1)[0]
        if key in script_options:
            script_options[key] = argument.split("=", 1)[1]
        else:
            remaining_arguments.append(argument)
    scenario_name, overrides = split_command_line(remaining_arguments)
    scenario = load_scenario(scenario_name, overrides)
    run_name = script_options["run_name"] or f"{scenario.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_directory = REPOSITORY_ROOT / "runs" / run_name

    environment = LanderEnv(build_lander_env_cfg(scenario, device=arguments.device))
    agent = build_agent(environment)
    if script_options["fault_grid"].lower() in ("true", "1", "yes"):
        from lunarsim.slots.faults.fault_grid import apply_fault_grid

        apply_fault_grid(environment.bindings["faults"], environment.num_envs, environment.device)
    trajectory_log = TrajectoryLog(environment, int(script_options["logged_envs"]))
    maximum_steps = int(1.2 * environment.max_episode_length * int(script_options["episodes_per_env"])) + 10

    start_time = time.perf_counter()
    episodes = collect_episodes(
        environment,
        agent,
        episodes_per_environment=int(script_options["episodes_per_env"]),
        maximum_steps=maximum_steps,
        after_step=trajectory_log.record,
    )
    elapsed = time.perf_counter() - start_time

    write_manifest(
        run_directory,
        scenario,
        environment.random_generators.seeds(),
        extra={
            "agent": scenario.slots["agent"].binding,
            "episodes": int(len(episodes)),
            "wall_time_s": round(elapsed, 1),
            "truth_readers": environment.bus.truth_read_log,
        },
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(run_directory / EPISODES_FILE_NAME, index=False)
    trajectory_log.write(run_directory)

    print(f"[lunarsim] run directory: {run_directory}")
    print(f"[lunarsim] episodes: {len(episodes)}  success rate: {episodes['success'].mean():.4f}  wall time: {elapsed:.0f} s")
    print(episodes["outcome"].value_counts().to_string())
    environment.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
