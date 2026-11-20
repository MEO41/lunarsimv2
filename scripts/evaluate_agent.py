# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate the agent of a scenario and write a run directory.

Example::

    python scripts/evaluate_agent.py scenario=s1_terminal_truth num_envs=2048 run_name=zem_zev_full_box

Writes ``runs/<run_name>/manifest.yaml``, ``episodes.parquet`` (one row per episode) and
``trajectories.parquet`` (a few environments, every control step). Figures are made from
these files only: ``lunarsim figures runs/<run_name>``.

Script arguments: ``run_name=``, ``episodes_per_env=`` (default 1), ``logged_envs=`` (default 16),
``fault_grid=true`` (S3: the held-out fault test grid of ``slots/faults/fault_grid.py``).
An agent with decision records (``site_pick_divert``, S6) gets them merged into the episode table.
Every other ``key=value`` is a Hydra override of the scenario.

With ``plant=analytic_rigid_body`` the agent flies in the analytic simulator (pure torch, no
Isaac Sim is started; ``--device`` picks the torch device, default ``cuda:0``). The same policy
and the same ``seed`` give the same start states in both simulators (cross-test of the
analytic-to-PhysX study).
"""

import argparse
import sys
import time
from pathlib import Path

from lunarsim.env.simulator_choice import command_line_uses_analytic_simulator  # noqa: E402  (no torch, no Isaac)

parser = argparse.ArgumentParser(description="Evaluate a LunarSim agent.")
USES_ANALYTIC_SIMULATOR = command_line_uses_analytic_simulator(sys.argv[1:])
if USES_ANALYTIC_SIMULATOR:
    parser.add_argument("--device", default="cuda:0", help="torch device of the analytic simulator")
    arguments, scenario_arguments = parser.parse_known_args()
    simulation_app = None
else:
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    arguments, scenario_arguments = parser.parse_known_args()
    simulation_app = AppLauncher(arguments).app

from lunarsim.env.scenario_loader import load_scenario, split_command_line  # noqa: E402
from lunarsim.env.simulator_choice import build_environment, check_simulator_choice  # noqa: E402
from lunarsim.tools.episode_records import EPISODES_FILE_NAME  # noqa: E402
from lunarsim.tools.episode_runner import build_agent, collect_episodes  # noqa: E402
from lunarsim.tools.run_manifest import write_manifest  # noqa: E402
from lunarsim.tools.trajectory_log import TrajectoryLog  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def merge_agent_decision_records(episodes, agent):
    """Add the agent's per-episode decision records (``drain_decision_records``, S6) to the episode table.

    Rows are matched on (environment_index, episode_number). Agents without records leave the table as it is.
    """
    if not hasattr(agent, "drain_decision_records") or episodes.empty:
        return episodes
    import pandas

    records = agent.drain_decision_records()
    if not records:
        return episodes
    decisions = pandas.DataFrame({name: values.numpy() for name, values in records.items()})
    merged = episodes.merge(decisions, on=["environment_index", "episode_number"], how="left")
    if "decision_mass_estimate_kg" in merged and "start_mass_kg" in merged and "propellant_used_kg" in merged:
        final_mass = merged["start_mass_kg"] - merged["propellant_used_kg"]
        merged["divert_propellant_estimate_kg"] = merged["decision_mass_estimate_kg"] - final_mass
    return merged


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
    check_simulator_choice(scenario, USES_ANALYTIC_SIMULATOR)
    run_name = script_options["run_name"] or f"{scenario.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_directory = REPOSITORY_ROOT / "runs" / run_name

    environment = build_environment(scenario, device=arguments.device)
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
    episodes = merge_agent_decision_records(episodes, agent)

    write_manifest(
        run_directory,
        scenario,
        environment.random_generators.seeds(),
        extra={
            "agent": scenario.slots["agent"].binding,
            "simulator": "analytic" if USES_ANALYTIC_SIMULATOR else "isaac_physx",
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
    if simulation_app is not None:
        simulation_app.close()
