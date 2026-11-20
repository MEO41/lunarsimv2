# SPDX-License-Identifier: BSD-3-Clause
"""Train a PPO landing policy with rsl_rl (milestone 1b) and export it.

Example::

    python scripts/train_ppo.py scenario=s1_terminal_truth num_envs=4096 seed=0 run_name=ppo_mlp_seed0

The same command with ``plant=analytic_rigid_body`` trains in the analytic simulator (pure torch,
no Isaac Sim is started; ``--device`` picks the torch device, default ``cuda:0``)::

    python scripts/train_ppo.py scenario=s1_terminal_truth plant=analytic_rigid_body num_envs=4096 seed=0 run_name=ppo_mlp_analytic_seed0

The scenario is used with ``agent=ppo_mlp training=landing_ppo`` unless the command line names
others. Writes into ``runs/<run_name>/``: ``manifest.yaml`` (scenario, seeds, PPO settings, git
state, simulator, start-up time), rsl_rl checkpoints ``model_<iteration>.pt`` and TensorBoard
logs, ``checkpoint_wall_clock.csv`` (wall-clock time and environment steps at every checkpoint,
for the time-to-success comparison of the analytic-to-PhysX study), and ``policy.pt`` (the
exported actor, flown by ``agent=ppo_mlp agent.policy_path=runs/<run_name>/policy.pt``).

Script arguments: ``run_name=`` (required), ``max_iterations=`` (default from the agent file),
``resume_from=<checkpoint>`` (continue an earlier training: networks, optimiser and iteration
count come from the checkpoint, and training runs on until iteration ``max_iterations``; the
scenario overrides must be those of the earlier run, except the plant: a policy trained in the
analytic simulator may continue in PhysX). Every other ``key=value`` is a Hydra override of the
scenario.

Continue a run in a new run directory, so the earlier results stay as they were::

    python scripts/train_ppo.py scenario=s3_faults agent=ppo_gru num_envs=4096 seed=0 \
        run_name=ppo_gru_s3_seed0_to1500 max_iterations=1500 resume_from=runs/ppo_gru_s3_seed0/model_600.pt
"""

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

SCRIPT_START_TIME = time.time()

from lunarsim.env.simulator_choice import command_line_uses_analytic_simulator  # noqa: E402  (no torch, no Isaac)

parser = argparse.ArgumentParser(description="Train a LunarSim PPO policy.")
USES_ANALYTIC_SIMULATOR = command_line_uses_analytic_simulator(sys.argv[1:])
if USES_ANALYTIC_SIMULATOR:
    parser.add_argument("--device", default="cuda:0", help="torch device of the analytic simulator and of PPO")
    arguments, scenario_arguments = parser.parse_known_args()
    simulation_app = None
else:
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    arguments, scenario_arguments = parser.parse_known_args()
    arguments.headless = True
    simulation_app = AppLauncher(arguments).app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from lunarsim.agents.rsl_rl.lander_vector_environment import LanderVectorEnvironment  # noqa: E402
from lunarsim.agents.rsl_rl.training_configuration import export_policy, policy_kind, runner_configuration  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario, split_command_line  # noqa: E402
from lunarsim.env.simulator_choice import build_environment, check_simulator_choice  # noqa: E402
from lunarsim.tools.run_manifest import write_manifest  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_WALL_CLOCK_FILE_NAME = "checkpoint_wall_clock.csv"


def record_checkpoint_times(runner, csv_path: Path, control_steps_per_iteration: int, learn_start: list[float]) -> None:
    """Make every ``runner.save`` also append one row to ``csv_path``: the checkpoint, its iteration and the times.

    Columns: ``checkpoint`` (file name), ``iteration`` (rsl_rl's iteration index stored in the
    checkpoint), ``iterations_in_this_run``, ``learn_wall_clock_s`` (since ``runner.learn``
    started), ``script_wall_clock_s`` (since the script started, simulator start-up included),
    ``environment_control_steps`` (control steps summed over all environments in this run).
    ``learn_start`` is a one-element list that holds the start time of ``runner.learn``.
    """
    first_iteration = runner.current_learning_iteration
    original_save = runner.save
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            ["checkpoint", "iteration", "iterations_in_this_run", "learn_wall_clock_s", "script_wall_clock_s", "environment_control_steps"]
        )

    def save_and_record(path: str, infos: dict | None = None) -> None:
        """Save the checkpoint as rsl_rl does, then log when it was saved and how much was simulated."""
        original_save(path, infos)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.time()
        iteration = runner.current_learning_iteration
        iterations_done = iteration - first_iteration + 1
        with open(csv_path, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [
                    Path(path).name,
                    iteration,
                    iterations_done,
                    round(now - learn_start[0], 3),
                    round(now - SCRIPT_START_TIME, 3),
                    iterations_done * control_steps_per_iteration,
                ]
            )

    runner.save = save_and_record


def main() -> None:
    """Train, save checkpoints, export the actor and write the manifest."""
    options = {"run_name": None, "max_iterations": None, "resume_from": None}
    remaining = []
    for argument in scenario_arguments:
        key, _, value = argument.partition("=")
        if key in options:
            options[key] = value
        else:
            remaining.append(argument)
    if not options["run_name"]:
        raise SystemExit("give run_name=<name>")
    scenario_name, overrides = split_command_line(remaining)
    keys = {override.split("=", 1)[0] for override in overrides}
    if "agent" not in keys:
        overrides.append("agent=ppo_mlp")
    if "training" not in keys:
        overrides.append("training=landing_ppo")
    scenario = load_scenario(scenario_name, overrides)
    check_simulator_choice(scenario, USES_ANALYTIC_SIMULATOR)
    # The PhysX path keeps its fixed device; the analytic path takes --device.
    device = arguments.device if USES_ANALYTIC_SIMULATOR else "cuda:0"
    agent_parameters = scenario.slots["agent"].parameters
    ppo = copy.deepcopy(agent_parameters["ppo"])
    if options["max_iterations"]:
        ppo["max_iterations"] = int(options["max_iterations"])
    run_directory = REPOSITORY_ROOT / "runs" / options["run_name"]
    run_directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(scenario.root_seed)
    environment = build_environment(scenario, device=device)
    vector_environment = LanderVectorEnvironment(environment, agent_parameters)
    configuration = runner_configuration(
        ppo, scenario.root_seed, options["run_name"], environment.critic_observation == "truth"
    )
    write_manifest(
        run_directory,
        scenario,
        environment.random_generators.seeds(),
        extra={
            "kind": "ppo_training",
            "runner": copy.deepcopy(configuration),
            "resumed_from": options["resume_from"],
            "simulator": "analytic" if USES_ANALYTIC_SIMULATOR else "isaac_physx",
            "device": device,
            "startup_wall_clock_s": round(time.time() - SCRIPT_START_TIME, 1),
        },
    )
    runner = OnPolicyRunner(vector_environment, configuration, log_dir=str(run_directory), device=device)
    if options["resume_from"]:
        runner.load(options["resume_from"], map_location=device)
        # A checkpoint stores the index of its last finished iteration (model_600.pt holds 599),
        # so the continued run starts at the next one.
        runner.current_learning_iteration += 1
        print(f"[lunarsim] resumed from {options['resume_from']} at iteration {runner.current_learning_iteration}")
    remaining_iterations = int(ppo["max_iterations"]) - runner.current_learning_iteration
    if remaining_iterations <= 0:
        raise SystemExit(f"max_iterations={ppo['max_iterations']} is not above the checkpoint's iteration {runner.current_learning_iteration}")
    control_steps_per_iteration = int(ppo["num_steps_per_env"]) * vector_environment.policy_period_steps * environment.num_envs
    learn_start = [time.time()]
    record_checkpoint_times(runner, run_directory / CHECKPOINT_WALL_CLOCK_FILE_NAME, control_steps_per_iteration, learn_start)
    learn_start[0] = time.time()
    runner.learn(num_learning_iterations=remaining_iterations, init_at_random_ep_len=False)
    wall_time = time.time() - learn_start[0]
    runner.save(str(run_directory / f"model_{ppo['max_iterations']}.pt"))
    export_policy(runner.alg.actor, run_directory / "policy.pt", policy_kind(ppo))
    print(f"[lunarsim] trained to iteration {ppo['max_iterations']} ({remaining_iterations} iterations) in {wall_time:.0f} s; policy: {run_directory / 'policy.pt'}")
    print(f"[lunarsim] truth reads: {environment.bus.truth_read_log}")
    environment.close()


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
