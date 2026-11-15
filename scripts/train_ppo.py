# SPDX-License-Identifier: BSD-3-Clause
"""Train a PPO landing policy with rsl_rl (milestone 1b) and export it.

Example::

    python scripts/train_ppo.py scenario=s1_terminal_truth num_envs=4096 seed=0 run_name=ppo_mlp_seed0

The scenario is used with ``agent=ppo_mlp training=landing_ppo`` unless the command line names
others. Writes into ``runs/<run_name>/``: ``manifest.yaml`` (scenario, seeds, PPO settings, git
state), rsl_rl checkpoints ``model_<iteration>.pt`` and TensorBoard logs, and ``policy.pt`` (the
exported actor, flown by ``agent=ppo_mlp agent.policy_path=runs/<run_name>/policy.pt``).

Script arguments: ``run_name=`` (required), ``max_iterations=`` (default from the agent file).
Every other ``key=value`` is a Hydra override of the scenario.
"""

import argparse
import copy
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a LunarSim PPO policy.")
AppLauncher.add_app_launcher_args(parser)
arguments, scenario_arguments = parser.parse_known_args()
arguments.headless = True
simulation_app = AppLauncher(arguments).app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from lunarsim.agents.rsl_rl.lander_vector_environment import LanderVectorEnvironment  # noqa: E402
from lunarsim.agents.rsl_rl.training_configuration import export_policy, runner_configuration  # noqa: E402
from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario, split_command_line  # noqa: E402
from lunarsim.tools.run_manifest import write_manifest  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Train, save checkpoints, export the actor and write the manifest."""
    options = {"run_name": None, "max_iterations": None}
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
    agent_parameters = scenario.slots["agent"].parameters
    ppo = copy.deepcopy(agent_parameters["ppo"])
    if options["max_iterations"]:
        ppo["max_iterations"] = int(options["max_iterations"])
    run_directory = REPOSITORY_ROOT / "runs" / options["run_name"]
    run_directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(scenario.root_seed)
    environment = LanderEnv(build_lander_env_cfg(scenario, device="cuda:0"))
    vector_environment = LanderVectorEnvironment(environment, agent_parameters)
    configuration = runner_configuration(
        ppo, scenario.root_seed, options["run_name"], environment.critic_observation == "truth"
    )
    write_manifest(
        run_directory,
        scenario,
        environment.random_generators.seeds(),
        extra={"kind": "ppo_training", "runner": copy.deepcopy(configuration)},
    )
    runner = OnPolicyRunner(vector_environment, configuration, log_dir=str(run_directory), device="cuda:0")
    start = time.time()
    runner.learn(num_learning_iterations=int(ppo["max_iterations"]), init_at_random_ep_len=False)
    wall_time = time.time() - start
    runner.save(str(run_directory / f"model_{ppo['max_iterations']}.pt"))
    export_policy(runner.alg.actor, run_directory / "policy.pt")
    print(f"[lunarsim] trained {ppo['max_iterations']} iterations in {wall_time:.0f} s; policy: {run_directory / 'policy.pt'}")
    print(f"[lunarsim] truth reads: {environment.bus.truth_read_log}")
    environment.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
