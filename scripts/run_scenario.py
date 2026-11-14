# SPDX-License-Identifier: BSD-3-Clause
"""Run a scenario with the agent named in its ``agent`` slot.

Example::

    python scripts/run_scenario.py scenario=m0_null_loop num_envs=1024 steps=100

``scenario=`` picks the file in ``configs/scenarios``; ``steps=`` is the number of control
steps; every other ``key=value`` is a Hydra override of the scenario. Add ``--viz kit`` to
open the viewer.
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a LunarSim scenario.")
AppLauncher.add_app_launcher_args(parser)
arguments, scenario_arguments = parser.parse_known_args()
simulation_app = AppLauncher(arguments).app

import torch  # noqa: E402

from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario, split_command_line  # noqa: E402
from lunarsim.tools.episode_runner import build_agent, run_agent_in_environment  # noqa: E402


def main() -> None:
    """Load the scenario, build the environment and the agent, and step the loop."""
    number_of_steps = 100
    remaining_arguments = []
    for argument in scenario_arguments:
        if argument.startswith("steps="):
            number_of_steps = int(argument.split("=", 1)[1])
        else:
            remaining_arguments.append(argument)
    scenario_name, overrides = split_command_line(remaining_arguments)
    scenario = load_scenario(scenario_name, overrides)
    environment = LanderEnv(build_lander_env_cfg(scenario, device=arguments.device))
    agent = build_agent(environment)

    start_time = time.perf_counter()
    run_agent_in_environment(environment, agent, number_of_steps)
    elapsed = time.perf_counter() - start_time
    print(
        f"[lunarsim] scenario={scenario.name} envs={environment.num_envs} control_steps={number_of_steps} "
        f"env_steps_per_second={number_of_steps * environment.num_envs / elapsed:.0f} "
        f"gpu_memory_MiB={torch.cuda.max_memory_allocated() / 2**20:.0f}",
        flush=True,
    )
    environment.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
