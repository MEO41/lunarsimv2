# SPDX-License-Identifier: BSD-3-Clause
"""Measure env steps per second and GPU memory of a scenario (goal G4).

Example::

    python scripts/benchmark_environment.py scenario=s1_terminal_truth num_envs=1024 steps=500

Prints one line that starts with ``[benchmark]`` and holds a JSON object. One process measures
one environment count; run it once per count. "Env steps" are control steps times environments,
and the time includes the agent named in the scenario.
"""

import argparse
import json
import os
import subprocess
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark a LunarSim scenario.")
AppLauncher.add_app_launcher_args(parser)
arguments, scenario_arguments = parser.parse_known_args()
simulation_app = AppLauncher(arguments).app

import torch  # noqa: E402

from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario, split_command_line  # noqa: E402
from lunarsim.tools.episode_runner import build_agent, run_agent_in_environment  # noqa: E402


def gpu_memory_of_this_process_mebibytes() -> float:
    """Return the GPU memory that nvidia-smi reports for this process (PhysX plus torch), in MiB."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        for line in output.splitlines():
            process_id, used = (part.strip() for part in line.split(","))
            if int(process_id) == os.getpid():
                return float(used)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return float("nan")


def main() -> None:
    """Warm up, then time a fixed number of control steps."""
    number_of_steps, warm_up_steps = 500, 50
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
    run_agent_in_environment(environment, agent, warm_up_steps)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    run_agent_in_environment(environment, agent, number_of_steps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    result = {
        "scenario": scenario.name,
        "agent": scenario.slots["agent"].binding,
        "num_envs": environment.num_envs,
        "control_steps": number_of_steps,
        "env_steps_per_second": round(number_of_steps * environment.num_envs / elapsed),
        "physics_steps_per_second": round(number_of_steps * environment.num_envs * environment.cfg.decimation / elapsed),
        "real_time_factor_per_env": round(number_of_steps * environment.clock.control_step_duration / elapsed, 2),
        "gpu_memory_process_MiB": gpu_memory_of_this_process_mebibytes(),
        "gpu_memory_torch_MiB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "gpu": torch.cuda.get_device_name(0),
    }
    print("[benchmark] " + json.dumps(result), flush=True)
    environment.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
