# SPDX-License-Identifier: BSD-3-Clause
"""Support for tests that need Isaac Lab and a GPU.

Each test file runs in its own process (``scripts/run_isaac_tests.py``), because one process
can hold only one simulator and one environment. Importing this module starts the simulator.
"""

from __future__ import annotations

from isaaclab.app import AppLauncher

simulation_app = AppLauncher().app

import torch  # noqa: E402

from lunarsim.core.message_types import MessageKey, Observation  # noqa: E402
from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario  # noqa: E402
from lunarsim.tools.episode_runner import build_agent  # noqa: E402


class _TruthProbe:
    """A test-only reader that is allowed to look at TruthState on the bus."""

    requires = frozenset({MessageKey.TRUTH_STATE})
    provides = frozenset()
    reads_truth = True


TRUTH_PROBE = _TruthProbe()


def make_environment(scenario_name: str, overrides: list[str] | None = None) -> LanderEnv:
    """Load a scenario and build its environment on the GPU."""
    scenario = load_scenario(scenario_name, overrides or [])
    return LanderEnv(build_lander_env_cfg(scenario, device="cuda:0"))


def read_truth(environment: LanderEnv):
    """Return the latest TruthState on the environment's bus."""
    return environment.bus.read(MessageKey.TRUTH_STATE, TRUTH_PROBE)


def step_with_agent(environment: LanderEnv, agent, observations: dict, number_of_steps: int) -> dict:
    """Step the closed loop for some control steps and return the last observations."""
    for _ in range(number_of_steps):
        with torch.inference_mode():
            command = agent.compute_command(
                Observation(policy=observations["policy"]), environment.clock.control_step_duration
            )
            observations, _, _, _, _ = environment.step(command.to_action_tensor())
    return observations


__all__ = ["simulation_app", "make_environment", "read_truth", "step_with_agent", "build_agent", "TRUTH_PROBE"]
