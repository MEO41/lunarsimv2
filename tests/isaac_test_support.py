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
from lunarsim.tools.episode_runner import build_agent, collect_episodes  # noqa: E402


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


def set_start_states(environment: LanderEnv, altitude, vertical_speed=0.0, horizontal_velocity=(0.0, 0.0), tilt_degrees=0.0):
    """Fix the start state of every environment and reset. Scalars apply to all environments.

    ``altitude`` is the height of the footpad plane above the ground (scalar or (N,) tensor).
    The tilt is about the body x axis. Returns the first observations.
    """
    import math

    from lunarsim.core.binding import SlotName
    from lunarsim.core.rotation_math import quaternion_from_axis_angle

    count, device = environment.num_envs, environment.device
    start = environment.bindings[SlotName.START]

    def expand(value):
        """Turn a scalar or tensor into an (N,) tensor on the device."""
        return torch.as_tensor(value, dtype=torch.float32, device=device).expand(count).clone()

    position = torch.zeros(count, 3, device=device)
    position[:, 2] = expand(altitude) + start.center_of_mass_height
    velocity = torch.zeros(count, 3, device=device)
    velocity[:, 0] = horizontal_velocity[0]
    velocity[:, 1] = horizontal_velocity[1]
    velocity[:, 2] = expand(vertical_speed)
    axis = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(count, 1)
    orientation = quaternion_from_axis_angle(axis, expand(tilt_degrees) * math.pi / 180.0)
    start.set_fixed_start_states(position, velocity, orientation)
    observations, _ = environment.reset()
    return observations


def make_actions(environment: LanderEnv, throttle=0.0, jets=None) -> torch.Tensor:
    """Build the flat action tensor: throttle (scalar or (N,)), zero gimbal, jets (N, n_jets) or all off."""
    actions = torch.zeros(environment.num_envs, environment.cfg.action_space, device=environment.device)
    actions[:, 0] = torch.as_tensor(throttle, dtype=torch.float32, device=environment.device)
    if jets is not None:
        actions[:, 3:] = jets
    return actions


__all__ = [
    "collect_episodes",
    "set_start_states",
    "make_actions","simulation_app", "make_environment", "read_truth", "step_with_agent", "build_agent", "TRUTH_PROBE"]
