# SPDX-License-Identifier: BSD-3-Clause
"""Support for CPU tests of the analytic simulator (``plant: analytic_rigid_body``).

The functions have the same names and arguments as those of ``tests/isaac_test_support.py``, so
a known-answer test of the PhysX lander reads the same for the analytic lander. No Isaac Sim is
started: the environment is ``AnalyticLanderEnvironment`` on the CPU.
"""

from __future__ import annotations

import math

import torch

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey
from lunarsim.core.rotation_math import quaternion_from_axis_angle
from lunarsim.env.analytic_lander_environment import AnalyticLanderEnvironment
from lunarsim.env.scenario_loader import load_scenario


class _TruthProbe:
    """A test-only reader that is allowed to look at TruthState on the bus."""

    requires = frozenset({MessageKey.TRUTH_STATE})
    provides = frozenset()
    reads_truth = True


TRUTH_PROBE = _TruthProbe()


def make_environment(
    scenario_name: str, overrides: list[str] | None = None, plant: str = "analytic_rigid_body"
) -> AnalyticLanderEnvironment:
    """Load a scenario with an analytic plant (default ``analytic_rigid_body``) and build its environment on the CPU."""
    scenario = load_scenario(scenario_name, [f"plant={plant}", *(overrides or [])])
    return AnalyticLanderEnvironment(scenario, device="cpu")


def read_truth(environment: AnalyticLanderEnvironment):
    """Return the latest TruthState on the environment's bus."""
    return environment.bus.read(MessageKey.TRUTH_STATE, TRUTH_PROBE)


def set_start_states(environment, altitude, vertical_speed=0.0, horizontal_velocity=(0.0, 0.0), tilt_degrees=0.0):
    """Fix the start state of every environment and reset. Scalars apply to all environments.

    ``altitude`` is the height of the footpad plane above the ground (scalar or (N,) tensor).
    The tilt is about the body x axis. Returns the first observations.
    """
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


def make_actions(environment, throttle=0.0, jets=None) -> torch.Tensor:
    """Build the flat action tensor: throttle (scalar or (N,)), zero gimbal, jets (N, n_jets) or all off."""
    actions = torch.zeros(environment.num_envs, environment.cfg.action_space, device=environment.device)
    actions[:, 0] = torch.as_tensor(throttle, dtype=torch.float32, device=environment.device)
    if jets is not None:
        actions[:, 3:] = jets
    return actions
