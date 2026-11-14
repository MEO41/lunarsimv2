# SPDX-License-Identifier: BSD-3-Clause
"""Shared helpers for the CPU unit tests."""

from __future__ import annotations

import torch

from lunarsim.core.binding import BindingContext
from lunarsim.core.message_bus import MessageBus
from lunarsim.core.message_types import Frame, TruthState
from lunarsim.core.simulation_clock import SimulationClock


def make_truth_state(number_of_environments: int = 4, number_of_legs: int = 4) -> TruthState:
    """Return a valid site-frame TruthState with simple values."""
    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(number_of_environments, 1)
    return TruthState(
        frame=Frame.SITE,
        time=torch.zeros(number_of_environments),
        position=torch.ones(number_of_environments, 3),
        velocity=torch.zeros(number_of_environments, 3),
        orientation_quaternion=identity_quaternion,
        angular_velocity=torch.zeros(number_of_environments, 3),
        mass=torch.full((number_of_environments,), 100.0),
        contact=torch.zeros(number_of_environments, number_of_legs),
    )


def make_binding_context(number_of_environments: int = 4) -> BindingContext:
    """Return a CPU binding context with a 5 ms physics step and 4 substeps per control step."""
    clock = SimulationClock(number_of_environments, physics_step_duration=0.005, physics_steps_per_control_step=4)
    return BindingContext(number_of_environments=number_of_environments, device="cpu", clock=clock, bus=MessageBus())
