# SPDX-License-Identifier: BSD-3-Clause
"""The one clock of the simulation."""

from __future__ import annotations

import torch
from torch import Tensor


class SimulationClock:
    """Episode time of every environment.

    All slots read time from here. ``Measurement.stamp`` values are capture times on this clock.
    The clock advances once per physics substep, so bindings that run every substep (actuators)
    see the correct time inside a control step.
    """

    def __init__(
        self,
        number_of_environments: int,
        physics_step_duration: float,
        physics_steps_per_control_step: int,
        device: torch.device | str = "cpu",
    ) -> None:
        """Create a clock with all environments at time zero."""
        if physics_step_duration <= 0.0 or physics_steps_per_control_step < 1:
            raise ValueError("physics_step_duration must be > 0 and physics_steps_per_control_step >= 1")
        self.physics_step_duration = float(physics_step_duration)
        self.physics_steps_per_control_step = int(physics_steps_per_control_step)
        # Time is stored as an integer count of physics steps so it never drifts.
        self._physics_step_count = torch.zeros(number_of_environments, dtype=torch.long, device=device)

    @property
    def control_step_duration(self) -> float:
        """Return the length of one control step in seconds."""
        return self.physics_step_duration * self.physics_steps_per_control_step

    @property
    def time(self) -> Tensor:
        """Return the episode time of every environment, shape (N,), seconds."""
        return self._physics_step_count.to(torch.float32) * self.physics_step_duration

    @property
    def physics_step_count(self) -> Tensor:
        """Return the number of physics steps since the episode start, shape (N,)."""
        return self._physics_step_count

    def advance_one_physics_step(self) -> None:
        """Move every environment forward by one physics step."""
        self._physics_step_count += 1

    def reset(self, env_ids: Tensor) -> None:
        """Set the episode time of the given environments back to zero."""
        self._physics_step_count[env_ids] = 0
