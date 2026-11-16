# SPDX-License-Identifier: BSD-3-Clause
"""The first navigation estimate of an episode: the true start state plus a sampled error.

The start slot knows the true start state, so it also writes the estimate that the navigation
filter starts from (message ``initial_navigation_state``). The filter never reads ``TruthState``.
Scenario parameters under ``nav_error:`` (one standard deviation per axis, all default 0):
``position_m``, ``velocity_mps``, ``attitude_deg``.

The errors come from a generator of their own, seeded from the start slot's generator, so a run
with navigation errors starts from exactly the same true states as the same run without them.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.core.message_types import Frame, MessageKey, TruthState
from lunarsim.core.rotation_math import quaternion_from_axis_angle, quaternion_multiply

NAVIGATION_ERROR_SEED_OFFSET = 0x5EED


class InitialNavigationError:
    """Samples and holds the first navigation estimate of every environment."""

    def __init__(self, parameters: dict | None, count: int, device) -> None:
        """Read the error sizes; allocate the per-environment errors (zero until the first reset)."""
        parameters = dict(parameters or {})
        self.position_sigma = float(parameters.get("position_m", 0.0))
        self.velocity_sigma = float(parameters.get("velocity_mps", 0.0))
        self.attitude_sigma = math.radians(float(parameters.get("attitude_deg", 0.0)))
        self.device = device
        self.position_error = torch.zeros(count, 3, device=device)
        self.velocity_error = torch.zeros(count, 3, device=device)
        self.attitude_error = torch.zeros(count, 3, device=device)
        self.generator: torch.Generator | None = None

    def reset(self, env_ids: Tensor, start_generator: torch.Generator) -> None:
        """Draw new errors for the given environments."""
        if self.generator is None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(int(start_generator.initial_seed()) ^ NAVIGATION_ERROR_SEED_OFFSET)
        count = len(env_ids)

        def normal(sigma: float) -> Tensor:
            """Draw (count, 3) normal errors with the given standard deviation."""
            return sigma * torch.randn(count, 3, generator=self.generator, device=self.device)

        self.position_error[env_ids] = normal(self.position_sigma)
        self.velocity_error[env_ids] = normal(self.velocity_sigma)
        self.attitude_error[env_ids] = normal(self.attitude_sigma)

    def estimate(self, start: TruthState) -> TruthState:
        """Return the first estimate: the start state plus the errors (attitude error in the body frame)."""
        angle = self.attitude_error.norm(dim=1)
        axis = self.attitude_error / angle.clamp_min(1e-12).unsqueeze(1)
        axis = torch.where(angle.unsqueeze(1) > 0.0, axis, torch.tensor([1.0, 0.0, 0.0], device=self.device))
        dtype = start.position.dtype
        error_rotation = quaternion_from_axis_angle(axis, angle).to(start.orientation_quaternion.dtype)
        orientation = quaternion_multiply(start.orientation_quaternion, error_rotation)
        return TruthState(
            frame=start.frame,
            time=start.time.clone(),
            position=start.position + self.position_error.to(dtype),
            velocity=start.velocity + self.velocity_error.to(dtype),
            orientation_quaternion=orientation,
            angular_velocity=start.angular_velocity.clone(),
            mass=start.mass.clone(),
            contact=start.contact.clone(),
        )

    def write(self, bus, start: TruthState, writer) -> None:
        """Write the first estimate to the bus."""
        bus.write(MessageKey.INITIAL_NAVIGATION_STATE, self.estimate(start), writer)
