# SPDX-License-Identifier: BSD-3-Clause
"""``start: fixed`` - every episode starts from the same configured state."""

import torch

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import Frame, MessageKey, TruthState
from lunarsim.core.registry import register_binding
from lunarsim.slots.start.initial_navigation_error import InitialNavigationError


@register_binding(SlotName.START, "fixed")
class FixedStart(BindingBase):
    """Writes one fixed start state for all environments.

    Parameters: ``position_m`` (3 values, site frame), ``velocity_mps`` (3 values),
    ``mass_kg`` (optional; the plant uses the asset's own mass when it is missing), ``nav_error``
    (first navigation estimate error, see ``initial_navigation_error.py``).
    """

    provides = frozenset({MessageKey.START_STATE, MessageKey.INITIAL_NAVIGATION_STATE})

    def __init__(self, parameters, context) -> None:
        """Prepare the first-estimate errors (parameters ``nav_error``)."""
        super().__init__(parameters, context)
        self.navigation_error = InitialNavigationError(self.parameters.get("nav_error"), self.number_of_environments, self.device)

    def reset(self, env_ids, gen) -> None:
        """Draw new first-estimate errors for the given environments."""
        self.navigation_error.reset(env_ids, gen)

    def step(self, bus, dt: float) -> None:
        """Write the start state. The plant reads it for the environments that reset."""
        count = self.number_of_environments
        position = torch.tensor(self.parameters.get("position_m", [0.0, 0.0, 1.0]), device=self.device)
        velocity = torch.tensor(self.parameters.get("velocity_mps", [0.0, 0.0, 0.0]), device=self.device)
        identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        mass = float(self.parameters.get("mass_kg", float("nan")))
        state = TruthState(
            frame=Frame.SITE,
            time=torch.zeros(count, device=self.device),
            position=position.repeat(count, 1),
            velocity=velocity.repeat(count, 1),
            orientation_quaternion=identity_quaternion.repeat(count, 1),
            angular_velocity=torch.zeros(count, 3, device=self.device),
            mass=torch.full((count,), mass, device=self.device),
            contact=torch.zeros(count, 0, device=self.device),
        )
        bus.write(MessageKey.START_STATE, state, self)
        self.navigation_error.write(bus, state, self)
