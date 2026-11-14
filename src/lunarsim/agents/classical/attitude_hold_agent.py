# SPDX-License-Identifier: BSD-3-Clause
"""``agent: attitude_hold`` - controller build-up step 1: the jets keep the lander upright.

The engine gives open-loop hover thrust (weight from the estimated mass); only the attitude
loop is closed.
"""

import torch

from lunarsim.agents.agent_base import AgentBase
from lunarsim.agents.classical.attitude_controller import AttitudeController
from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import Command, NavigationStateLayout, Observation
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.core.registry import register_binding
from lunarsim.core.rotation_math import body_up_axis_in_parent


@register_binding(SlotName.AGENT, "attitude_hold")
class AttitudeHoldAgent(AgentBase):
    """Holds the upright attitude with the jets; thrust balances the weight."""

    def __init__(self, parameters, number_of_environments, number_of_jets, device, shared_data=None) -> None:
        """Build the attitude controller from the lander preset."""
        super().__init__(parameters, number_of_environments, number_of_jets, device, shared_data)
        self.preset = self.shared_data["lander_preset"]
        self.attitude_controller = AttitudeController(self.preset, device=device)
        self.upright = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(number_of_environments, 1)

    def compute_command(self, observation: Observation, control_step_duration: float) -> Command:
        """Return hover throttle (corrected for the tilt) and the jet duties that right the lander."""
        layout = NavigationStateLayout
        state = observation.policy
        orientation = state[:, layout.ORIENTATION_QUATERNION]
        mass = state[:, layout.MASS].squeeze(1)
        vertical_share = body_up_axis_in_parent(orientation)[:, 2].clamp_min(0.5)
        throttle = mass * MOON_SURFACE_GRAVITY / (self.preset.engine_rated_thrust * vertical_share)
        jets = self.attitude_controller.compute_jet_duties(
            orientation, state[:, layout.ANGULAR_VELOCITY], self.upright, control_step_duration
        )
        return Command(
            throttle=throttle, gimbal=torch.zeros(state.shape[0], 2, device=state.device), reaction_control_jets=jets
        )
