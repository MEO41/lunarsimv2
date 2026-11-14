# SPDX-License-Identifier: BSD-3-Clause
"""Classical agents for controller build-up steps 2 and 3: hover hold and vertical descent."""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.agents.agent_base import AgentBase
from lunarsim.agents.classical.attitude_controller import AttitudeController
from lunarsim.agents.classical.vertical_controller import VerticalController
from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import Command, NavigationStateLayout, Observation
from lunarsim.core.registry import register_binding

CONTROLLER_PARAMETER_GROUPS = ("attitude", "vertical")


class ClassicalLandingAgentBase(AgentBase):
    """Shared parts of the classical agents: controllers, observation parsing, engine cut-off latch.

    Scenario parameters ``attitude:`` and ``vertical:`` are passed to the two controllers.
    """

    def __init__(self, parameters, number_of_environments, number_of_jets, device, shared_data=None) -> None:
        """Build the attitude and vertical controllers from the lander preset."""
        super().__init__(parameters, number_of_environments, number_of_jets, device, shared_data)
        self.preset = self.shared_data["lander_preset"]
        self.attitude_controller = AttitudeController(self.preset, device=device, **self.parameters.get("attitude", {}))
        self.vertical_controller = VerticalController(self.preset, **self.parameters.get("vertical", {}))
        self.upright = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(number_of_environments, 1)
        # Once the engine is cut near the ground it stays off until the next episode.
        self.engine_is_cut = torch.zeros(number_of_environments, dtype=torch.bool, device=device)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Re-arm the engine for a new episode."""
        self.engine_is_cut[env_ids] = False

    @staticmethod
    def split_state(observation: Observation) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return position, velocity, orientation, body rates and mass from the policy observation."""
        layout = NavigationStateLayout
        state = observation.policy
        return (
            state[:, layout.POSITION],
            state[:, layout.VELOCITY],
            state[:, layout.ORIENTATION_QUATERNION],
            state[:, layout.ANGULAR_VELOCITY],
            state[:, layout.MASS].squeeze(1),
        )

    def assemble_command(
        self,
        throttle: Tensor,
        orientation: Tensor,
        angular_velocity: Tensor,
        wanted_orientation: Tensor,
        altitude: Tensor,
        control_step_duration: float,
    ) -> Command:
        """Combine throttle and attitude loop into a Command, and apply the engine cut-off.

        Below the cut-off height the engine is shut down and the jets are closed: the lander
        drops the last few centimetres, and nothing fights the ground afterwards.
        """
        self.engine_is_cut |= altitude < self.vertical_controller.engine_cutoff_altitude
        jets = self.attitude_controller.compute_jet_duties(
            orientation, angular_velocity, wanted_orientation, control_step_duration
        )
        throttle = torch.where(self.engine_is_cut, torch.zeros_like(throttle), throttle)
        jets = torch.where(self.engine_is_cut.unsqueeze(1), torch.zeros_like(jets), jets)
        return Command(
            throttle=throttle,
            gimbal=torch.zeros(throttle.shape[0], 2, device=throttle.device),
            reaction_control_jets=jets,
        )


@register_binding(SlotName.AGENT, "hover_hold")
class HoverHoldAgent(ClassicalLandingAgentBase):
    """Step 2: holds ``target_altitude_m`` (footpad height) and the upright attitude."""

    def compute_command(self, observation: Observation, control_step_duration: float) -> Command:
        """Return the command of the altitude hold."""
        position, velocity, orientation, angular_velocity, mass = self.split_state(observation)
        controller = self.vertical_controller
        altitude = controller.footpad_altitude(position)
        target = torch.full_like(altitude, float(self.parameters.get("target_altitude_m", 30.0)))
        acceleration = controller.acceleration_to_hold_altitude(altitude, velocity[:, 2], target)
        throttle = controller.throttle_for_vertical_acceleration(acceleration, mass, orientation)
        return self.assemble_command(throttle, orientation, angular_velocity, self.upright, altitude, control_step_duration)


@register_binding(SlotName.AGENT, "vertical_descent")
class VerticalDescentAgent(ClassicalLandingAgentBase):
    """Step 3: descends straight down on the descent-rate profile and cuts the engine near the ground."""

    def compute_command(self, observation: Observation, control_step_duration: float) -> Command:
        """Return the command of the vertical descent."""
        position, velocity, orientation, angular_velocity, mass = self.split_state(observation)
        controller = self.vertical_controller
        altitude = controller.footpad_altitude(position)
        acceleration = controller.acceleration_to_follow_descent_profile(altitude, velocity[:, 2])
        throttle = controller.throttle_for_vertical_acceleration(acceleration, mass, orientation)
        return self.assemble_command(throttle, orientation, angular_velocity, self.upright, altitude, control_step_duration)
