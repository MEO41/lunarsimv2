# SPDX-License-Identifier: BSD-3-Clause
"""Classical agents for controller build-up steps 2 and 3: hover hold and vertical descent."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.agents.agent_base import AgentBase
from lunarsim.agents.classical.attitude_controller import AttitudeController, quaternion_for_thrust_direction
from lunarsim.agents.classical.vertical_controller import VerticalController
from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import Command, NavigationStateLayout, Observation
from lunarsim.core.registry import register_binding
from lunarsim.core.rotation_math import tilt_angle_from_vertical

CONTROLLER_PARAMETER_GROUPS = ("attitude", "vertical")


ENGINE_CUTOFF_SIGNALS = ("surface_probe_contact", "height_above_ground")


class ClassicalLandingAgentBase(AgentBase):
    """Shared parts of the classical agents: controllers, observation parsing, engine cut-off latch.

    Scenario parameters ``attitude:`` and ``vertical:`` are passed to the two controllers.

    Engine cut-off (``engine_cutoff_signal``):

    - ``surface_probe_contact`` (default, the Apollo "probe mode"): the engine is shut down
      ``engine_stop_delay_after_probe_contact_s`` seconds after the contact light first comes on
      (observation value ``SURFACE_PROBE_CONTACT``), but only once the lander is steady: body rate
      below ``steady_rate_for_engine_stop_deg_per_s`` and tilt below
      ``steady_tilt_for_engine_stop_deg``. A lander that is still turning keeps descending
      ("pad mode", the Apollo backup) and the engine is shut down at once when the height above
      the ground straight below falls under ``backup_engine_cutoff_height_m``.
    - ``height_above_ground``: the engine is shut down when the height above the ground straight
      below falls under ``vertical.engine_cutoff_altitude_m`` (the rule used up to 2026-11-15).
    """

    def __init__(self, parameters, number_of_environments, number_of_jets, device, shared_data=None) -> None:
        """Build the attitude and vertical controllers from the lander preset."""
        super().__init__(parameters, number_of_environments, number_of_jets, device, shared_data)
        self.preset = self.shared_data["lander_preset"]
        self.attitude_controller = AttitudeController(self.preset, device=device, **self.parameters.get("attitude", {}))
        self.vertical_controller = VerticalController(self.preset, **self.parameters.get("vertical", {}))
        # Upright = thrust axis along site +z, at the heading the lander has.
        self.vertical_direction = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(number_of_environments, 1)
        # Once the engine is cut near the ground it stays off until the next episode.
        self.engine_is_cut = torch.zeros(number_of_environments, dtype=torch.bool, device=device)
        self.engine_cutoff_signal = str(self.parameters.get("engine_cutoff_signal", "surface_probe_contact"))
        if self.engine_cutoff_signal not in ENGINE_CUTOFF_SIGNALS:
            raise ValueError(f"engine_cutoff_signal must be one of {ENGINE_CUTOFF_SIGNALS}, not '{self.engine_cutoff_signal}'")
        self.engine_stop_delay = float(self.parameters.get("engine_stop_delay_after_probe_contact_s", 1.0))
        self.backup_cutoff_height = float(self.parameters.get("backup_engine_cutoff_height_m", 0.0))
        self.steady_rate = math.radians(float(self.parameters.get("steady_rate_for_engine_stop_deg_per_s", 1.0)))
        self.steady_tilt = math.radians(float(self.parameters.get("steady_tilt_for_engine_stop_deg", 3.0)))
        # Seconds since the contact light first came on in this episode; negative while it has not.
        self.time_since_probe_contact = torch.full((number_of_environments,), -1.0, device=device)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Re-arm the engine and forget the contact light for a new episode."""
        self.engine_is_cut[env_ids] = False
        self.time_since_probe_contact[env_ids] = -1.0

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

    @staticmethod
    def height_above_ground(observation: Observation) -> Tensor:
        """Return the footpad height above the ground straight below the lander, from the observation."""
        return observation.policy[:, NavigationStateLayout.HEIGHT_ABOVE_GROUND].squeeze(1)

    @staticmethod
    def surface_probe_contact(observation: Observation) -> Tensor:
        """Return True where the contact light (a surface probe touches the ground) is on."""
        return observation.policy[:, NavigationStateLayout.SURFACE_PROBE_CONTACT].squeeze(1) > 0.5

    def update_engine_cutoff(self, observation: Observation, control_step_duration: float) -> None:
        """Latch the engine cut-off from the chosen signal (see the class docstring)."""
        altitude = self.height_above_ground(observation)
        if self.engine_cutoff_signal == "height_above_ground":
            self.engine_is_cut |= altitude < self.vertical_controller.engine_cutoff_altitude
            return
        light_seen = self.time_since_probe_contact >= 0.0
        light_on_now = self.surface_probe_contact(observation)
        self.time_since_probe_contact = torch.where(
            light_seen,
            self.time_since_probe_contact + control_step_duration,
            torch.where(light_on_now, torch.zeros_like(self.time_since_probe_contact), self.time_since_probe_contact),
        )
        # The delay is counted from the step that first saw the light; a zero delay stops at once.
        delay_passed = (self.time_since_probe_contact >= self.engine_stop_delay - 1e-6) & (
            self.time_since_probe_contact >= 0.0
        )
        orientation = observation.policy[:, NavigationStateLayout.ORIENTATION_QUATERNION]
        angular_velocity = observation.policy[:, NavigationStateLayout.ANGULAR_VELOCITY]
        steady = (angular_velocity.norm(dim=1) < self.steady_rate) & (tilt_angle_from_vertical(orientation) < self.steady_tilt)
        self.engine_is_cut |= delay_passed & steady
        self.engine_is_cut |= altitude < self.backup_cutoff_height

    def assemble_command(
        self,
        throttle: Tensor,
        orientation: Tensor,
        angular_velocity: Tensor,
        wanted_orientation: Tensor,
        observation: Observation,
        control_step_duration: float,
    ) -> Command:
        """Combine throttle and attitude loop into a Command, and apply the engine cut-off.

        After the cut-off the engine is shut down and the jets are closed: the lander drops the
        rest of the way, and nothing fights the ground afterwards.
        """
        self.update_engine_cutoff(observation, control_step_duration)
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
        altitude = self.height_above_ground(observation)
        target = torch.full_like(altitude, float(self.parameters.get("target_altitude_m", 30.0)))
        acceleration = controller.acceleration_to_hold_altitude(altitude, velocity[:, 2], target)
        throttle = controller.throttle_for_vertical_acceleration(acceleration, mass, orientation)
        return self.assemble_command(
            throttle,
            orientation,
            angular_velocity,
            quaternion_for_thrust_direction(self.vertical_direction, orientation),
            observation,
            control_step_duration,
        )


@register_binding(SlotName.AGENT, "vertical_descent")
class VerticalDescentAgent(ClassicalLandingAgentBase):
    """Step 3: descends straight down on the descent-rate profile and cuts the engine near the ground."""

    def compute_command(self, observation: Observation, control_step_duration: float) -> Command:
        """Return the command of the vertical descent."""
        position, velocity, orientation, angular_velocity, mass = self.split_state(observation)
        controller = self.vertical_controller
        altitude = self.height_above_ground(observation)
        acceleration = controller.acceleration_to_follow_descent_profile(altitude, velocity[:, 2])
        throttle = controller.throttle_for_vertical_acceleration(acceleration, mass, orientation)
        return self.assemble_command(
            throttle,
            orientation,
            angular_velocity,
            quaternion_for_thrust_direction(self.vertical_direction, orientation),
            observation,
            control_step_duration,
        )
