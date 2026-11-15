# SPDX-License-Identifier: BSD-3-Clause
"""``start: terminal_box`` - random start states for the terminal landing phase."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.assets.lander_preset import LanderPreset
from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import Frame, MessageKey, TruthState
from lunarsim.core.registry import register_binding
from lunarsim.slots.start.initial_navigation_error import InitialNavigationError
from lunarsim.core.rotation_math import quaternion_from_axis_angle


def sample_uniform(low: float, high: float, count: int, generator: torch.Generator, device) -> Tensor:
    """Draw ``count`` uniform numbers in [low, high] from ``generator``."""
    return low + (high - low) * torch.rand(count, generator=generator, device=device)


@register_binding(SlotName.START, "terminal_box")
class TerminalBoxStart(BindingBase):
    """Samples each start quantity uniformly from its range.

    Parameters (each a ``[low, high]`` pair):

    - ``altitude_m``: height of the footpad plane above the ground.
    - ``v_vertical_mps``: vertical speed (negative is down).
    - ``v_horizontal_mps``: each of the two horizontal velocity components.
    - ``tilt_deg``: tilt angle from the vertical; the tilt direction is uniform in azimuth.
    - ``horizontal_position_m`` (optional, default ``[0, 0]``): each horizontal position component.
    - ``nav_error`` (optional): error of the first navigation estimate, see ``initial_navigation_error.py``.
    """

    provides = frozenset({MessageKey.START_STATE, MessageKey.INITIAL_NAVIGATION_STATE})

    def __init__(self, parameters, context) -> None:
        """Allocate the sampled start state of every environment."""
        super().__init__(parameters, context)
        preset: LanderPreset = context.shared_data["lander_preset"]
        self.center_of_mass_height = preset.center_of_mass_height_above_footpad_plane
        count = self.number_of_environments
        self.position = torch.zeros(count, 3, device=self.device)
        self.position[:, 2] = self.center_of_mass_height + float(self.parameters["altitude_m"][1])
        self.velocity = torch.zeros(count, 3, device=self.device)
        self.orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(count, 1)
        self.mass = torch.full((count,), preset.mass_at_start, device=self.device)
        self.angular_velocity = torch.zeros(count, 3, device=self.device)
        self.fixed_start_states: tuple[Tensor, Tensor, Tensor] | None = None
        self.navigation_error = InitialNavigationError(self.parameters.get("nav_error"), count, self.device)

    def set_fixed_start_states(
        self, position: Tensor, velocity: Tensor, orientation_quaternion: Tensor, angular_velocity: Tensor | None = None
    ) -> None:
        """Use the given per-environment states at every reset instead of sampling.

        ``position`` is the centre-of-mass position (N, 3) in the site frame. ``angular_velocity``
        (body rates, rad/s, optional, default zero) lets sensor checks start a lander turning.
        This is how a run replays a fixed list of start states (physics tests, held-out sets).
        """
        self.fixed_start_states = (
            position.to(self.device).clone(),
            velocity.to(self.device).clone(),
            orientation_quaternion.to(self.device).clone(),
        )
        rates = torch.zeros_like(self.fixed_start_states[0]) if angular_velocity is None else angular_velocity.to(self.device)
        self.fixed_start_angular_velocity = rates.clone()

    def clear_fixed_start_states(self) -> None:
        """Go back to random sampling."""
        self.fixed_start_states = None

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Sample a new start state for the given environments, using only ``gen``."""
        if self.fixed_start_states is not None:
            position, velocity, orientation = self.fixed_start_states
            self.position[env_ids] = position[env_ids]
            self.velocity[env_ids] = velocity[env_ids]
            self.orientation[env_ids] = orientation[env_ids]
            self.angular_velocity[env_ids] = self.fixed_start_angular_velocity[env_ids]
            self.navigation_error.reset(env_ids, gen)
            return
        count = len(env_ids)
        parameters = self.parameters

        def draw(name: str, default=None) -> Tensor:
            """Draw one uniform sample per environment from the range named ``name``."""
            low, high = parameters.get(name, default)
            return sample_uniform(float(low), float(high), count, gen, self.device)

        altitude = draw("altitude_m")
        vertical_speed = draw("v_vertical_mps")
        horizontal_velocity_x = draw("v_horizontal_mps")
        horizontal_velocity_y = draw("v_horizontal_mps")
        tilt = draw("tilt_deg") * (math.pi / 180.0)
        tilt_azimuth = sample_uniform(0.0, 2.0 * math.pi, count, gen, self.device)
        position_x = draw("horizontal_position_m", default=[0.0, 0.0])
        position_y = draw("horizontal_position_m", default=[0.0, 0.0])

        self.position[env_ids] = torch.stack((position_x, position_y, altitude + self.center_of_mass_height), dim=1)
        self.velocity[env_ids] = torch.stack((horizontal_velocity_x, horizontal_velocity_y, vertical_speed), dim=1)
        tilt_axis = torch.stack((torch.cos(tilt_azimuth), torch.sin(tilt_azimuth), torch.zeros_like(tilt)), dim=1)
        self.orientation[env_ids] = quaternion_from_axis_angle(tilt_axis, tilt)
        self.angular_velocity[env_ids] = 0.0
        # Drawn from its own generator, after the start state: the start states do not change.
        self.navigation_error.reset(env_ids, gen)

    def step(self, bus, dt: float) -> None:
        """Write the sampled start state. The plant reads it for the environments that reset."""
        count = self.number_of_environments
        state = TruthState(
            frame=Frame.SITE,
            time=torch.zeros(count, device=self.device),
            position=self.position.clone(),
            velocity=self.velocity.clone(),
            orientation_quaternion=self.orientation.clone(),
            angular_velocity=self.angular_velocity.clone(),
            mass=self.mass.clone(),
            contact=torch.zeros(count, 0, device=self.device),
        )
        bus.write(MessageKey.START_STATE, state, self)
        self.navigation_error.write(bus, state, self)
