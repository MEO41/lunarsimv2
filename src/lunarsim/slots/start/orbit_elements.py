# SPDX-License-Identifier: BSD-3-Clause
"""``start: orbit_elements`` - the lander in its parking orbit, at the descent-orbit-insertion point (S7).

Apollo's descent ([D-6846] Bennett, NASA TN D-6846, p. 3 and p. 8): from the CSM parking orbit
(about 62 by 58 nautical miles) the LM makes the descent orbit insertion (DOI), "a short
retrograde maneuver of approximately 75 fps [...] at a position in the orbit 180 deg from
powered descent initiation (PDI)", which lowers the orbit to 50,000 ft at PDI; PDI starts "at a
preselected range approximately 260 nautical miles from the landing site near the perilune".

This start builds that geometry around a site (in MCI, float64, time 0 = epoch):

1. the nominal landing time ``t_L`` = half the descent-orbit period + ``pdi_to_touchdown_s``
   (714 s, [D-6846] Table I: touchdown at 11:54 after ignition);
2. the site's direction at ``t_L`` in MCI (the Moon turns about z), and the ground-track heading
   ``approach_azimuth_deg`` at the site (from north, clockwise) fix the orbit plane;
3. PDI lies ``pdi_range_m`` uprange of the site on that plane; the lander starts on a circular
   orbit of altitude ``orbit_altitude_m`` at the opposite point (180 deg from PDI), flying
   prograde, its thrust axis pointing retrograde for the DOI burn.

Dispersions (model choices): ``position_sigma_m`` and ``velocity_sigma_mps`` per axis. Optional
``nav_error`` as for ``terminal_box``. Parameters ``site`` (``latitude_deg``, ``longitude_deg``,
``radius_m``), ``perilune_altitude_m`` (for the descent-orbit period), ``start_before_doi_s`` (the
start lies this long before the DOI point on the circular orbit, so the middle of the DOI burn
falls on it).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.agents.classical.attitude_controller import quaternion_for_thrust_direction
from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.frames import rotation_about_z
from lunarsim.core.message_types import Frame, MessageKey, TruthState
from lunarsim.core.physical_constants import MOON_GRAVITATIONAL_PARAMETER, MOON_ROTATION_RATE
from lunarsim.core.registry import register_binding
from lunarsim.slots.plant.orbit_float64 import site_from_parameters
from lunarsim.slots.start.initial_navigation_error import InitialNavigationError


def rotate_about_axis(vector: Tensor, axis: Tensor, angle: float) -> Tensor:
    """Rotate ``vector`` (3,) about unit ``axis`` (3,) by ``angle`` (Rodrigues)."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return vector * cosine + torch.cross(axis, vector, dim=0) * sine + axis * torch.dot(axis, vector) * (1.0 - cosine)


@register_binding(SlotName.START, "orbit_elements")
class OrbitElementsStart(BindingBase):
    """Parking-orbit start at the DOI point of a descent to ``site`` (see the module text)."""

    provides = frozenset({MessageKey.START_STATE, MessageKey.INITIAL_NAVIGATION_STATE})

    def __init__(self, parameters, context) -> None:
        """Compute the nominal DOI state once; resets add the dispersions."""
        super().__init__(parameters, context)
        get = self.parameters.get
        self.site = site_from_parameters(get("site"))
        dtype, device = torch.float64, self.device
        radius_orbit = self.site.radius_m + float(get("orbit_altitude_m", 111120.0))
        radius_perilune = self.site.radius_m + float(get("perilune_altitude_m", 15240.0))
        semi_major_axis = 0.5 * (radius_orbit + radius_perilune)
        half_period = math.pi * math.sqrt(semi_major_axis**3 / MOON_GRAVITATIONAL_PARAMETER)
        self.landing_time = half_period + float(get("pdi_to_touchdown_s", 714.0))
        basis = self.site.east_north_up(dtype, device)
        azimuth = math.radians(float(get("approach_azimuth_deg", 270.0)))
        heading_fixed = math.sin(azimuth) * basis[:, 0] + math.cos(azimuth) * basis[:, 1]
        to_inertial = rotation_about_z(torch.tensor([MOON_ROTATION_RATE * self.landing_time], dtype=dtype, device=device))[0]
        site_direction = to_inertial @ basis[:, 2]
        heading = to_inertial @ heading_fixed
        normal = torch.nn.functional.normalize(torch.cross(site_direction, heading, dim=0), dim=0)
        pdi_angle = float(get("pdi_range_m", 481520.0)) / self.site.radius_m
        pdi_direction = rotate_about_axis(site_direction, normal, -pdi_angle)
        speed = math.sqrt(MOON_GRAVITATIONAL_PARAMETER / radius_orbit)
        # The lander starts ``start_before_doi_s`` before the DOI point (the burn starts a little
        # later and lasts a while; its middle should fall on the DOI point).
        lead_angle = float(get("start_before_doi_s", 0.0)) * speed / radius_orbit
        start_direction = rotate_about_axis(-pdi_direction, normal, -lead_angle)
        self.nominal_position = radius_orbit * start_direction
        self.nominal_velocity = speed * torch.nn.functional.normalize(torch.cross(normal, start_direction, dim=0), dim=0)
        self.nominal_orientation = quaternion_for_thrust_direction(-self.nominal_velocity.unsqueeze(0))[0]
        self.position_sigma = float(get("position_sigma_m", 0.0))
        self.velocity_sigma = float(get("velocity_sigma_mps", 0.0))
        preset = context.shared_data["lander_preset"]
        count = self.number_of_environments
        self.position = self.nominal_position.repeat(count, 1)
        self.velocity = self.nominal_velocity.repeat(count, 1)
        self.mass = torch.full((count,), float(preset.mass_at_start), dtype=dtype, device=device)
        self.navigation_error = InitialNavigationError(get("nav_error"), count, device)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Draw the dispersed start states of ``env_ids``."""
        count = len(env_ids)
        position_noise = self.position_sigma * torch.randn(count, 3, generator=gen, device=self.device)
        velocity_noise = self.velocity_sigma * torch.randn(count, 3, generator=gen, device=self.device)
        self.position[env_ids] = self.nominal_position + position_noise.to(torch.float64)
        self.velocity[env_ids] = self.nominal_velocity + velocity_noise.to(torch.float64)
        self.navigation_error.reset(env_ids, gen)

    def step(self, bus, dt: float) -> None:
        """Write the start state (MCI, float64, time 0)."""
        count = self.number_of_environments
        dtype = torch.float64
        state = TruthState(
            frame=Frame.MOON_CENTERED_INERTIAL,
            time=torch.zeros(count, dtype=dtype, device=self.device),
            position=self.position.clone(),
            velocity=self.velocity.clone(),
            orientation_quaternion=self.nominal_orientation.repeat(count, 1),
            angular_velocity=torch.zeros(count, 3, dtype=dtype, device=self.device),
            mass=self.mass.clone(),
            contact=torch.zeros(count, 0, dtype=dtype, device=self.device),
        )
        bus.write(MessageKey.START_STATE, state, self)
        self.navigation_error.write(bus, state, self)
