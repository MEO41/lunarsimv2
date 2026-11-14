# SPDX-License-Identifier: BSD-3-Clause
"""Vertical loop on the descent engine (pure torch).

Two reference shapes, one thrust law:

- hold an altitude (controller build-up step 2), and
- follow a descent-rate profile down to the engine cut-off height (step 3).

Both give a wanted vertical acceleration. The thrust law turns it into a throttle command,
using the estimated mass and the estimated tilt, so the vertical force is right even while
the lander is tilted and while propellant burns.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.assets.lander_preset import LanderPreset
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.core.rotation_math import body_up_axis_in_parent


class VerticalController:
    """Altitude hold and descent-rate tracking."""

    def __init__(
        self,
        preset: LanderPreset,
        altitude_natural_frequency_per_s: float = 1.0,
        altitude_damping_ratio: float = 1.0,
        descent_rate_gain_per_s: float = 2.0,
        braking_deceleration_mps2: float = 0.8,
        maximum_descent_rate_mps: float = 6.0,
        touchdown_descent_rate_mps: float = 0.6,
        constant_rate_below_altitude_m: float = 1.0,
        engine_cutoff_altitude_m: float = 0.25,
    ) -> None:
        """Store the gains and the shape of the descent profile.

        Altitudes are heights of the footpad plane above the ground.
        """
        self.preset = preset
        self.altitude_gain = altitude_natural_frequency_per_s**2
        self.altitude_rate_gain = 2.0 * altitude_damping_ratio * altitude_natural_frequency_per_s
        self.descent_rate_gain = float(descent_rate_gain_per_s)
        self.braking_deceleration = float(braking_deceleration_mps2)
        self.maximum_descent_rate = float(maximum_descent_rate_mps)
        self.touchdown_descent_rate = float(touchdown_descent_rate_mps)
        self.constant_rate_below_altitude = float(constant_rate_below_altitude_m)
        self.engine_cutoff_altitude = float(engine_cutoff_altitude_m)

    def footpad_altitude(self, position: Tensor) -> Tensor:
        """Return the height of the footpad plane above the ground from the centre-of-mass position."""
        return position[:, 2] - self.preset.center_of_mass_height_above_footpad_plane

    def acceleration_to_hold_altitude(self, altitude: Tensor, vertical_speed: Tensor, target_altitude: Tensor) -> Tensor:
        """Return the wanted vertical acceleration (m/s^2, up positive) of a critically damped altitude hold."""
        return self.altitude_gain * (target_altitude - altitude) - self.altitude_rate_gain * vertical_speed

    def descent_rate_reference(self, altitude: Tensor) -> tuple[Tensor, Tensor]:
        """Return (wanted vertical speed, feed-forward acceleration) for an altitude.

        Above ``constant_rate_below_altitude`` the lander brakes with a constant deceleration
        toward the touchdown rate, so the wanted speed is ``-sqrt(v_td^2 + 2 a (h - h_c))``,
        capped at the maximum descent rate. Below it, the wanted speed is the touchdown rate.
        """
        height_above_constant_part = (altitude - self.constant_rate_below_altitude).clamp_min(0.0)
        braking_speed = torch.sqrt(
            self.touchdown_descent_rate**2 + 2.0 * self.braking_deceleration * height_above_constant_part
        )
        on_braking_curve = (height_above_constant_part > 0.0) & (braking_speed < self.maximum_descent_rate)
        wanted_speed = -braking_speed.clamp_max(self.maximum_descent_rate)
        feed_forward = torch.where(on_braking_curve, self.braking_deceleration, 0.0)
        return wanted_speed, feed_forward

    def acceleration_to_follow_descent_profile(self, altitude: Tensor, vertical_speed: Tensor) -> Tensor:
        """Return the wanted vertical acceleration that tracks the descent-rate profile."""
        wanted_speed, feed_forward = self.descent_rate_reference(altitude)
        return feed_forward + self.descent_rate_gain * (wanted_speed - vertical_speed)

    def throttle_for_vertical_acceleration(self, vertical_acceleration: Tensor, mass: Tensor, orientation: Tensor) -> Tensor:
        """Return the throttle (fraction of rated thrust) that gives the wanted vertical acceleration.

        The thrust must carry the weight plus ``m a``; it acts along body +z, so it is divided
        by the vertical share of that axis. The result is clamped to the throttleable range.
        """
        vertical_share = body_up_axis_in_parent(orientation)[:, 2].clamp_min(0.5)
        thrust = mass * (MOON_SURFACE_GRAVITY + vertical_acceleration) / vertical_share
        preset = self.preset
        return (thrust / preset.engine_rated_thrust).clamp(
            preset.engine_minimum_throttle_fraction, preset.engine_maximum_throttle_fraction
        )
