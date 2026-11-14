# SPDX-License-Identifier: BSD-3-Clause
"""Attitude loop on the reaction-control jets (pure torch).

The controller turns the lander so that its thrust axis (body +z) points along a wanted
direction, and holds the heading. It uses only the estimated attitude and body rates, plus
design data of the lander (inertia, jet couples).

Per body axis:

1. attitude error -> wanted body rate. Near the target the rate is proportional to the error;
   far away it follows the braking curve ``sqrt(2 * a * error)`` so that the jets can still
   stop the turn in time; it is capped at a maximum rate.
2. rate error -> jet on-time. The couple of that axis fires for the time that removes the
   rate error, at most the whole control step. Rate errors inside a small dead band are left
   alone, so the jets do not chatter.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.agents.classical.jet_selection import JetCoupleTable
from lunarsim.assets.lander_preset import LanderPreset
from lunarsim.core.rotation_math import (
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_vector,
)


def quaternion_for_thrust_direction(thrust_direction: Tensor) -> Tensor:
    """Return the attitude (w, x, y, z) that points body +z along ``thrust_direction`` with zero heading change.

    It is the shortest rotation from site +z to the wanted direction. ``thrust_direction`` has
    shape (N, 3) and need not be normalised.
    """
    direction = thrust_direction / thrust_direction.norm(dim=1, keepdim=True).clamp_min(1e-9)
    # Shortest rotation from z to d: axis z x d, half-angle formula q = (1 + z.d, z x d) normalised.
    scalar = 1.0 + direction[:, 2]
    vector = torch.stack((-direction[:, 1], direction[:, 0], torch.zeros_like(scalar)), dim=1)
    quaternion = torch.cat((scalar.unsqueeze(1), vector), dim=1)
    return quaternion / quaternion.norm(dim=1, keepdim=True).clamp_min(1e-9)


class AttitudeController:
    """Phase-plane attitude controller with pulse-width jet commands."""

    def __init__(
        self,
        preset: LanderPreset,
        device: torch.device | str = "cpu",
        proportional_gain_per_s: float = 1.0,
        braking_fraction: float = 0.5,
        maximum_rate_deg_per_s: float = 5.0,
        attitude_dead_band_deg: float = 0.3,
        rate_dead_band_deg_per_s: float = 0.2,
    ) -> None:
        """Store the gains and the angular acceleration that each jet couple gives.

        ``braking_fraction`` is the share of the available angular acceleration that the
        braking curve assumes, which leaves margin for delays.
        """
        self.couples = JetCoupleTable(preset, device=device)
        inertia = torch.tensor([preset.inertia_about_x, preset.inertia_about_y, preset.inertia_about_z], device=device)
        self.angular_acceleration = self.couples.couple_torque / inertia  # (3,) rad/s^2
        self.proportional_gain = float(proportional_gain_per_s)
        self.braking_fraction = float(braking_fraction)
        self.maximum_rate = math.radians(maximum_rate_deg_per_s)
        self.attitude_dead_band = math.radians(attitude_dead_band_deg)
        self.rate_dead_band = math.radians(rate_dead_band_deg_per_s)

    def attitude_error(self, orientation: Tensor, wanted_orientation: Tensor) -> Tensor:
        """Return the rotation vector (rad, body frame, (N, 3)) that takes the attitude to the wanted one."""
        error_quaternion = quaternion_multiply(quaternion_conjugate(orientation), wanted_orientation)
        return quaternion_to_rotation_vector(error_quaternion)

    def wanted_body_rate(self, attitude_error: Tensor) -> Tensor:
        """Return the wanted body rate (N, 3) for an attitude error, with dead band, braking curve and cap."""
        magnitude = (attitude_error.abs() - self.attitude_dead_band).clamp_min(0.0)
        braking_rate = torch.sqrt(2.0 * self.braking_fraction * self.angular_acceleration * magnitude)
        rate = torch.minimum(self.proportional_gain * magnitude, braking_rate).clamp_max(self.maximum_rate)
        return torch.sign(attitude_error) * rate

    def compute_jet_duties(
        self, orientation: Tensor, angular_velocity: Tensor, wanted_orientation: Tensor, control_step_duration: float
    ) -> Tensor:
        """Return the jet duties (N, n_jets) for one control step."""
        rate_error = self.wanted_body_rate(self.attitude_error(orientation, wanted_orientation)) - angular_velocity
        # A rate error inside the dead band is left alone; without this the jets would fire
        # back and forth every control step, because one minimum pulse overshoots a tiny error.
        rate_error = torch.where(rate_error.abs() < self.rate_dead_band, torch.zeros_like(rate_error), rate_error)
        # On-time that removes the rate error, as a share of the control step.
        axis_commands = (rate_error / (self.angular_acceleration * control_step_duration)).clamp(-1.0, 1.0)
        return self.couples.jets_for_axis_commands(axis_commands)
