# SPDX-License-Identifier: BSD-3-Clause
"""Frame changes. This is the only module that may change the frame of a TruthState (design rule 7).

Three frames (milestone 6):

- **MCI** (Moon-centred inertial): origin at the Moon's centre, z along the spin axis, x along the
  Moon-fixed x axis at the epoch (time 0). Model choice: the spin axis is fixed in inertial space
  (no libration, no precession) over a landing.
- **MCMF** (Moon-centred Moon-fixed): turns with the Moon about z at the sidereal rate
  (:data:`~lunarsim.core.physical_constants.MOON_ROTATION_RATE`); x through latitude 0,
  longitude 0.
- **SITE**: east-north-up axes at a site (latitude, longitude) on the Moon, origin at the site's
  ground point (radius ``radius_m``), fixed to the Moon.

Positions and velocities convert exactly (rotations and the transport term ``omega x r``);
orientations convert by the same rotation; body rates are body-frame and do not change. All in
float64 for the orbit phases; :func:`convert_truth_state` keeps the dtype of its input.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from lunarsim.core.message_types import Frame, TruthState
from lunarsim.core.physical_constants import MOON_MEAN_RADIUS, MOON_ROTATION_RATE
from lunarsim.core.rotation_math import quaternion_multiply


@dataclass(frozen=True)
class SiteLocation:
    """Where the SITE frame sits: latitude and longitude (deg) and the radius of its origin (m)."""

    latitude_deg: float
    longitude_deg: float
    radius_m: float = MOON_MEAN_RADIUS

    def east_north_up(self, dtype=torch.float64, device="cpu") -> Tensor:
        """Return the 3 x 3 matrix whose columns are east, north and up in MCMF."""
        latitude, longitude = math.radians(self.latitude_deg), math.radians(self.longitude_deg)
        east = (-math.sin(longitude), math.cos(longitude), 0.0)
        north = (-math.sin(latitude) * math.cos(longitude), -math.sin(latitude) * math.sin(longitude), math.cos(latitude))
        up = (math.cos(latitude) * math.cos(longitude), math.cos(latitude) * math.sin(longitude), math.sin(latitude))
        return torch.tensor([east, north, up], dtype=dtype, device=device).T

    def origin_mcmf(self, dtype=torch.float64, device="cpu") -> Tensor:
        """Return the site origin in MCMF, (3,), m."""
        return self.radius_m * self.east_north_up(dtype, device)[:, 2]


def rotation_about_z(angle: Tensor) -> Tensor:
    """Return rotation matrices (N, 3, 3) about z by ``angle`` (N,)."""
    cosine, sine = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack(
        (torch.stack((cosine, -sine, zero), dim=-1), torch.stack((sine, cosine, zero), dim=-1), torch.stack((zero, zero, one), dim=-1)),
        dim=-2,
    )


def quaternion_from_matrix(matrix: Tensor) -> Tensor:
    """Return unit quaternions (..., 4, w x y z) of rotation matrices (..., 3, 3) (Shepperd's method)."""
    m = matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    candidates = torch.stack(
        (
            torch.stack((1.0 + trace, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]), dim=-1),
            torch.stack((m[..., 2, 1] - m[..., 1, 2], 1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]), dim=-1),
            torch.stack((m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0], 1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], m[..., 1, 2] + m[..., 2, 1]), dim=-1),
            torch.stack((m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]), dim=-1),
        ),
        dim=-2,
    )  # (..., 4, 4): each row is 4 q_k q for the k-th largest component
    diagonal = torch.stack((1.0 + trace, 1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], 1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]), dim=-1)
    best = diagonal.argmax(dim=-1, keepdim=True)
    row = torch.gather(candidates, -2, best.unsqueeze(-1).expand(*best.shape[:-1], 1, 4)).squeeze(-2)
    quaternion = torch.nn.functional.normalize(row, dim=-1)
    return quaternion * torch.where(quaternion[..., 0:1] < 0.0, -1.0, 1.0)


def _rotate_state(state: TruthState, matrix: Tensor, position: Tensor, velocity: Tensor, frame: Frame) -> TruthState:
    """Return ``state`` with new position, velocity and the orientation turned by ``matrix`` (N, 3, 3)."""
    rotation = quaternion_from_matrix(matrix).to(state.orientation_quaternion.dtype)
    orientation = quaternion_multiply(rotation, state.orientation_quaternion)
    return dataclasses.replace(
        state, frame=frame, position=position.to(state.position.dtype), velocity=velocity.to(state.velocity.dtype),
        orientation_quaternion=torch.nn.functional.normalize(orientation, dim=1),
    )


def _spin(time: Tensor) -> tuple[Tensor, Tensor]:
    """Return the MCI -> MCMF rotation (N, 3, 3) at ``time`` and the spin vector (3,)."""
    angle = MOON_ROTATION_RATE * time.to(torch.float64)
    spin = torch.tensor([0.0, 0.0, MOON_ROTATION_RATE], dtype=torch.float64, device=time.device)
    return rotation_about_z(-angle), spin


def convert_truth_state(state: TruthState, target_frame: Frame, site: SiteLocation | None = None) -> TruthState:
    """Return ``state`` expressed in ``target_frame``; ``site`` is needed for any change to or from SITE."""
    if not isinstance(target_frame, Frame):
        raise TypeError("target_frame must be a Frame")
    if state.frame is target_frame:
        return state
    if Frame.SITE in (state.frame, target_frame) and site is None:
        raise ValueError("a change to or from the SITE frame needs the SiteLocation")
    position, velocity = state.position.to(torch.float64), state.velocity.to(torch.float64)
    device = position.device
    if state.frame is Frame.SITE:  # SITE -> MCMF first
        basis = site.east_north_up(device=device)
        position = site.origin_mcmf(device=device) + position @ basis.T
        velocity = velocity @ basis.T
        state = _rotate_state(state, basis.expand(position.shape[0], 3, 3), position, velocity, Frame.MOON_CENTERED_MOON_FIXED)
        if target_frame is Frame.MOON_CENTERED_MOON_FIXED:
            return state
        position, velocity = state.position.to(torch.float64), state.velocity.to(torch.float64)
    if state.frame is Frame.MOON_CENTERED_INERTIAL:  # MCI -> MCMF
        to_fixed, spin = _spin(state.time)
        fixed_position = (to_fixed @ position.unsqueeze(2)).squeeze(2)
        fixed_velocity = (to_fixed @ velocity.unsqueeze(2)).squeeze(2) - torch.cross(spin.expand_as(fixed_position), fixed_position, dim=1)
        state = _rotate_state(state, to_fixed, fixed_position, fixed_velocity, Frame.MOON_CENTERED_MOON_FIXED)
        if target_frame is Frame.MOON_CENTERED_MOON_FIXED:
            return state
        position, velocity = state.position.to(torch.float64), state.velocity.to(torch.float64)
    # Now in MCMF.
    if target_frame is Frame.MOON_CENTERED_INERTIAL:
        to_fixed, spin = _spin(state.time)
        to_inertial = to_fixed.transpose(1, 2)
        inertial_velocity = velocity + torch.cross(spin.expand_as(position), position, dim=1)
        return _rotate_state(
            state, to_inertial, (to_inertial @ position.unsqueeze(2)).squeeze(2), (to_inertial @ inertial_velocity.unsqueeze(2)).squeeze(2),
            Frame.MOON_CENTERED_INERTIAL,
        )
    basis = site.east_north_up(device=device)
    site_position = (position - site.origin_mcmf(device=device)) @ basis
    site_velocity = velocity @ basis
    return _rotate_state(state, basis.T.expand(position.shape[0], 3, 3), site_position, site_velocity, Frame.SITE)


def specific_orbital_energy(state: TruthState, site: SiteLocation | None = None) -> Tensor:
    """Return the specific mechanical energy (N,) J/kg in the MCI frame: v^2 / 2 - mu / r (point-mass Moon)."""
    from lunarsim.core.physical_constants import MOON_GRAVITATIONAL_PARAMETER

    inertial = convert_truth_state(state, Frame.MOON_CENTERED_INERTIAL, site)
    position, velocity = inertial.position.to(torch.float64), inertial.velocity.to(torch.float64)
    return 0.5 * (velocity**2).sum(dim=1) - MOON_GRAVITATIONAL_PARAMETER / position.norm(dim=1)
