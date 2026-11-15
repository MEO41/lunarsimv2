# SPDX-License-Identifier: BSD-3-Clause
"""Batched error-state Kalman filter for inertial navigation near the surface (milestone 2).

Nominal state per environment (SITE frame): position p, velocity v, attitude quaternion q
(w, x, y, z, body to site), accelerometer bias, gyro bias. Error state (15 values):
``[dp(3), dv(3), dtheta(3), d_accelerometer_bias(3), d_gyro_bias(3)]``, with the attitude error
as a small rotation in the body frame (q_true = q * exp(dtheta)). Formulation after J. Solà,
"Quaternion kinematics for the error-state Kalman filter" (2017, arXiv:1711.02508), sections
5 and 6 (discrete propagation, injection, reset with G = I).

- :meth:`propagate` uses one IMU sample (delta-velocity, delta-angle over the control step).
- :meth:`update` takes any stack of scalar measurements with their predictions and Jacobians;
  entries that are masked out (NaN, not valid) are ignored. Joseph form keeps P symmetric.
- :meth:`predict_ranges` and :meth:`predict_line_of_sight_velocities` build the predictions and
  Jacobians of body-fixed beams on the onboard map (the surface's ground lookup).

Covariances are float64; states are float32 like the rest of the simulator.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.message_types import SurfaceGeometry
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.core.rotation_math import quaternion_from_axis_angle, quaternion_multiply
from lunarsim.slots.sensors.sensor_models import ray_cast_ground

ERROR_STATE_DIMENSION = 15
POSITION, VELOCITY, ATTITUDE, ACCELEROMETER_BIAS, GYRO_BIAS = (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15))
IGNORED_VARIANCE = 1.0e12


def skew(vector: Tensor) -> Tensor:
    """Return the cross-product matrix [v]x, shape (..., 3, 3), so that [v]x w = v x w."""
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    zero = torch.zeros_like(x)
    return torch.stack(
        (torch.stack((zero, -z, y), dim=-1), torch.stack((z, zero, -x), dim=-1), torch.stack((-y, x, zero), dim=-1)), dim=-2
    )


def rotation_matrix(quaternion: Tensor) -> Tensor:
    """Return the rotation matrix (..., 3, 3) of unit quaternions (w, x, y, z), body to parent."""
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), dim=-1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), dim=-1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )


def quaternion_exponential(rotation_vector: Tensor) -> Tensor:
    """Return the unit quaternion of a rotation vector (axis times angle), shape (..., 4)."""
    angle = rotation_vector.norm(dim=-1)
    axis = rotation_vector / angle.clamp_min(1e-12).unsqueeze(-1)
    axis = torch.where(angle.unsqueeze(-1) > 1e-12, axis, torch.zeros_like(axis) + torch.tensor([1.0, 0.0, 0.0], device=axis.device))
    return quaternion_from_axis_angle(axis, angle)


class ErrorStateKalmanFilter:
    """Holds and advances the estimate of every environment."""

    def __init__(self, count: int, device, process_noise: dict[str, float]) -> None:
        """Allocate the states; ``process_noise`` gives the white-noise densities in SI units.

        Keys: ``velocity_random_walk`` (m/s/sqrt(s)), ``angle_random_walk`` (rad/sqrt(s)),
        ``accelerometer_bias_random_walk`` (m/s^2/sqrt(s)), ``gyro_bias_random_walk`` (rad/s/sqrt(s)).
        """
        self.device = device
        self.position = torch.zeros(count, 3, device=device)
        self.velocity = torch.zeros(count, 3, device=device)
        self.orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(count, 1)
        self.accelerometer_bias = torch.zeros(count, 3, device=device)
        self.gyro_bias = torch.zeros(count, 3, device=device)
        self.covariance = torch.eye(ERROR_STATE_DIMENSION, dtype=torch.float64, device=device).repeat(count, 1, 1)
        self.gravity = torch.tensor([0.0, 0.0, -MOON_SURFACE_GRAVITY], device=device)
        self.process_noise = {name: float(value) for name, value in process_noise.items()}
        self.identity = torch.eye(ERROR_STATE_DIMENSION, dtype=torch.float64, device=device)

    def initialize(self, env_ids: Tensor, position: Tensor, velocity: Tensor, orientation: Tensor, sigmas: Tensor) -> None:
        """Start the given environments from an estimate with a diagonal covariance (sigmas: (15,))."""
        self.position[env_ids] = position
        self.velocity[env_ids] = velocity
        self.orientation[env_ids] = orientation
        self.accelerometer_bias[env_ids] = 0.0
        self.gyro_bias[env_ids] = 0.0
        self.covariance[env_ids] = torch.diag(sigmas.to(torch.float64) ** 2)

    def propagate(self, delta_velocity: Tensor, delta_angle: Tensor, dt: float, active: Tensor) -> None:
        """Advance the environments where ``active`` by one IMU sample of length ``dt``."""
        corrected_angle = delta_angle - self.gyro_bias * dt
        corrected_velocity = delta_velocity - self.accelerometer_bias * dt
        orientation = quaternion_multiply(self.orientation, quaternion_exponential(corrected_angle))
        orientation = orientation / orientation.norm(dim=1, keepdim=True)
        rotation = rotation_matrix(orientation)
        acceleration = (rotation @ corrected_velocity.unsqueeze(-1)).squeeze(-1) / dt + self.gravity
        position = self.position + self.velocity * dt + 0.5 * acceleration * dt * dt
        velocity = self.velocity + acceleration * dt

        count = self.position.shape[0]
        transition = self.identity.repeat(count, 1, 1)
        eye3 = torch.eye(3, dtype=torch.float64, device=self.device)
        rotation64 = rotation.to(torch.float64)
        transition[:, POSITION, VELOCITY] = eye3 * dt
        transition[:, VELOCITY, ATTITUDE] = -rotation64 @ skew(corrected_velocity.to(torch.float64))
        transition[:, VELOCITY, ACCELEROMETER_BIAS] = -rotation64 * dt
        transition[:, ATTITUDE, ATTITUDE] = rotation_matrix(quaternion_exponential(corrected_angle)).to(torch.float64).transpose(1, 2)
        transition[:, ATTITUDE, GYRO_BIAS] = -eye3 * dt
        noise = self.process_noise
        diagonal = torch.cat(
            (
                torch.zeros(3, dtype=torch.float64, device=self.device),
                torch.full((3,), noise["velocity_random_walk"] ** 2 * dt, dtype=torch.float64, device=self.device),
                torch.full((3,), noise["angle_random_walk"] ** 2 * dt, dtype=torch.float64, device=self.device),
                torch.full((3,), noise["accelerometer_bias_random_walk"] ** 2 * dt, dtype=torch.float64, device=self.device),
                torch.full((3,), noise["gyro_bias_random_walk"] ** 2 * dt, dtype=torch.float64, device=self.device),
            )
        )
        covariance = transition @ self.covariance @ transition.transpose(1, 2) + torch.diag(diagonal)

        mask = active.unsqueeze(1)
        self.position = torch.where(mask, position, self.position)
        self.velocity = torch.where(mask, velocity, self.velocity)
        self.orientation = torch.where(mask, orientation, self.orientation)
        self.covariance = torch.where(active.view(-1, 1, 1), 0.5 * (covariance + covariance.transpose(1, 2)), self.covariance)

    def update(self, residual: Tensor, jacobian: Tensor, variance: Tensor, use: Tensor) -> None:
        """Fuse scalar measurements: residual (N, m), jacobian (N, m, 15), variance (N, m), use (N, m) bool."""
        residual = torch.where(use, residual, torch.zeros_like(residual)).to(torch.float64)
        variance = torch.where(use, variance, torch.full_like(variance, IGNORED_VARIANCE)).to(torch.float64)
        jacobian = torch.where(use.unsqueeze(-1), jacobian, torch.zeros_like(jacobian)).to(torch.float64)
        covariance = self.covariance
        innovation_covariance = jacobian @ covariance @ jacobian.transpose(1, 2) + torch.diag_embed(variance)
        gain = torch.linalg.solve(innovation_covariance, jacobian @ covariance).transpose(1, 2)  # (N, 15, m)
        correction = (gain @ residual.unsqueeze(-1)).squeeze(-1)
        keep = self.identity - gain @ jacobian
        covariance = keep @ covariance @ keep.transpose(1, 2) + gain @ torch.diag_embed(variance) @ gain.transpose(1, 2)
        any_used = use.any(dim=1)
        self.covariance = torch.where(any_used.view(-1, 1, 1), 0.5 * (covariance + covariance.transpose(1, 2)), self.covariance)
        correction = torch.where(any_used.unsqueeze(1), correction, torch.zeros_like(correction)).to(torch.float32)
        self.inject(correction)

    def inject(self, correction: Tensor) -> None:
        """Add an error-state correction (N, 15) to the nominal state."""
        self.position = self.position + correction[:, POSITION]
        self.velocity = self.velocity + correction[:, VELOCITY]
        orientation = quaternion_multiply(self.orientation, quaternion_exponential(correction[:, ATTITUDE]))
        self.orientation = orientation / orientation.norm(dim=1, keepdim=True)
        self.accelerometer_bias = self.accelerometer_bias + correction[:, ACCELEROMETER_BIAS]
        self.gyro_bias = self.gyro_bias + correction[:, GYRO_BIAS]

    def beam_directions(self, directions_body: Tensor) -> tuple[Tensor, Tensor]:
        """Return (rotation (N, 3, 3), beam directions in the site frame (N, beams, 3))."""
        rotation = rotation_matrix(self.orientation)
        return rotation, torch.einsum("nij,bj->nbi", rotation, directions_body)

    def predict_ranges(
        self, directions_body: Tensor, geometry: SurfaceGeometry, maximum_range: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return predicted slant ranges (N, beams), their Jacobians (N, beams, 15) and a hit mask.

        The ground near the hit point is taken as a plane z = g + g_x x + g_y y. With
        n = (g_x, g_y, -1) and the beam d = R b: range r = (plane height - p_z) / (d_z - g_x d_x -
        g_y d_y), dr/dp = n / denominator, dr/dtheta = -(r / denominator) n^T R [b]x.
        """
        rotation, directions = self.beam_directions(directions_body)
        count, beams = directions.shape[0], directions.shape[1]
        origin = (self.position + geometry.site_origin).unsqueeze(1).expand(count, beams, 3)
        distance, hit = ray_cast_ground(origin, directions, geometry, maximum_range)
        point = origin + distance.unsqueeze(-1) * directions
        slope_x, slope_y = geometry.ground_gradient_at(point[..., 0], point[..., 1])
        normal = torch.stack((slope_x, slope_y, -torch.ones_like(slope_x)), dim=-1)  # (N, beams, 3)
        denominator = directions[..., 2] - slope_x * directions[..., 0] - slope_y * directions[..., 1]
        denominator = torch.where(denominator.abs() > 1e-3, denominator, torch.full_like(denominator, -1e-3))
        jacobian = torch.zeros(count, beams, ERROR_STATE_DIMENSION, device=self.device)
        jacobian[..., POSITION] = normal / denominator.unsqueeze(-1)
        normal_rotated = torch.einsum("nbi,nij->nbj", normal, rotation)  # n^T R
        jacobian[..., ATTITUDE] = -(distance / denominator).unsqueeze(-1) * torch.einsum(
            "nbj,bjk->nbk", normal_rotated, skew(directions_body)
        )
        return distance, jacobian, hit

    def predict_line_of_sight_velocities(self, directions_body: Tensor) -> tuple[Tensor, Tensor]:
        """Return predicted velocities along the beams (N, beams) and their Jacobians (N, beams, 15).

        z = b^T R^T v; dz/dv = (R b)^T; dz/dtheta = b^T [R^T v]x.
        """
        rotation, directions = self.beam_directions(directions_body)
        count, beams = directions.shape[0], directions.shape[1]
        prediction = (directions * self.velocity.unsqueeze(1)).sum(dim=-1)
        velocity_body = (rotation.transpose(1, 2) @ self.velocity.unsqueeze(-1)).squeeze(-1)
        jacobian = torch.zeros(count, beams, ERROR_STATE_DIMENSION, device=self.device)
        jacobian[..., VELOCITY] = directions
        jacobian[..., ATTITUDE] = torch.einsum("bj,njk->nbk", directions_body, skew(velocity_body))
        return prediction, jacobian
