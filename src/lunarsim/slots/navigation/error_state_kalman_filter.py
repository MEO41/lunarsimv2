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

The covariance is scaled and float32 (see ``ErrorStateKalmanFilter``); states are float32 like the rest of the simulator.
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
    """Holds and advances the estimate of every environment.

    **Scaled covariance.** The covariance is kept in units of a fixed scale per error-state
    component (the initial sigmas): ``scaled_covariance = D^-1 P D^-1`` with ``D = diag(scale)``,
    so its diagonal starts at 1 and stays well conditioned although the physical variances span
    about 12 orders of magnitude (position ~1 m^2, gyro bias ~1e-11 rad^2/s^2). All covariance
    arithmetic then runs in float32 (consumer GPUs run float64 at about 1/64 of the float32
    speed; float64 made one filter step about 4.7 ms at 1,024 environments). The physical
    covariance is available as ``covariance`` (float64, computed on demand).
    """

    def __init__(self, count: int, device, process_noise: dict[str, float], scale: Tensor | None = None) -> None:
        """Allocate the states; ``process_noise`` gives the white-noise densities in SI units.

        Keys: ``velocity_random_walk`` (m/s/sqrt(s)), ``angle_random_walk`` (rad/sqrt(s)),
        ``accelerometer_bias_random_walk`` (m/s^2/sqrt(s)), ``gyro_bias_random_walk`` (rad/s/sqrt(s)).
        ``scale`` (15,) is the unit of each error-state component in the covariance (default 1).
        """
        self.device = device
        self.position = torch.zeros(count, 3, device=device)
        self.velocity = torch.zeros(count, 3, device=device)
        self.orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(count, 1)
        self.accelerometer_bias = torch.zeros(count, 3, device=device)
        self.gyro_bias = torch.zeros(count, 3, device=device)
        self.scale = (torch.ones(ERROR_STATE_DIMENSION, device=device) if scale is None else scale.to(device).float()).clone()
        self.identity = torch.eye(ERROR_STATE_DIMENSION, device=device)
        self.scaled_covariance = self.identity.repeat(count, 1, 1)
        self.gravity = torch.tensor([0.0, 0.0, -MOON_SURFACE_GRAVITY], device=device)
        self.process_noise = {name: float(value) for name, value in process_noise.items()}
        # Scaling of matrices: F_s = F * scale_j / scale_i, H_s = H * scale_j.
        self.transition_scale = self.scale.unsqueeze(0) / self.scale.unsqueeze(1)

    @property
    def covariance(self) -> Tensor:
        """Return the physical covariance (N, 15, 15), float64."""
        scale = self.scale.to(torch.float64)
        return self.scaled_covariance.to(torch.float64) * scale.unsqueeze(1) * scale.unsqueeze(0)

    @covariance.setter
    def covariance(self, value: Tensor) -> None:
        """Set the covariance from physical units."""
        scale = self.scale.to(value.dtype)
        self.scaled_covariance = (value / scale.unsqueeze(1) / scale.unsqueeze(0)).float()

    def initialize(self, env_ids: Tensor, position: Tensor, velocity: Tensor, orientation: Tensor, sigmas: Tensor) -> None:
        """Start the given environments from an estimate with a diagonal covariance (sigmas: (15,))."""
        self.position[env_ids] = position
        self.velocity[env_ids] = velocity
        self.orientation[env_ids] = orientation
        self.accelerometer_bias[env_ids] = 0.0
        self.gyro_bias[env_ids] = 0.0
        self.scaled_covariance[env_ids] = torch.diag((sigmas.to(self.device).float() / self.scale) ** 2)

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
        eye3 = torch.eye(3, device=self.device)
        transition[:, POSITION, VELOCITY] = eye3 * dt
        transition[:, VELOCITY, ATTITUDE] = -rotation @ skew(corrected_velocity)
        transition[:, VELOCITY, ACCELEROMETER_BIAS] = -rotation * dt
        transition[:, ATTITUDE, ATTITUDE] = rotation_matrix(quaternion_exponential(corrected_angle)).transpose(1, 2)
        transition[:, ATTITUDE, GYRO_BIAS] = -eye3 * dt
        transition = transition * self.transition_scale
        noise = self.process_noise
        diagonal = torch.cat(
            (
                torch.zeros(3, device=self.device),
                torch.full((3,), noise["velocity_random_walk"] ** 2 * dt, device=self.device),
                torch.full((3,), noise["angle_random_walk"] ** 2 * dt, device=self.device),
                torch.full((3,), noise["accelerometer_bias_random_walk"] ** 2 * dt, device=self.device),
                torch.full((3,), noise["gyro_bias_random_walk"] ** 2 * dt, device=self.device),
            )
        ) / self.scale**2
        covariance = transition @ self.scaled_covariance @ transition.transpose(1, 2) + torch.diag(diagonal)

        mask = active.unsqueeze(1)
        self.position = torch.where(mask, position, self.position)
        self.velocity = torch.where(mask, velocity, self.velocity)
        self.orientation = torch.where(mask, orientation, self.orientation)
        self.scaled_covariance = torch.where(
            active.view(-1, 1, 1), 0.5 * (covariance + covariance.transpose(1, 2)), self.scaled_covariance
        )

    def _prepare(self, residual: Tensor, jacobian: Tensor, variance: Tensor, use: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return residual, scaled Jacobian and variance with unused entries neutralised (float32)."""
        residual = torch.where(use, residual, torch.zeros_like(residual)).float()
        variance = torch.where(use, variance, torch.full_like(variance, IGNORED_VARIANCE)).float()
        jacobian = torch.where(use.unsqueeze(-1), jacobian, torch.zeros_like(jacobian)).float() * self.scale
        return residual, jacobian, variance

    def update(self, residual: Tensor, jacobian: Tensor, variance: Tensor, use: Tensor) -> None:
        """Fuse scalar measurements: residual (N, m), jacobian (N, m, 15), variance (N, m), use (N, m) bool."""
        residual, jacobian, variance = self._prepare(residual, jacobian, variance, use)
        covariance = self.scaled_covariance
        innovation_covariance = jacobian @ covariance @ jacobian.transpose(1, 2) + torch.diag_embed(variance)
        gain = torch.linalg.solve(innovation_covariance, jacobian @ covariance).transpose(1, 2)  # (N, 15, m), scaled
        correction = (gain @ residual.unsqueeze(-1)).squeeze(-1) * self.scale
        keep = self.identity - gain @ jacobian
        covariance = keep @ covariance @ keep.transpose(1, 2) + (gain * variance.unsqueeze(1)) @ gain.transpose(1, 2)
        any_used = use.any(dim=1)
        self.scaled_covariance = torch.where(
            any_used.view(-1, 1, 1), 0.5 * (covariance + covariance.transpose(1, 2)), self.scaled_covariance
        )
        correction = torch.where(any_used.unsqueeze(1), correction, torch.zeros_like(correction))
        self.inject(correction)

    def normalised_innovation(self, residual: Tensor, jacobian: Tensor, variance: Tensor, use: Tensor) -> tuple[Tensor, Tensor]:
        """Return the normalised innovation squared y^T S^-1 y (N,) and its degrees of freedom (N,).

        Only the used entries count: the others get a huge variance and no residual, so they add
        nothing. S = H P H^T + R is the innovation covariance of the used entries.
        """
        residual, jacobian, variance = self._prepare(residual, jacobian, variance, use)
        innovation_covariance = jacobian @ self.scaled_covariance @ jacobian.transpose(1, 2) + torch.diag_embed(variance)
        solved = torch.linalg.solve(innovation_covariance, residual.unsqueeze(-1)).squeeze(-1)
        return (residual * solved).sum(dim=1), use.sum(dim=1)

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
