# SPDX-License-Identifier: BSD-3-Clause
"""Batched sensor models for the terminal landing phase (milestone 2).

Every model turns the true state into a :class:`~lunarsim.core.message_types.Measurement` with a
rate, noise, a ``valid`` flag and a capture time ``stamp``. All numbers come from the sensor
config file (``configs/sensors/terminal_suite.yaml``), where each has its public source.

- :class:`SampleSchedule`: one rate per sensor on the shared clock. A sensor samples in the
  first control step at or after each multiple of ``1 / rate``; its ``stamp`` is that step's time.
- :class:`InertialMeasurementUnit`: delta-velocity and delta-angle over one control step (what a
  strapdown IMU delivers after integrating its fast samples), with a constant turn-on bias per
  episode and white noise (velocity and angle random walk).
- :class:`RangeBeams`: beams fixed to the body that measure the slant range to the ground
  (ray cast on the surface's ground lookup) and, optionally, the line-of-sight velocity. Used for
  the radar altimeter (one beam) and the Doppler velocimeter (three beams).
- :class:`PropellantGauge`: vehicle mass from a propellant quantity gauge with a fixed error.
- :class:`ContactProbeSwitch`: the contact light of the surface probes, passed on unchanged.

Sensors sit at the vehicle centre of mass (model choice: no lever arm).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.core.message_types import Measurement, SurfaceGeometry, TruthState
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY, STANDARD_GRAVITY
from lunarsim.core.rotation_math import rotate_vector_body_to_parent, rotate_vector_parent_to_body

RAY_CAST_ITERATIONS = 8


class SampleSchedule:
    """Decides, per environment, in which control steps a sensor of a given rate samples."""

    def __init__(self, rate_hz: float | None, count: int, device) -> None:
        """``rate_hz`` None means every control step."""
        self.rate = None if rate_hz is None else float(rate_hz)
        self.last_index = torch.full((count,), -1, dtype=torch.long, device=device)

    def reset(self, env_ids: Tensor) -> None:
        """Make the next control step a sample step."""
        self.last_index[env_ids] = -1

    def due(self, time: Tensor) -> Tensor:
        """Return True where the sensor samples in this control step, and remember it."""
        if self.rate is None:
            return torch.ones_like(self.last_index, dtype=torch.bool)
        index = torch.floor(time * self.rate + 1e-6).to(torch.long)
        due = index > self.last_index
        self.last_index = torch.where(due, index, self.last_index)
        return due


def ray_cast_ground(origin: Tensor, direction: Tensor, geometry: SurfaceGeometry, maximum_range: float) -> tuple[Tensor, Tensor]:
    """Return (range, hit) of rays from world ``origin`` along unit ``direction`` to the ground.

    Shapes (..., 3) for origin and direction, (...) for the results. Flat ground (no height
    field) is solved exactly; on a height field, Newton steps on the height along the ray start
    from the flat-ground guess below the origin. A ray that points up, or whose ground point is
    farther than ``maximum_range``, does not hit.
    """
    down = -direction[..., 2]
    pointing_down = down > 0.05
    ground_below = geometry.ground_height_at(origin[..., 0], origin[..., 1])
    distance = ((origin[..., 2] - ground_below) / down.clamp_min(0.05)).clamp_min(0.0)
    if geometry.height_field is not None:
        for _ in range(RAY_CAST_ITERATIONS):
            point = origin + distance.unsqueeze(-1) * direction
            ground = geometry.ground_height_at(point[..., 0], point[..., 1])
            slope_x, slope_y = geometry.ground_gradient_at(point[..., 0], point[..., 1])
            gap = point[..., 2] - ground
            derivative = direction[..., 2] - slope_x * direction[..., 0] - slope_y * direction[..., 1]
            step = gap / torch.where(derivative.abs() > 1e-3, derivative, torch.full_like(derivative, -1e-3))
            distance = (distance - step).clamp_min(0.0)
    hit = pointing_down & (distance <= maximum_range) & torch.isfinite(distance)
    return distance, hit


class InertialMeasurementUnit:
    """Delta-velocity (specific force times dt) and delta-angle in the body frame, every control step.

    Value layout (N, 6): delta-velocity x, y, z [m/s], delta-angle x, y, z [rad]. The specific
    force is the true acceleration (from the change of the true velocity over the control step)
    minus the lunar gravity, turned into the body frame. Errors: a constant bias per episode,
    drawn at reset with the configured one-sigma values, plus white noise per step
    (``velocity_random_walk * sqrt(dt)``, ``angle_random_walk * sqrt(dt)``). The first sample
    after a reset is not valid (no previous velocity).
    """

    def __init__(self, parameters: dict, count: int, device) -> None:
        """Convert the datasheet values to SI units."""
        degree = math.pi / 180.0
        self.gyro_bias_sigma = float(parameters["gyro_bias_deg_per_h"]) * degree / 3600.0
        self.angle_random_walk = float(parameters["angle_random_walk_deg_per_sqrt_h"]) * degree / 60.0
        self.accelerometer_bias_sigma = float(parameters["accelerometer_bias_micro_g"]) * 1e-6 * STANDARD_GRAVITY
        self.velocity_random_walk = float(parameters["accelerometer_noise_micro_g_per_sqrt_hz"]) * 1e-6 * STANDARD_GRAVITY
        self.gyro_bias = torch.zeros(count, 3, device=device)
        self.accelerometer_bias = torch.zeros(count, 3, device=device)
        self.previous_velocity = torch.zeros(count, 3, device=device)
        self.previous_angular_velocity = torch.zeros(count, 3, device=device)
        self.has_previous = torch.zeros(count, dtype=torch.bool, device=device)
        self.gravity = torch.tensor([0.0, 0.0, -MOON_SURFACE_GRAVITY], device=device)

    def reset(self, env_ids: Tensor, generator: torch.Generator) -> None:
        """Draw new turn-on biases and forget the previous state."""
        count, device = len(env_ids), self.gyro_bias.device
        self.gyro_bias[env_ids] = self.gyro_bias_sigma * torch.randn(count, 3, generator=generator, device=device)
        self.accelerometer_bias[env_ids] = self.accelerometer_bias_sigma * torch.randn(count, 3, generator=generator, device=device)
        self.has_previous[env_ids] = False

    def measure(self, truth: TruthState, dt: float, generator: torch.Generator) -> Measurement:
        """Return this control step's delta-velocity and delta-angle."""
        count, device = truth.position.shape[0], truth.position.device
        acceleration = (truth.velocity - self.previous_velocity) / dt
        specific_force_body = rotate_vector_parent_to_body(truth.orientation_quaternion, acceleration - self.gravity)
        mean_rate = 0.5 * (truth.angular_velocity + self.previous_angular_velocity)
        delta_velocity = (specific_force_body + self.accelerometer_bias) * dt
        delta_velocity = delta_velocity + self.velocity_random_walk * math.sqrt(dt) * torch.randn(count, 3, generator=generator, device=device)
        delta_angle = (mean_rate + self.gyro_bias) * dt
        delta_angle = delta_angle + self.angle_random_walk * math.sqrt(dt) * torch.randn(count, 3, generator=generator, device=device)
        valid = self.has_previous.clone()
        self.previous_velocity = truth.velocity.clone()
        self.previous_angular_velocity = truth.angular_velocity.clone()
        self.has_previous[:] = True
        variance = torch.cat(
            (
                torch.full((count, 3), self.velocity_random_walk**2 * dt, device=device),
                torch.full((count, 3), self.angle_random_walk**2 * dt, device=device),
            ),
            dim=1,
        )
        return Measurement(value=torch.cat((delta_velocity, delta_angle), dim=1), valid=valid, stamp=truth.time.clone(), covariance=variance)


class RangeBeams:
    """Body-fixed beams: slant range to the ground and, optionally, line-of-sight velocity.

    Value layout (N, beams) for an altimeter, or (N, 2 * beams) with ``measures_velocity``:
    line-of-sight velocities first (velocity of the vehicle along the beam direction, m/s),
    then ranges (m). A beam without a ground return (pointing up, farther than the maximum
    range, closer than the minimum range) gives NaN for its values. ``valid`` is True when at
    least one beam has a return in a sample step. Noise: one sigma = ``offset + slope * range``.
    """

    def __init__(self, parameters: dict, count: int, device, measures_velocity: bool) -> None:
        """Build the beam directions in the body frame and read the noise models."""
        self.measures_velocity = measures_velocity
        self.schedule = SampleSchedule(float(parameters["rate_hz"]), count, device)
        self.minimum_range = float(parameters["minimum_range_m"])
        self.maximum_range = float(parameters["maximum_range_m"])
        self.range_noise_offset = float(parameters["range_noise_m"])
        self.range_noise_slope = float(parameters["range_noise_per_m"])
        self.velocity_noise_offset = float(parameters.get("velocity_noise_mps", 0.0))
        self.velocity_noise_slope = float(parameters.get("velocity_noise_per_m", 0.0))
        directions = []
        for beam in parameters["beams"]:
            off_axis = math.radians(float(beam["off_axis_deg"]))
            azimuth = math.radians(float(beam["azimuth_deg"]))
            directions.append(
                (math.sin(off_axis) * math.cos(azimuth), math.sin(off_axis) * math.sin(azimuth), -math.cos(off_axis))
            )
        self.directions_body = torch.tensor(directions, device=device)  # (beams, 3), body -z is straight down
        self.number_of_beams = len(directions)
        self.last_value = torch.full((count, self.value_dimension), float("nan"), device=device)
        self.last_stamp = torch.full((count,), -1.0, device=device)

    @property
    def value_dimension(self) -> int:
        """Return the number of values per measurement."""
        return 2 * self.number_of_beams if self.measures_velocity else self.number_of_beams

    def reset(self, env_ids: Tensor) -> None:
        """Forget the last sample and start the schedule again."""
        self.schedule.reset(env_ids)
        self.last_value[env_ids] = float("nan")
        self.last_stamp[env_ids] = -1.0

    def measure(self, truth: TruthState, geometry: SurfaceGeometry, generator: torch.Generator) -> Measurement:
        """Return the newest sample (held between sample steps); ``valid`` only in sample steps."""
        count, device = truth.position.shape[0], truth.position.device
        due = self.schedule.due(truth.time)
        beams = self.number_of_beams
        orientation = truth.orientation_quaternion.unsqueeze(1).expand(count, beams, 4)
        directions = rotate_vector_body_to_parent(orientation, self.directions_body.expand(count, beams, 3))
        origin = (truth.position + geometry.site_origin).unsqueeze(1).expand(count, beams, 3)
        distance, hit = ray_cast_ground(origin, directions, geometry, self.maximum_range)
        hit = hit & (distance >= self.minimum_range)
        range_sigma = self.range_noise_offset + self.range_noise_slope * distance
        noisy_range = distance + range_sigma * torch.randn(count, beams, generator=generator, device=device)
        values = [torch.where(hit, noisy_range, torch.full_like(distance, float("nan")))]
        variances = [range_sigma**2]
        if self.measures_velocity:
            line_of_sight = (directions * truth.velocity.unsqueeze(1)).sum(dim=-1)
            velocity_sigma = self.velocity_noise_offset + self.velocity_noise_slope * distance
            noisy_velocity = line_of_sight + velocity_sigma * torch.randn(count, beams, generator=generator, device=device)
            values.insert(0, torch.where(hit, noisy_velocity, torch.full_like(distance, float("nan"))))
            variances.insert(0, velocity_sigma**2)
        value = torch.cat(values, dim=1)
        variance = torch.nan_to_num(torch.cat(variances, dim=1), nan=1.0)
        self.last_value = torch.where(due.unsqueeze(1), value, self.last_value)
        self.last_stamp = torch.where(due, truth.time, self.last_stamp)
        valid = due & hit.any(dim=1)
        return Measurement(value=self.last_value.clone(), valid=valid, stamp=self.last_stamp.clone(), covariance=variance)


class PropellantGauge:
    """Vehicle mass from a propellant quantity gauge: true mass plus a fixed error per episode.

    The error is ``accuracy_fraction`` (one sigma) of the propellant load at the start, drawn at
    reset. Value (N, 1): mass in kg.
    """

    def __init__(self, parameters: dict, propellant_at_start: float, count: int, device) -> None:
        """Read the accuracy and the rate."""
        self.error_sigma = float(parameters["accuracy_fraction"]) * float(propellant_at_start)
        self.schedule = SampleSchedule(float(parameters["rate_hz"]), count, device)
        self.error = torch.zeros(count, device=device)
        self.last_value = torch.full((count, 1), float("nan"), device=device)
        self.last_stamp = torch.full((count,), -1.0, device=device)

    def reset(self, env_ids: Tensor, generator: torch.Generator) -> None:
        """Draw a new gauge error and start the schedule again."""
        self.error[env_ids] = self.error_sigma * torch.randn(len(env_ids), generator=generator, device=self.error.device)
        self.schedule.reset(env_ids)
        self.last_value[env_ids] = float("nan")
        self.last_stamp[env_ids] = -1.0

    def measure(self, truth: TruthState) -> Measurement:
        """Return the newest mass reading."""
        due = self.schedule.due(truth.time)
        value = (truth.mass + self.error).unsqueeze(1)
        self.last_value = torch.where(due.unsqueeze(1), value, self.last_value)
        self.last_stamp = torch.where(due, truth.time, self.last_stamp)
        variance = torch.full_like(self.last_value, self.error_sigma**2)
        return Measurement(value=self.last_value.clone(), valid=due.clone(), stamp=self.last_stamp.clone(), covariance=variance)


class ContactProbeSwitch:
    """The surface probes' contact light, every control step, without error. Value (N, 1): 0 or 1."""

    def measure(self, truth: TruthState) -> Measurement:
        """Return the contact light."""
        value = truth.surface_probe_contact.unsqueeze(1).clone()
        return Measurement(
            value=value,
            valid=torch.ones_like(truth.time, dtype=torch.bool),
            stamp=truth.time.clone(),
            covariance=torch.zeros_like(value),
        )
