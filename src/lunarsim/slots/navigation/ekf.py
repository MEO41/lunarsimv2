# SPDX-License-Identifier: BSD-3-Clause
"""``nav: ekf`` - inertial navigation with altimeter and Doppler updates (milestone 2).

Every control step:

1. propagate the error-state filter with the IMU sample (delta-velocity, delta-angle);
2. fuse the radar altimeter's slant range, and the Doppler velocimeter's three line-of-sight
   velocities and three ranges, when they have a new valid sample (NaN beams are skipped);
3. write the estimate in the 16-value NavigationStateLayout.

The filter never reads ``TruthState``. It starts each episode from the start slot's first
estimate (``initial_navigation_state``, truth plus a sampled error) and uses the surface's ground
lookup as an onboard map (``surface_geometry``; a perfect map, as the design leaves map errors to
milestone 6). Outputs that the filter does not estimate:

- body rates: the IMU's delta-angle over the step minus the estimated gyro bias;
- mass: the newest propellant-gauge reading (the start estimate's mass before the first one);
- height above the ground: estimated position against the onboard map;
- contact light: the probe switches.

Parameters: ``initial_sigma`` (position_m, velocity_mps, attitude_deg, accelerometer_bias_micro_g,
gyro_bias_deg_per_h) and ``process_noise`` (the IMU's random walks plus bias random walks), see
``configs/nav/ekf.yaml``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import (
    MessageKey,
    NavEstimate,
    NavigationStateLayout,
    SurfaceGeometry,
    TruthState,
)
from lunarsim.core.physical_constants import STANDARD_GRAVITY
from lunarsim.core.registry import register_binding
from lunarsim.slots.navigation.error_state_kalman_filter import ERROR_STATE_DIMENSION, ErrorStateKalmanFilter
from lunarsim.slots.sensors.terminal_suite import (
    CONTACT_PROBE_KEY,
    DOPPLER_VELOCIMETER_KEY,
    IMU_KEY,
    PROPELLANT_GAUGE_KEY,
    RADAR_ALTIMETER_KEY,
)


def beam_directions(beams: list[dict], device) -> Tensor:
    """Return body-frame unit vectors (beams, 3) from ``off_axis_deg`` / ``azimuth_deg`` entries."""
    directions = []
    for beam in beams:
        off_axis, azimuth = math.radians(float(beam["off_axis_deg"])), math.radians(float(beam["azimuth_deg"]))
        directions.append((math.sin(off_axis) * math.cos(azimuth), math.sin(off_axis) * math.sin(azimuth), -math.cos(off_axis)))
    return torch.tensor(directions, device=device)


# Aided sensors of the filter, in the order of the health columns.
AIDED_SENSORS = ("radar_altimeter", "doppler_velocimeter")


@register_binding(SlotName.NAVIGATION, "ekf")
class ExtendedKalmanFilterNavigation(BindingBase):
    """Error-state Kalman filter on IMU, radar altimeter and Doppler velocimeter."""

    requires = frozenset(
        {
            MessageKey.INITIAL_NAVIGATION_STATE,
            MessageKey.SURFACE_GEOMETRY,
            IMU_KEY,
            RADAR_ALTIMETER_KEY,
            DOPPLER_VELOCIMETER_KEY,
            PROPELLANT_GAUGE_KEY,
            CONTACT_PROBE_KEY,
        }
    )
    provides = frozenset({MessageKey.NAVIGATION_ESTIMATE})
    reads_truth = False
    # Health columns of the estimate: failed flag and data age per aided sensor.
    health_dimension = 2 * len(AIDED_SENSORS)

    def __init__(self, parameters, context) -> None:
        """Read the filter settings and the beam geometry of the sensor suite."""
        super().__init__(parameters, context)
        count, device = self.number_of_environments, self.device
        preset = context.shared_data["lander_preset"]
        self.center_of_mass_height = preset.center_of_mass_height_above_footpad_plane
        self.default_mass = preset.mass_at_start
        sigma = self.parameters["initial_sigma"]
        degree = math.pi / 180.0
        self.initial_sigmas = torch.tensor(
            [float(sigma["position_m"])] * 3
            + [float(sigma["velocity_mps"])] * 3
            + [float(sigma["attitude_deg"]) * degree] * 3
            + [float(sigma["accelerometer_bias_micro_g"]) * 1e-6 * STANDARD_GRAVITY] * 3
            + [float(sigma["gyro_bias_deg_per_h"]) * degree / 3600.0] * 3,
            device=device,
        )
        noise = self.parameters["process_noise"]
        self.filter = ErrorStateKalmanFilter(
            count,
            device,
            {
                "velocity_random_walk": float(noise["accelerometer_noise_micro_g_per_sqrt_hz"]) * 1e-6 * STANDARD_GRAVITY,
                "angle_random_walk": float(noise["angle_random_walk_deg_per_sqrt_h"]) * degree / 60.0,
                "accelerometer_bias_random_walk": float(noise["accelerometer_bias_random_walk_mps2_per_sqrt_s"]),
                "gyro_bias_random_walk": float(noise["gyro_bias_random_walk_deg_per_h_per_sqrt_s"]) * degree / 3600.0,
            },
        )
        suite = context.shared_data.get("sensor_suite_parameters") or {}
        self.altimeter_directions = beam_directions(suite["radar_altimeter"]["beams"], device)
        self.altimeter_maximum_range = float(suite["radar_altimeter"]["maximum_range_m"])
        self.altimeter_minimum_range = float(suite["radar_altimeter"]["minimum_range_m"])
        self.doppler_minimum_range = float(suite["doppler_velocimeter"]["minimum_range_m"])
        self.doppler_directions = beam_directions(suite["doppler_velocimeter"]["beams"], device)
        self.doppler_maximum_range = float(suite["doppler_velocimeter"]["maximum_range_m"])
        self.mass = torch.full((count,), self.default_mass, device=device)
        self.body_rates = torch.zeros(count, 3, device=device)
        # Per aided sensor (altimeter, Doppler): capture time of the last fused sample, and the
        # filter's own clock (time since the episode start), for the data age in ``health``.
        self.last_fused_stamp = torch.zeros(count, len(AIDED_SENSORS), device=device)
        self.time = torch.zeros(count, device=device)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Start the given environments from the start slot's first estimate."""
        start: TruthState = self.context.bus.read(MessageKey.INITIAL_NAVIGATION_STATE, self)
        self.filter.initialize(
            env_ids,
            start.position[env_ids].to(torch.float32),
            start.velocity[env_ids].to(torch.float32),
            start.orientation_quaternion[env_ids].to(torch.float32),
            self.initial_sigmas,
        )
        start_mass = start.mass[env_ids].to(torch.float32)
        self.mass[env_ids] = torch.where(torch.isnan(start_mass), torch.full_like(start_mass, self.default_mass), start_mass)
        self.body_rates[env_ids] = 0.0
        self.last_fused_stamp[env_ids] = 0.0
        self.time[env_ids] = 0.0

    def step(self, bus, dt: float) -> None:
        """Propagate, fuse the new measurements, and write the estimate."""
        geometry: SurfaceGeometry = bus.read(MessageKey.SURFACE_GEOMETRY, self)
        imu = bus.read(IMU_KEY, self)
        self.propagate_with(imu.value, imu.valid, dt)
        altimeter = bus.read(RADAR_ALTIMETER_KEY, self)
        self.fuse_altimeter(geometry, altimeter.value, altimeter.valid, altimeter.stamp, altimeter.covariance)
        doppler = bus.read(DOPPLER_VELOCIMETER_KEY, self)
        self.fuse_doppler(geometry, doppler.value, doppler.valid, doppler.stamp, doppler.covariance)
        self.finish_step(bus, geometry, dt)

    def propagate_with(self, imu_value: Tensor, imu_valid: Tensor, dt: float) -> None:
        """Propagate with one IMU sample (delta-velocity, delta-angle) and refresh the body rates."""
        delta_velocity, delta_angle = imu_value[:, :3], imu_value[:, 3:6]
        self.filter.propagate(delta_velocity, delta_angle, dt, imu_valid)
        self.body_rates = torch.where(imu_valid.unsqueeze(1), delta_angle / dt - self.filter.gyro_bias, self.body_rates)

    def fuse_altimeter(self, geometry: SurfaceGeometry, value: Tensor, valid: Tensor, stamp: Tensor, covariance: Tensor) -> None:
        """Fuse one radar altimeter message (slant range along body -z)."""
        predicted, jacobian, hit = self.filter.predict_ranges(self.altimeter_directions, geometry, self.altimeter_maximum_range)
        use = valid.unsqueeze(1) & torch.isfinite(value) & hit
        expected = (hit & (predicted >= self.altimeter_minimum_range)).any(dim=1)
        self.fuse(0, torch.nan_to_num(value - predicted), jacobian, covariance, use, stamp, valid, expected)

    def fuse_doppler(self, geometry: SurfaceGeometry, value: Tensor, valid: Tensor, stamp: Tensor, covariance: Tensor) -> None:
        """Fuse one Doppler message (three line-of-sight velocities, three ranges)."""
        beams = self.doppler_directions.shape[0]
        velocity_prediction, velocity_jacobian = self.filter.predict_line_of_sight_velocities(self.doppler_directions)
        range_prediction, range_jacobian, range_hit = self.filter.predict_ranges(
            self.doppler_directions, geometry, self.doppler_maximum_range
        )
        residual = torch.cat((value[:, :beams] - velocity_prediction, value[:, beams:] - range_prediction), dim=1)
        jacobian = torch.cat((velocity_jacobian, range_jacobian), dim=1)
        use = valid.unsqueeze(1) & torch.isfinite(value) & torch.cat((range_hit, range_hit), dim=1)
        expected = (range_hit & (range_prediction >= self.doppler_minimum_range)).any(dim=1)
        self.fuse(1, torch.nan_to_num(residual), jacobian, covariance, use, stamp, valid, expected)

    def finish_step(self, bus, geometry: SurfaceGeometry, dt: float) -> None:
        """Take the gauge and the contact light, advance the filter clock, and write the estimate."""
        gauge = bus.read(PROPELLANT_GAUGE_KEY, self)
        self.mass = torch.where(gauge.valid & torch.isfinite(gauge.value[:, 0]), gauge.value[:, 0], self.mass)
        contact = bus.read(CONTACT_PROBE_KEY, self)
        self.time = self.time + dt
        bus.write(MessageKey.NAVIGATION_ESTIMATE, self.estimate(geometry, contact.value[:, 0]), self)

    def fuse(
        self, sensor: int, residual: Tensor, jacobian: Tensor, variance: Tensor, use: Tensor, stamp: Tensor, sample: Tensor, expected: Tensor
    ) -> None:
        """Fuse one sensor's sample (sensor 0 altimeter, 1 Doppler). The plain filter takes every valid sample.

        ``stamp`` is the capture time, ``sample`` is True where the message holds a new sample,
        ``expected`` is True where the filter predicts a ground return inside the sensor's range limits.
        Subclasses gate the samples here (``ekf_fdi``) or apply them at their capture time.
        """
        self.filter.update(residual, jacobian, variance, use)
        self.last_fused_stamp[:, sensor] = torch.where(use.any(dim=1), stamp, self.last_fused_stamp[:, sensor])

    def estimate(self, geometry: SurfaceGeometry, contact_light: Tensor) -> NavEstimate:
        """Return the estimate in NavigationStateLayout with diagonal variances."""
        state = self.filter
        world = state.position + geometry.site_origin
        ground = geometry.ground_height_at(world[:, 0], world[:, 1])
        height = world[:, 2] - self.center_of_mass_height - ground
        mean = torch.cat(
            (
                state.position,
                state.velocity,
                state.orientation,
                self.body_rates,
                self.mass.unsqueeze(1),
                height.unsqueeze(1),
                contact_light.unsqueeze(1),
            ),
            dim=1,
        )
        diagonal = torch.diagonal(state.covariance, dim1=1, dim2=2).to(torch.float32)  # (N, 15)
        layout = NavigationStateLayout
        variance = torch.zeros(mean.shape[0], layout.DIMENSION, device=self.device)
        variance[:, layout.POSITION] = diagonal[:, 0:3]
        variance[:, layout.VELOCITY] = diagonal[:, 3:6]
        variance[:, 7:10] = 0.25 * diagonal[:, 6:9]  # quaternion x, y, z: half the angle error
        variance[:, layout.HEIGHT_ABOVE_GROUND] = diagonal[:, 2:3]
        return NavEstimate(mean=mean, covariance=variance, health=self.health())

    def health(self) -> Tensor:
        """Return the health columns (N, 2 per aided sensor): declared failed (0 or 1), data age (s).

        Data age = time since the capture of the last fused sample. The plain filter never
        declares a sensor failed.
        """
        failed = self.declared_failed() if hasattr(self, "declared_failed") else torch.zeros_like(self.last_fused_stamp)
        age = (self.time.unsqueeze(1) - self.last_fused_stamp).clamp_min(0.0)
        return torch.stack((failed, age), dim=2).reshape(self.time.shape[0], -1)

    @property
    def error_state_dimension(self) -> int:
        """Return the size of the filter's error state (for tests and logs)."""
        return ERROR_STATE_DIMENSION
