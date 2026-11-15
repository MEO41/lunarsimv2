# SPDX-License-Identifier: BSD-3-Clause
"""Record what every sensor read, what it should have read, and what the filter made of it.

Used by ``scripts/sensor_check.py`` for the visual validation of the sensors (milestone 2). Every
control step, for every environment still in its first episode, it keeps:

- the true state (position, velocity, attitude, body rates, footpad height, contact light, leg
  and body contact forces, mass);
- the **expected** sensor values without noise, computed here from the true state with the same
  beam directions as the sensors: specific force and body rates for the IMU, slant ranges and
  line-of-sight velocities for the altimeter, the Doppler beams and the LiDAR pixels;
- the **measured** values, ``valid`` flags and stated variances from the sensor messages;
- the filter's estimate, its covariance diagonal, its bias estimates, and the true IMU biases.

The recorder is truth-side (``reads_truth``), like the metrics. Nothing flows back to the agent.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import torch

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.core.rotation_math import rotate_vector_body_to_parent, rotate_vector_parent_to_body
from lunarsim.slots.sensors.sensor_models import ray_cast_ground
from lunarsim.slots.sensors.terminal_suite import (
    CONTACT_PROBE_KEY,
    DOPPLER_VELOCIMETER_KEY,
    IMU_KEY,
    LIDAR_KEY,
    PROPELLANT_GAUGE_KEY,
    RADAR_ALTIMETER_KEY,
)

SENSOR_KEYS = {
    "imu": IMU_KEY,
    "altimeter": RADAR_ALTIMETER_KEY,
    "doppler": DOPPLER_VELOCIMETER_KEY,
    "gauge": PROPELLANT_GAUGE_KEY,
    "probe": CONTACT_PROBE_KEY,
    "lidar": LIDAR_KEY,
}


class SensorRecorder:
    """Keeps one record per control step for every environment still in its first episode."""

    slot = SlotName.METRICS
    requires = frozenset({MessageKey.TRUTH_STATE, MessageKey.SURFACE_GEOMETRY, *SENSOR_KEYS.values()})
    provides = frozenset()
    reads_truth = True

    def __init__(self, environment) -> None:
        """Find the sensor suite and the filter of the environment."""
        self.environment = environment
        self.suite = environment.bindings[SlotName.SENSORS]
        self.navigation = environment.bindings[SlotName.NAVIGATION]
        self.dt = environment.scenario.control_step_duration
        count, device = environment.num_envs, environment.device
        self.still_recording = torch.ones(count, dtype=torch.bool, device=device)
        self.previous_time = torch.full((count,), -1.0, device=device)
        self.previous_velocity = None
        self.gravity = torch.tensor([0.0, 0.0, -MOON_SURFACE_GRAVITY], device=device)
        self.records: dict[str, list[numpy.ndarray]] = {}

    def _expected_beams(self, beams, truth, geometry, with_velocity: bool):
        """Return noise-free (ranges, hit, line-of-sight velocities) of body-fixed beams from the true pose."""
        count, number = truth.position.shape[0], beams.directions_body.shape[0]
        orientation = truth.orientation_quaternion.unsqueeze(1).expand(count, number, 4)
        directions = rotate_vector_body_to_parent(orientation, beams.directions_body.expand(count, number, 3))
        origin = (truth.position + geometry.site_origin).unsqueeze(1).expand(count, number, 3)
        distance, hit = ray_cast_ground(origin, directions, geometry, beams.maximum_range)
        hit = hit & (distance >= beams.minimum_range)
        velocity = (directions * truth.velocity.unsqueeze(1)).sum(dim=-1) if with_velocity else None
        return distance, hit, velocity

    def _keep(self, name: str, values: torch.Tensor) -> None:
        """Append one step of a quantity (moved to the CPU)."""
        self.records.setdefault(name, []).append(values.detach().to(torch.float32).cpu().numpy())

    def record(self, step_index: int, command) -> None:
        """Store this control step."""
        bus = self.environment.bus
        truth = bus.read(MessageKey.TRUTH_STATE, self)
        geometry = bus.read(MessageKey.SURFACE_GEOMETRY, self)
        self.still_recording &= truth.time > self.previous_time
        self.previous_time = truth.time.clone()
        if self.previous_velocity is None:
            self.previous_velocity = truth.velocity.clone()
        acceleration = (truth.velocity - self.previous_velocity) / self.dt
        self.previous_velocity = truth.velocity.clone()

        self._keep("valid_record", self.still_recording.float())
        self._keep("time", truth.time)
        for name, values in (
            ("true_position", truth.position), ("true_velocity", truth.velocity),
            ("true_orientation", truth.orientation_quaternion), ("true_body_rates", truth.angular_velocity),
            ("true_footpad_height", truth.footpad_height_above_ground), ("true_probe_contact", truth.surface_probe_contact),
            ("true_leg_force", truth.contact.max(dim=1).values), ("true_body_force", truth.body_contact),
            ("true_mass", truth.mass),
        ):
            self._keep(name, values)
        self._keep("expected_specific_force", rotate_vector_parent_to_body(truth.orientation_quaternion, acceleration - self.gravity))
        suite = self.suite
        for label, beams, with_velocity in (
            ("altimeter", suite.radar_altimeter, False),
            ("doppler", suite.doppler_velocimeter, True),
            ("lidar", suite.lidar, False),
        ):
            if beams is None:
                continue
            distance, hit, velocity = self._expected_beams(beams, truth, geometry, with_velocity)
            self._keep(f"expected_{label}_range", torch.where(hit, distance, torch.full_like(distance, float("nan"))))
            if velocity is not None:
                self._keep(f"expected_{label}_velocity", torch.where(hit, velocity, torch.full_like(velocity, float("nan"))))
        for label, key in SENSOR_KEYS.items():
            if label == "lidar" and suite.lidar is None:
                continue
            measurement = bus.read(key, self)
            self._keep(f"measured_{label}", measurement.value)
            self._keep(f"measured_{label}_valid", measurement.valid.float())
            self._keep(f"measured_{label}_variance", measurement.covariance if measurement.covariance.dim() == 2 else torch.diagonal(measurement.covariance, dim1=1, dim2=2))
        state = self.navigation.filter
        self._keep("estimated_position", state.position)
        self._keep("estimated_velocity", state.velocity)
        self._keep("estimated_orientation", state.orientation)
        self._keep("estimated_covariance_diagonal", torch.diagonal(state.scaled_covariance, dim1=1, dim2=2) * state.scale**2)
        self._keep("estimated_biases", torch.cat((state.accelerometer_bias, state.gyro_bias), dim=1))
        self._keep("true_biases", torch.cat((suite.imu.accelerometer_bias, suite.imu.gyro_bias), dim=1))

    def save(self, path: Path | str, metadata: dict) -> Path:
        """Write all records as arrays (steps, environments, ...) plus the metadata to an ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: numpy.stack(values) for name, values in self.records.items()}
        suite = self.suite
        arrays["doppler_directions_body"] = suite.doppler_velocimeter.directions_body.cpu().numpy()
        arrays["altimeter_directions_body"] = suite.radar_altimeter.directions_body.cpu().numpy()
        if suite.lidar is not None:
            arrays["lidar_directions_body"] = suite.lidar.directions_body.cpu().numpy()
        for name, value in metadata.items():
            arrays[f"meta_{name}"] = numpy.array(value)
        numpy.savez_compressed(path, **arrays)
        return path
