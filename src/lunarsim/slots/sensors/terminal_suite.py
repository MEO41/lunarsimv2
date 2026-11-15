# SPDX-License-Identifier: BSD-3-Clause
"""``sensors: terminal_suite`` - the sensors of scenario S2 (milestone 2).

The design lists the S2 sensors as ``[imu, radar_alt, doppler_vel]``. A slot holds one binding,
so this binding holds one model per sensor, each with its own rate, noise, ``valid`` and
``stamp`` (:mod:`sensor_models`), and writes one message per sensor:

| Message key | Sensor | Value |
| --- | --- | --- |
| ``measurement.imu`` | inertial measurement unit | delta-velocity (3), delta-angle (3), body frame |
| ``measurement.radar_altimeter`` | one beam along body -z | slant range |
| ``measurement.doppler_velocimeter`` | three beams | line-of-sight velocities (3), ranges (3) |
| ``measurement.propellant_gauge`` | propellant quantity gauge | vehicle mass |
| ``measurement.contact_probe`` | surface probe switches | contact light (0 or 1) |

Parameters: one section per sensor (``imu``, ``radar_altimeter``, ``doppler_velocimeter``,
``propellant_gauge``), see ``configs/sensors/terminal_suite.yaml`` for the values and sources.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import MessageKey, SurfaceGeometry, TruthState
from lunarsim.core.registry import register_binding
from lunarsim.slots.sensors.sensor_models import (
    ContactProbeSwitch,
    InertialMeasurementUnit,
    PropellantGauge,
    RangeBeams,
)

IMU_KEY = MessageKey.MEASUREMENT_PREFIX + "imu"
RADAR_ALTIMETER_KEY = MessageKey.MEASUREMENT_PREFIX + "radar_altimeter"
DOPPLER_VELOCIMETER_KEY = MessageKey.MEASUREMENT_PREFIX + "doppler_velocimeter"
PROPELLANT_GAUGE_KEY = MessageKey.MEASUREMENT_PREFIX + "propellant_gauge"
CONTACT_PROBE_KEY = MessageKey.MEASUREMENT_PREFIX + "contact_probe"
TERMINAL_SUITE_KEYS = frozenset(
    {IMU_KEY, RADAR_ALTIMETER_KEY, DOPPLER_VELOCIMETER_KEY, PROPELLANT_GAUGE_KEY, CONTACT_PROBE_KEY}
)


@register_binding(SlotName.SENSORS, "terminal_suite")
class TerminalSensorSuite(BindingBase):
    """IMU, radar altimeter, Doppler velocimeter, propellant gauge and contact probes."""

    requires = frozenset({MessageKey.TRUTH_STATE, MessageKey.SURFACE_GEOMETRY})
    provides = TERMINAL_SUITE_KEYS
    reads_truth = True

    def __init__(self, parameters, context) -> None:
        """Build the sensor models from their parameter sections."""
        super().__init__(parameters, context)
        count, device = self.number_of_environments, self.device
        preset = context.shared_data["lander_preset"]
        self.imu = InertialMeasurementUnit(self.parameters["imu"], count, device)
        self.radar_altimeter = RangeBeams(self.parameters["radar_altimeter"], count, device, measures_velocity=False)
        self.doppler_velocimeter = RangeBeams(self.parameters["doppler_velocimeter"], count, device, measures_velocity=True)
        self.propellant_gauge = PropellantGauge(
            self.parameters["propellant_gauge"], preset.usable_descent_propellant_at_start, count, device
        )
        self.contact_probe = ContactProbeSwitch()
        self.generator: torch.Generator | None = None

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Draw new sensor errors for the given environments (from the sensors slot's generator)."""
        self.generator = gen
        self.imu.reset(env_ids, gen)
        self.radar_altimeter.reset(env_ids)
        self.doppler_velocimeter.reset(env_ids)
        self.propellant_gauge.reset(env_ids, gen)

    def step(self, bus, dt: float) -> None:
        """Measure the true state and write one message per sensor."""
        truth: TruthState = bus.read(MessageKey.TRUTH_STATE, self)
        geometry: SurfaceGeometry = bus.read(MessageKey.SURFACE_GEOMETRY, self)
        if self.generator is None:
            self.generator = torch.Generator(device=self.device)
        bus.write(IMU_KEY, self.imu.measure(truth, dt, self.generator), self)
        bus.write(RADAR_ALTIMETER_KEY, self.radar_altimeter.measure(truth, geometry, self.generator), self)
        bus.write(DOPPLER_VELOCIMETER_KEY, self.doppler_velocimeter.measure(truth, geometry, self.generator), self)
        bus.write(PROPELLANT_GAUGE_KEY, self.propellant_gauge.measure(truth), self)
        bus.write(CONTACT_PROBE_KEY, self.contact_probe.measure(truth), self)
