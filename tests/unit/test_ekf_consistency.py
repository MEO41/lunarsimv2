# SPDX-License-Identifier: BSD-3-Clause
"""EKF on a synthetic flight (CPU, milestone 2): the sensor suite measures, the filter estimates.

Known answer: 200 landers fly a smooth trajectory known in closed form (constant acceleration,
constant body rate) over flat ground for 15 s. The real sensor models (configs/sensors/
terminal_suite.yaml) measure it and ``nav: ekf`` estimates it, starting from the S2 first-estimate
errors. A filter whose covariance matches its errors has a mean normalised estimation error
squared (NEES) equal to the number of error components.

Pass criteria and why:
- NEES of position, velocity and attitude (9 components) averaged over the 200 landers and over
  the last 10 s lies between 7 and 11. The 99 % chi-square band for one time step of 200
  landers is about 8.0 to 10.1; averaging over correlated steps leaves the mean near 9 for a
  consistent filter, and a filter that is over- or under-confident by a factor of 1.5 in
  variance falls outside 7 to 11.
- The vertical position error at the end is below 0.2 m (altimeter and Doppler ranges) and the
  velocity error below 0.02 m/s (Doppler, 1.4 mm/s per beam), both for 95 % of the landers.
- The filter never reads TruthState (the bus raises on a truth read without reads_truth).
"""

import math

import pytest
import torch
import yaml
from helpers import make_binding_context

from lunarsim.assets.lander_preset import load_lander_preset
from lunarsim.core.message_types import Frame, MessageKey, SurfaceGeometry, TruthState
from lunarsim.core.rotation_math import (
    quaternion_conjugate,
    quaternion_from_axis_angle,
    quaternion_multiply,
    quaternion_to_rotation_vector,
)
from lunarsim.env.scenario_loader import default_config_directory
from lunarsim.slots.navigation.ekf import ExtendedKalmanFilterNavigation
from lunarsim.slots.sensors.terminal_suite import TerminalSensorSuite
from lunarsim.slots.start.initial_navigation_error import InitialNavigationError

COUNT = 200
DURATION_S = 15.0
CONTROL_STEP = 0.02
CONFIG = default_config_directory()


class _World:
    provides = frozenset({MessageKey.TRUTH_STATE, MessageKey.SURFACE_GEOMETRY, MessageKey.INITIAL_NAVIGATION_STATE})
    requires = frozenset({MessageKey.NAVIGATION_ESTIMATE})
    reads_truth = False


WORLD = _World()


def load(name: str) -> dict:
    """Return a config file as a dictionary without its binding name."""
    with open(CONFIG / name) as file:
        data = yaml.safe_load(file)
    data.pop("binding", None)
    return data


def truth_at(time: float, generator_seed: int = 0) -> TruthState:
    """Return the closed-form true state of every lander at ``time``."""
    torch.manual_seed(generator_seed)
    start_position = torch.stack((torch.rand(COUNT) * 4 - 2, torch.rand(COUNT) * 4 - 2, 30.0 + 10.0 * torch.rand(COUNT)), dim=1)
    start_velocity = torch.stack((torch.rand(COUNT) - 0.5, torch.rand(COUNT) - 0.5, -2.5 - torch.rand(COUNT)), dim=1)
    acceleration = torch.stack((0.05 * torch.ones(COUNT), -0.03 * torch.ones(COUNT), 0.12 + 0.02 * torch.rand(COUNT)), dim=1)
    body_rate = torch.tensor([0.01, -0.02, 0.005]).repeat(COUNT, 1)
    angle = body_rate.norm(dim=1) * time
    rotation = quaternion_from_axis_angle(body_rate / body_rate.norm(dim=1, keepdim=True), angle)
    tilt = quaternion_from_axis_angle(torch.tensor([[1.0, 0.0, 0.0]]).repeat(COUNT, 1), torch.full((COUNT,), math.radians(3.0)))
    position = start_position + start_velocity * time + 0.5 * acceleration * time * time
    return TruthState(
        frame=Frame.SITE,
        time=torch.full((COUNT,), time),
        position=position,
        velocity=start_velocity + acceleration * time,
        orientation_quaternion=quaternion_multiply(tilt, rotation),
        angular_velocity=body_rate,
        mass=torch.full((COUNT,), 7200.0),
        contact=torch.zeros(COUNT, 4),
        footpad_height_above_ground=position[:, 2] - 3.62,
    )


@pytest.fixture(scope="module")
def flight():
    """Fly the synthetic trajectory through the sensors and the filter; return errors and covariances."""
    context = make_binding_context(COUNT)
    preset = load_lander_preset(CONFIG / "landers" / "apollo_lunar_module.yaml")
    suite_parameters = load("sensors/terminal_suite.yaml")
    context.shared_data.update({"lander_preset": preset, "sensor_suite_parameters": suite_parameters})
    suite = TerminalSensorSuite(suite_parameters, context)
    navigation = ExtendedKalmanFilterNavigation(load("nav/ekf.yaml"), context)
    bus = context.bus
    geometry = SurfaceGeometry(site_origin=torch.zeros(COUNT, 3))
    bus.write(MessageKey.SURFACE_GEOMETRY, geometry, WORLD)
    first = truth_at(0.0)
    error = InitialNavigationError({"position_m": 1.0, "velocity_mps": 0.2, "attitude_deg": 0.1}, COUNT, "cpu")
    generator = torch.Generator().manual_seed(7)
    everyone = torch.arange(COUNT)
    error.reset(everyone, generator)
    bus.write(MessageKey.INITIAL_NAVIGATION_STATE, error.estimate(first), WORLD)
    suite.reset(everyone, generator)
    navigation.reset(everyone, generator)
    records = []
    steps = int(DURATION_S / CONTROL_STEP)
    for step in range(steps + 1):
        truth = truth_at(step * CONTROL_STEP)
        bus.write(MessageKey.TRUTH_STATE, truth, WORLD)
        suite.step(bus, CONTROL_STEP)
        navigation.step(bus, CONTROL_STEP)
        state = navigation.filter
        attitude_error = quaternion_to_rotation_vector(
            quaternion_multiply(quaternion_conjugate(state.orientation), truth.orientation_quaternion)
        )
        errors = torch.cat((truth.position - state.position, truth.velocity - state.velocity, attitude_error), dim=1)
        records.append((step * CONTROL_STEP, errors.to(torch.float64), state.covariance[:, :9, :9].clone()))
    return records


def normalised_error(errors: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    """Return e^T P^-1 e per lander."""
    return (errors.unsqueeze(1) @ torch.linalg.solve(covariance, errors.unsqueeze(-1))).squeeze(-1).squeeze(-1)


def test_filter_is_consistent(flight):
    """Mean NEES of position, velocity and attitude over the last 10 s lies between 7 and 11."""
    values = [normalised_error(errors, covariance).mean() for time, errors, covariance in flight if time >= DURATION_S - 10.0]
    mean = float(torch.stack(values).mean())
    print("mean NEES (9 components):", round(mean, 2))
    assert 7.0 < mean < 11.0


def test_estimate_is_accurate_at_the_end(flight):
    """At 15 s: vertical position error below 0.2 m and velocity error below 0.02 m/s for 95 %."""
    _, errors, _ = flight[-1]
    vertical = errors[:, 2].abs()
    velocity = errors[:, 3:6].norm(dim=1)
    print("95th percentile: vertical", float(vertical.quantile(0.95)), "velocity", float(velocity.quantile(0.95)))
    assert float(vertical.quantile(0.95)) < 0.2
    assert float(velocity.quantile(0.95)) < 0.02
