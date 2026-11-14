# SPDX-License-Identifier: BSD-3-Clause
"""Attitude controller (CPU): error signs, wanted attitude, and a rigid-body closed loop."""

import math

import pytest
import torch

from lunarsim.agents.classical.attitude_controller import AttitudeController, quaternion_for_thrust_direction
from lunarsim.assets.lander_preset import load_lander_preset
from lunarsim.core import rotation_math as rotation
from lunarsim.env.scenario_loader import default_config_directory


@pytest.fixture(scope="module")
def preset():
    """Load the Apollo Lunar Module preset."""
    return load_lander_preset(default_config_directory() / "landers" / "apollo_lunar_module.yaml")


def test_wanted_attitude_points_the_thrust_axis(preset):
    """The quaternion for a thrust direction rotates body +z onto that direction."""
    directions = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.0, -0.3, 1.0], [0.1, 0.1, 0.9]])
    quaternion = quaternion_for_thrust_direction(directions)
    unit = directions / directions.norm(dim=1, keepdim=True)
    assert torch.allclose(rotation.body_up_axis_in_parent(quaternion), unit, atol=1e-6)
    assert torch.allclose(quaternion[:, 3], torch.zeros(4), atol=1e-7)  # no heading change


def test_error_sign_and_jet_torque_oppose_the_tilt(preset):
    """A lander rolled +5 degrees about x gets a negative x-axis error and a negative x torque."""
    controller = AttitudeController(preset)
    tilted = rotation.quaternion_from_axis_angle(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([math.radians(5.0)]))
    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    error = controller.attitude_error(tilted, upright)
    assert error[0].tolist() == pytest.approx([-math.radians(5.0), 0.0, 0.0], abs=1e-6)
    duties = controller.compute_jet_duties(tilted, torch.zeros(1, 3), upright, 0.02)
    torque = duties @ torch.tensor(preset.jet_torque_matrix()).T
    assert float(torque[0, 0]) < -1000.0 and abs(float(torque[0, 1])) < 1.0 and abs(float(torque[0, 2])) < 1.0


def test_inside_the_dead_band_no_jet_fires(preset):
    """An error of 0.2 degrees with no rate is inside the 0.3 degree dead band."""
    controller = AttitudeController(preset)
    tilted = rotation.quaternion_from_axis_angle(torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([math.radians(0.2)]))
    duties = controller.compute_jet_duties(tilted, torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), 0.02)
    assert float(duties.abs().sum()) == 0.0


def test_closed_loop_on_an_ideal_rigid_body_rights_a_ten_degree_tilt(preset):
    """Euler-integrated rigid body with ideal jets: a 10 degree tilt about x and y goes below 0.8 degrees in 10 s.

    0.8 degrees is the two-axis dead band (0.3 degrees per axis, 0.42 together) plus the coast
    at up to the rate dead band (0.2 deg/s) before the next correcting pulse.
    """
    controller = AttitudeController(preset)
    torque_matrix = torch.tensor(preset.jet_torque_matrix()).T
    inertia = torch.tensor([preset.inertia_about_x, preset.inertia_about_y, preset.inertia_about_z])
    axis = torch.tensor([[1.0, 1.0, 0.0]])
    orientation = rotation.quaternion_from_axis_angle(axis, torch.tensor([math.radians(10.0)]))
    rate = torch.zeros(1, 3)
    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    step = 0.02
    largest_rate = 0.0
    for _ in range(int(10.0 / step)):
        duties = controller.compute_jet_duties(orientation, rate, upright, step)
        duties = torch.where(duties * step < preset.jet_minimum_pulse_duration, torch.zeros_like(duties), duties)
        torque = duties @ torque_matrix
        rate = rate + (torque - torch.cross(rate, inertia * rate, dim=1)) / inertia * step
        increment = rotation.quaternion_from_axis_angle(rate, rate.norm(dim=1) * step)
        orientation = rotation.quaternion_multiply(orientation, increment)
        largest_rate = max(largest_rate, float(rate.norm()))
    assert float(rotation.tilt_angle_from_vertical(orientation)) < math.radians(0.8)
    assert float(rate.norm()) < math.radians(0.3)
    assert largest_rate < math.radians(8.0)
