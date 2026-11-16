# SPDX-License-Identifier: BSD-3-Clause
"""Terrain-relative navigation (milestone 6, S5): correlation peak and the fix of a known position error."""

import math

import torch
from helpers import make_binding_context

from lunarsim.assets.lander_preset import lander_preset_from_dictionary
from lunarsim.core.message_types import Frame, MessageKey, SurfaceGeometry, TruthState
from lunarsim.env.scenario_loader import load_scenario
from lunarsim.slots.navigation.terrain_relative_navigation import TerrainRelativeNavigation, correlation_peak
from lunarsim.slots.sensors.shaded_relief_camera import ShadedReliefCamera
from lunarsim.terrain.terrain_height_field import RegularHeightGrid, TerrainHeightField


def test_correlation_finds_a_known_shift():
    """A window cut from a random image at (+3, -2) cells is found there, to a tenth of a cell."""
    generator = torch.Generator().manual_seed(1)
    size, search = 32, 8
    reference = torch.randn(1, size + 2 * search, size + 2 * search, generator=generator)
    reference = torch.nn.functional.avg_pool2d(reference.unsqueeze(0), 3, stride=1, padding=1)[0]
    row0, column0 = search + 3, search - 2
    observed = reference[:, row0 : row0 + size, column0 : column0 + size].clone()
    row, column, peak = correlation_peak(observed, torch.ones_like(observed, dtype=torch.bool), reference)
    assert abs(float(row[0]) - 3.0) < 0.1 and abs(float(column[0]) + 2.0) < 0.1
    assert float(peak[0]) > 0.99


class _World:
    """Test-side writer of the messages the TRN filter reads at reset."""

    provides = frozenset({MessageKey.SURFACE_GEOMETRY, MessageKey.INITIAL_NAVIGATION_STATE})
    requires = frozenset()
    reads_truth = False


def test_terrain_fix_recovers_a_known_position_error():
    """With a perfect map, an estimate 40 m east and 25 m south of the truth gets a fix within 5 m of the truth."""
    scenario = load_scenario("s5_trn_descent", ["env.num_envs=1"])
    context = make_binding_context(1)
    context.shared_data["lander_preset"] = lander_preset_from_dictionary(scenario.lander)
    context.shared_data["sensor_suite_parameters"] = dict(scenario.slots["sensors"].parameters)
    context.shared_data["control_step_duration"] = 0.1
    # Smooth random terrain, 8 m cells, +-4 km, a few tens of metres of relief.
    generator = torch.Generator().manual_seed(4)
    noise = torch.randn(1, 1, 1001, 1001, generator=generator)
    heights = torch.nn.functional.avg_pool2d(noise, 9, stride=1, padding=4)[0, 0] * 60.0
    grid = RegularHeightGrid(-4000.0, -4000.0, 8.0, heights)
    geometry = SurfaceGeometry(site_origin=torch.zeros(1, 3), height_field=TerrainHeightField([grid]))
    parameters = dict(scenario.slots["nav"].parameters)
    parameters["compile"] = False
    parameters["trn"] = dict(parameters["trn"], map_error={"resolution_m": 8.0, "offset_m": [0.0, 0.0], "half_size_m": 3900.0})
    navigation = TerrainRelativeNavigation(parameters, context)
    truth_position = torch.tensor([[150.0, -80.0, 1500.0]])
    error = torch.tensor([[40.0, -25.0, 0.0]])
    level = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    start = TruthState(
        frame=Frame.SITE, time=torch.zeros(1), position=truth_position + error, velocity=torch.zeros(1, 3), orientation_quaternion=level,
        angular_velocity=torch.zeros(1, 3), mass=torch.full((1,), 8000.0), contact=torch.zeros(1, 0),
    )
    world = _World()
    context.bus.write(MessageKey.SURFACE_GEOMETRY, geometry, world)
    context.bus.write(MessageKey.INITIAL_NAVIGATION_STATE, start, world)
    navigation.reset(torch.arange(1), torch.Generator())
    camera_parameters = dict(context.shared_data["sensor_suite_parameters"]["camera"], noise=0.0, ray_march_step_m=4.0)
    camera = ShadedReliefCamera(camera_parameters, 1, "cpu")
    truth = TruthState(
        frame=Frame.SITE, time=torch.zeros(1), position=truth_position, velocity=torch.zeros(1, 3), orientation_quaternion=level,
        angular_velocity=torch.zeros(1, 3), mass=torch.full((1,), 8000.0), contact=torch.zeros(1, 4),
    )
    image = camera.measure(truth, geometry, torch.Generator()).value
    fix, cell, peak = navigation.terrain_fix(image)
    assert float(peak[0]) > 0.5
    assert math.hypot(float(fix[0, 0] - truth_position[0, 0]), float(fix[0, 1] - truth_position[0, 1])) < 5.0
