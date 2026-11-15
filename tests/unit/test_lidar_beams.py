# SPDX-License-Identifier: BSD-3-Clause
"""Flash LiDAR beams (CPU): pixel grid geometry and slant ranges over flat ground."""

import math

import pytest
import torch
from helpers import make_truth_state

from lunarsim.core.message_types import SurfaceGeometry
from lunarsim.slots.sensors.sensor_models import RangeBeams, flash_lidar_beams


def test_grid_is_centred_on_the_tilted_boresight():
    """8 x 8 pixels over 10 deg around a 30 deg boresight: off-axis angles span 25 to 35 deg."""
    beams = flash_lidar_beams(8, 10.0, 30.0)
    off_axis = [beam["off_axis_deg"] for beam in beams]
    assert len(beams) == 64
    assert min(off_axis) > 25.0 - 1.0 and max(off_axis) < 35.0 + 1.0
    assert sum(off_axis) / len(off_axis) == pytest.approx(30.0, abs=0.2)


def test_ranges_over_flat_ground_follow_the_slant_geometry():
    """Without noise, every pixel of an upright lander at 40 m sees height / cos(off-axis)."""
    beams = flash_lidar_beams(8, 10.0, 30.0)
    parameters = {
        "rate_hz": 10.0, "minimum_range_m": 1.0, "maximum_range_m": 1000.0,
        "range_noise_m": 0.0, "range_noise_per_m": 0.0, "beams": beams,
    }
    lidar = RangeBeams(parameters, 1, "cpu", measures_velocity=False)
    truth = make_truth_state(1)
    truth.position[:] = torch.tensor([0.0, 0.0, 40.0])
    measurement = lidar.measure(truth, SurfaceGeometry(site_origin=torch.zeros(1, 3)), torch.Generator().manual_seed(0))
    expected = torch.tensor([40.0 / math.cos(math.radians(beam["off_axis_deg"])) for beam in beams])
    assert bool(measurement.valid[0])
    assert torch.allclose(measurement.value[0], expected, rtol=1e-5)
