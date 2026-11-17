# SPDX-License-Identifier: BSD-3-Clause
"""Range noise of the pointed flash LiDAR (CPU): specification line and measured-error table."""

import torch

from lunarsim.slots.sensors.pointed_flash_lidar import PointedFlashLidar

BASE = {
    "rate_hz": 10.0, "field_of_view_deg": 6.0, "pixels_per_side": 4, "minimum_range_m": 2.8,
    "maximum_range_m": 1000.0, "range_noise_m": 0.10, "range_noise_per_m": 0.01, "compile": False,
}


def test_without_a_table_the_noise_is_offset_plus_slope():
    """No table: sigma = 10 cm + 1 % of range, as before the table existed."""
    lidar = PointedFlashLidar(dict(BASE), 1, "cpu")
    sigma = lidar.range_noise_sigma(torch.tensor([0.0, 100.0, 354.0]))
    assert torch.allclose(sigma, torch.tensor([0.10, 1.10, 3.64]), atol=1e-6)


def test_table_interpolates_linearly_and_holds_the_ends():
    """A table replaces the line: linear between pairs, flat below the first and beyond the last, any order."""
    lidar = PointedFlashLidar(dict(BASE, range_noise_table_m=[[354.0, 0.336], [102.0, 0.295], [1009.0, 0.765]]), 1, "cpu")
    sigma = lidar.range_noise_sigma(torch.tensor([[10.0, 102.0, 228.0], [354.0, 681.5, 5000.0]]))
    expected = torch.tensor([[0.295, 0.295, 0.3155], [0.336, 0.5505, 0.765]])
    assert sigma.shape == (2, 3)
    assert torch.allclose(sigma, expected, atol=1e-5)


def test_single_pair_table_is_a_constant():
    """One pair: the same sigma at every range."""
    lidar = PointedFlashLidar(dict(BASE, range_noise_table_m=[[100.0, 0.08]]), 1, "cpu")
    assert torch.allclose(lidar.range_noise_sigma(torch.tensor([3.0, 100.0, 900.0])), torch.full((3,), 0.08))
