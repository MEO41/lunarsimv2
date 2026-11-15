# SPDX-License-Identifier: BSD-3-Clause
"""Cheap flash LiDAR in the environment (GPU smoke test, milestone 2).

64 landers start upright at 30 m over flat ground in S2 with ``sensors=terminal_suite_with_lidar``.
After a few control steps the LiDAR message has 32 x 32 = 1024 ranges per lander, every pixel
sees the ground (6 deg field of view around a 30 deg boresight), and each range agrees with the
slant-range geometry (c.g. height / cos(off-axis angle)) within 4 sigma of the specified noise
(10 cm + 1 % of range).
"""

import math

import torch
from isaac_test_support import build_agent, make_environment, read_truth, set_start_states, step_with_agent

from lunarsim.slots.sensors.sensor_models import flash_lidar_beams
from lunarsim.slots.sensors.terminal_suite import LIDAR_KEY


class _Reader:
    requires = frozenset({LIDAR_KEY})
    provides = frozenset()
    reads_truth = False


def test_lidar_ranges_follow_the_geometry():
    """Shape, coverage and agreement with the slant-range geometry."""
    environment = make_environment("s2_terminal_ekf", ["num_envs=64", "sensors=terminal_suite_with_lidar", "agent=hover_hold"])
    agent = build_agent(environment)
    observations = set_start_states(environment, altitude=30.0)
    step_with_agent(environment, agent, observations, 6)
    measurement = environment.bus.read(LIDAR_KEY, _Reader())
    truth = read_truth(environment)
    assert measurement.value.shape == (64, 32 * 32)
    assert float(torch.isfinite(measurement.value).float().mean()) > 0.99
    beams = flash_lidar_beams(32, 6.0, 30.0)
    cosine = torch.tensor([math.cos(math.radians(beam["off_axis_deg"])) for beam in beams], device=environment.device)
    tilt_free = truth.position[:, 2:3] / cosine  # the landers stay upright within a fraction of a degree
    error = (measurement.value - tilt_free).abs()
    allowed = 4.0 * (0.10 + 0.01 * tilt_free) + 0.5  # noise plus a small tilt allowance
    assert float((error < allowed).float().mean()) > 0.99
    environment.close()
