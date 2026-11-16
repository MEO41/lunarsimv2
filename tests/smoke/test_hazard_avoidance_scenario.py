# SPDX-License-Identifier: BSD-3-Clause
"""S6 smoke test (GPU, milestone 4): pointed LiDAR, onboard hazard map, site choice and site safety.

16 landers start at the Apollo low gate on the Apollo 17 terrain. Checks:

- the pointed LiDAR returns lie on the true ground when rebuilt from the true pose;
- the onboard map's known cells agree with the true ground (up to the navigation error);
- with the true map (``nav: truth_with_hazard_map``) most candidate sites near the flat valley
  floor are judged, and some are safe;
- only truth-side bindings read TruthState (the hazard map navigation does not).
"""

import torch
from isaac_test_support import build_agent, make_environment

from lunarsim.core.message_types import MessageKey, SiteMapLayout
from lunarsim.slots.navigation.onboard_hazard_map import lidar_ground_points
from lunarsim.slots.sensors.pointed_flash_lidar import LIDAR_POINTING_KEY, pixel_directions_body
from lunarsim.slots.sensors.terminal_suite import LIDAR_KEY
from lunarsim.tools.episode_runner import run_agent_in_environment


class _Reader:
    """Test-side reader of the LiDAR messages and the truth (declared like a truth-side binding)."""

    requires = frozenset({LIDAR_KEY, LIDAR_POINTING_KEY, MessageKey.TRUTH_STATE, MessageKey.SURFACE_GEOMETRY})
    provides = frozenset()
    reads_truth = True


def test_lidar_points_lie_on_the_ground_and_the_map_matches_it():
    """Rebuilt from the true pose, LiDAR returns sit on the ground; the map's known cells follow the terrain."""
    environment = make_environment("s6_hazard_avoidance", ["num_envs=16"])
    agent = build_agent(environment)
    reader = _Reader()
    checked = {"frames": 0}
    nav = environment.bindings["nav"]

    def check(step_index: int) -> None:
        """Compare this step's LiDAR returns with the true ground."""
        bus = environment.bus
        ranges = bus.read(LIDAR_KEY, reader)
        if not bool(ranges.valid.any()):
            return
        pointing = bus.read(LIDAR_POINTING_KEY, reader)
        truth = bus.read(MessageKey.TRUTH_STATE, reader)
        geometry = bus.read(MessageKey.SURFACE_GEOMETRY, reader)
        directions = pixel_directions_body(pointing.value, nav.tangent_right, nav.tangent_up)
        points = lidar_ground_points(truth.position, truth.orientation_quaternion, ranges.value, directions) + geometry.site_origin.unsqueeze(1)
        use = ranges.valid.unsqueeze(1) & torch.isfinite(ranges.value)
        points = torch.where(use.unsqueeze(2), points, torch.zeros_like(points))
        ground = geometry.ground_height_at(points[..., 0], points[..., 1])
        sigma = ranges.covariance.sqrt()
        # The return's height error is at most a few range sigmas (range noise along a slanted ray).
        assert bool(((points[..., 2] - ground).abs()[use] <= 5.0 * sigma[use] + 0.05).float().mean() > 0.99)
        checked["frames"] += 1

    run_agent_in_environment(environment, agent, number_of_steps=2500, after_step=check)
    assert checked["frames"] > 10
    heights, known = nav.hazard_map.heights_and_known()
    geometry = environment.bus.read(MessageKey.SURFACE_GEOMETRY, reader)
    if bool(known.any()):
        hazard_map = nav.hazard_map
        cells = hazard_map.cells
        centres = hazard_map.origin + (torch.arange(cells, device=environment.device, dtype=torch.float32) + 0.5).unsqueeze(1) * hazard_map.cell_size
        y, x = torch.meshgrid(centres[:, 1], centres[:, 0], indexing="ij")
        origin = geometry.site_origin
        true_heights = geometry.ground_height_at(origin[:, 0, None, None] + x, origin[:, 1, None, None] + y) - origin[:, 2, None, None]
        error = (heights - true_heights)[known]
        assert float(error.abs().median()) < 1.0  # navigation error (1 m start sigma) plus cell noise
    readers = set(environment.bus.truth_read_log)
    assert readers <= {"TerminalSensorSuite", "SiteSafetyMetrics", "_Reader"}
    environment.close()


def test_true_map_judges_candidates():
    """With the true map every candidate is judged at the first assessment, and some are safe."""
    environment = make_environment("s6_hazard_avoidance_truth_map", ["num_envs=16"])
    agent = build_agent(environment)
    run_agent_in_environment(environment, agent, number_of_steps=5)
    site_map = environment.bindings["nav"].hazard_map.site_map
    assert float(site_map[:, SiteMapLayout.COVERAGE].min()) > 0.99
    assert bool((site_map[:, SiteMapLayout.SAFE] > 0.5).any())
    environment.close()
