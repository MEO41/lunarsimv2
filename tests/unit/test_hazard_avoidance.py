# SPDX-License-Identifier: BSD-3-Clause
"""Scenario S6 parts on the CPU: ray marching, pointed LiDAR pixels, onboard map, start, site choice."""

import math

import torch

from lunarsim.assets.lander_preset import lander_preset_from_dictionary
from lunarsim.core.message_types import NavigationStateLayout, Observation, SiteMapLayout, SurfaceGeometry
from lunarsim.core.rotation_math import body_up_axis_in_parent
from lunarsim.env.scenario_loader import load_scenario
from lunarsim.slots.navigation.onboard_hazard_map import OnboardHazardMap, lidar_ground_points
from lunarsim.slots.sensors.pointed_flash_lidar import (
    boresight_frame,
    flash_pixel_tangents,
    march_rays_to_ground,
    pixel_directions_body,
)
from lunarsim.slots.start.approach_gate import braking_attitude
from lunarsim.slots.surface.measured_terrain import rock_area_fraction
from lunarsim.terrain.hazard_assessment import HazardCriteria
from lunarsim.terrain.terrain_height_field import RegularHeightGrid, TerrainHeightField


def step_terrain() -> SurfaceGeometry:
    """Return flat ground at z = 0 with a 1 m high box (a 'rock') over 10 <= x <= 12, all y."""
    cell = 0.25
    x = torch.arange(-80, 81, dtype=torch.float32) * cell + 0.0
    heights = torch.zeros(161, 161)
    columns = (x >= 10.0) & (x <= 12.0)
    heights[:, columns] = 1.0
    grid = RegularHeightGrid(origin_x=-20.0, origin_y=-20.0, cell_size=cell, heights=heights)
    return SurfaceGeometry(site_origin=torch.zeros(1, 3), height_field=TerrainHeightField([grid]))


def test_march_finds_the_first_crossing_in_front_of_a_rock():
    """A shallow ray hits the near face of the rock, not the ground behind it."""
    geometry = step_terrain()
    origin = torch.tensor([[0.0, 0.0, 3.0]])
    # Ray z = 3 - 0.1 x: at the rock (x = 10..12) it is 2.0..1.8 m up, so it passes over and meets
    # the ground behind at x = 30.
    direction = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, -0.1]]), dim=1)
    distance, hit = march_rays_to_ground(origin, direction, geometry, 100.0, 0.25)
    point = origin + distance.unsqueeze(1) * direction
    assert bool(hit[0])
    assert abs(float(point[0, 0]) - 30.0) < 0.3
    # Ray z = 3 - 0.25 x is 0.5 m up at x = 10: it meets the rock's face (a ramp over the cell
    # 9.75 <= x <= 10 on the triangulated grid), not the ground behind it at x = 12.
    direction = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, -0.25]]), dim=1)
    distance, hit = march_rays_to_ground(origin, direction, geometry, 100.0, 0.25)
    point = origin + distance.unsqueeze(1) * direction
    assert bool(hit[0]) and 9.7 < float(point[0, 0]) < 10.1 and 0.4 < float(point[0, 2]) < 0.7


def test_march_on_flat_ground_is_exact_and_upward_rays_miss():
    """Without a height field the range is the exact distance to z = 0."""
    geometry = SurfaceGeometry(site_origin=torch.zeros(2, 3))
    origin = torch.tensor([[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]])
    direction = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 1.0]]), dim=1)
    distance, hit = march_rays_to_ground(origin, direction, geometry, 1000.0, 0.5)
    assert abs(float(distance[0]) - 100.0 * math.sqrt(2.0)) < 1e-3
    assert bool(hit[0]) and not bool(hit[1])


def test_pixel_grid_spans_the_field_of_view_around_the_boresight():
    """Corner pixels sit near half the field of view from the boresight; the frame is orthonormal."""
    right_tangent, up_tangent = flash_pixel_tangents(8, 6.0)
    boresight = torch.nn.functional.normalize(torch.tensor([[0.3, -0.2, -0.9]]), dim=1)
    directions = pixel_directions_body(boresight, right_tangent, up_tangent)
    angles = torch.rad2deg(torch.acos((directions[0] @ boresight[0]).clamp(-1, 1)))
    assert float(angles.max()) < 3.0 * math.sqrt(2.0) + 0.01
    right, up = boresight_frame(boresight)
    assert abs(float((right * boresight).sum())) < 1e-6 and abs(float((up * boresight).sum())) < 1e-6
    assert abs(float(right.norm()) - 1.0) < 1e-6 and abs(float(up.norm()) - 1.0) < 1e-6


def test_onboard_map_fuses_points_and_knows_cells_only_when_precise_enough():
    """Points of a flat 0.2 m plateau give 0.2 m heights; cells with too few noisy points stay unknown."""
    criteria = HazardCriteria()
    hazard_map = OnboardHazardMap({"half_size_m": 10.0, "cell_size_m": 0.5, "maximum_cell_sigma_m": 0.15}, criteria, 1, "cpu")
    xy = torch.stack(torch.meshgrid(torch.linspace(-9.9, 9.9, 80), torch.linspace(-9.9, 9.9, 80), indexing="xy"), dim=-1).reshape(1, -1, 2)
    points = torch.cat((xy, torch.full_like(xy[..., :1], 0.2)), dim=2)
    # Precise points (sigma 0.05 m): every hit cell becomes known.
    hazard_map.add_points(points, torch.full(points.shape[:2], 0.05), torch.ones(points.shape[:2], dtype=torch.bool))
    heights, known = hazard_map.heights_and_known()
    assert bool(known.all())
    assert torch.allclose(heights[known], torch.full_like(heights[known], 0.2), atol=1e-5)
    # Noisy points (sigma 3 m) on a fresh map: not one cell is known.
    hazard_map.reset(torch.tensor([0]))
    hazard_map.add_points(points, torch.full(points.shape[:2], 3.0), torch.ones(points.shape[:2], dtype=torch.bool))
    _, known = hazard_map.heights_and_known()
    assert not bool(known.any())


def test_true_height_map_judges_a_flat_site_safe_and_a_steep_site_unsafe():
    """With the true ground filled in, flat candidates are safe and a 10 deg slope is not."""
    criteria = HazardCriteria()
    hazard_map = OnboardHazardMap({"half_size_m": 10.0, "cell_size_m": 0.5, "candidate_spacing_m": 2.0}, criteria, 2, "cpu")
    cell = 0.5
    flat = RegularHeightGrid(-40.0, -40.0, cell, torch.zeros(161, 161))
    x = torch.arange(161, dtype=torch.float32) * cell - 40.0
    steep = RegularHeightGrid(-40.0, -40.0, cell, (math.tan(math.radians(10.0)) * x).repeat(161, 1))
    for index, grid in enumerate((flat, steep)):
        geometry = SurfaceGeometry(site_origin=torch.zeros(2, 3), height_field=TerrainHeightField([grid]))
        hazard_map.set_true_heights(geometry, torch.tensor([index]))
    hazard_map.assess_due(torch.zeros(2))
    safe = hazard_map.site_map[:, SiteMapLayout.SAFE]
    assert bool((safe[0] > 0.5).all())
    assert not bool((safe[1] > 0.5).any())
    assert hazard_map.geometry()["rows"] == safe.shape[1]


def test_ground_points_rebuild_a_known_ray():
    """A 10 m range straight down from 10 m up lands on the ground below the pose."""
    position = torch.tensor([[1.0, 2.0, 10.0]])
    orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    points = lidar_ground_points(position, orientation, torch.tensor([[10.0]]), torch.tensor([[[0.0, 0.0, -1.0]]]))
    assert torch.allclose(points, torch.tensor([[[1.0, 2.0, 0.0]]]))


def test_braking_attitude_tilts_thrust_against_the_motion():
    """Flying along +x with 16 deg pitch, body +z leans toward -x by 16 deg."""
    quaternion = braking_attitude(torch.tensor([0.0]), torch.tensor([math.radians(16.0)]))
    up = body_up_axis_in_parent(quaternion)[0]
    assert float(up[0]) < 0.0 and abs(float(up[1])) < 1e-6
    assert abs(math.degrees(math.acos(float(up[2]))) - 16.0) < 1e-4


def test_rock_area_fraction_of_the_measured_law():
    """The Surveyor law N(>D) = 0.00234 D^-2.11 between 0.75 and 2 m covers about 0.37 % of the ground."""
    assert abs(rock_area_fraction(0.00234, -2.11, 0.75, 2.0) - 0.00372) < 5e-5


def test_site_pick_divert_moves_to_the_best_safe_site_below_the_decision_altitude():
    """Above the decision altitude nothing moves; below it the target moves to the best safe candidate."""
    from lunarsim.agents.classical.site_pick_divert_agent import SitePickDivertAgent

    scenario = load_scenario("s6_hazard_avoidance", ["env.num_envs=2"])
    preset = lander_preset_from_dictionary(scenario.lander)
    rows = 5
    geometry = {"candidate_origin_m": [-8.0, -8.0], "candidate_spacing_m": 4.0, "rows": rows, "columns": rows}
    agent = SitePickDivertAgent(
        dict(scenario.slots["agent"].parameters), 2, scenario.number_of_jets, "cpu",
        shared_data={"lander_preset": preset, "site_map_geometry": geometry},
    )
    agent.reset(torch.arange(2), torch.Generator())
    site_map = torch.zeros(2, SiteMapLayout.CHANNELS, rows, rows)
    site_map[:, SiteMapLayout.MARGIN] = -1.0
    site_map[:, SiteMapLayout.SAFE, 3, 4] = 1.0  # candidate at (8, 4)
    site_map[:, SiteMapLayout.MARGIN, 3, 4] = 0.5
    policy = torch.zeros(2, NavigationStateLayout.DIMENSION)
    policy[:, 6] = 1.0
    policy[:, 13] = 7000.0
    policy[:, 14] = torch.tensor([150.0, 60.0])  # first env above, second below the 80 m decision altitude
    agent.select_sites(Observation(policy=policy, site_map=site_map))
    assert not bool(agent.decided[0]) and bool(agent.decided[1])
    assert torch.allclose(agent.gate_positions[1, :2], torch.tensor([8.0, 4.0]))
    assert torch.allclose(agent.gate_positions[0, :2], torch.zeros(2))
    agent.reset(torch.arange(2), torch.Generator())
    records = agent.drain_decision_records()
    assert records["site_decided"].tolist() == [False, True]
    assert abs(float(records["chosen_divert_distance_m"][1]) - math.hypot(8.0, 4.0)) < 1e-4
