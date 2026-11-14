# SPDX-License-Identifier: BSD-3-Clause
"""Terrain package (CPU): exact height lookup, seamless meshes, detail statistics, height-map reader."""

import math

import numpy
import pytest
import torch

from lunarsim.terrain.lunar_height_map_reader import HeightMapError, LunarHeightMap, parse_planetary_data_system_label
from lunarsim.terrain.measured_terrain_builder import MeasuredTerrainSettings, TerrainBuildError, build_measured_terrain
from lunarsim.terrain.surface_detail import (
    CraterPopulation,
    RockPopulation,
    crater_profile,
    draw_craters,
    draw_power_law_sizes,
    draw_rocks,
)
from lunarsim.terrain.terrain_height_field import RegularHeightGrid, TerrainHeightField, grid_triangles, grid_vertices

CRATERS = CraterPopulation(
    cumulative_coefficient_per_m2=0.0336,
    cumulative_exponent=2.0,
    minimum_diameter_m=1.0,
    maximum_diameter_m=10.0,
    class_shares=(0.005, 0.025, 0.17, 0.30, 0.50),
    class_depth_to_diameter=((0.12, 0.20), (0.12, 0.20), (0.10, 0.15), (0.07, 0.10), (0.02, 0.07)),
    rim_height_to_depth=0.2,
    classes_without_rim=(4,),
)
ROCKS = RockPopulation(
    cumulative_coefficient_per_m2=0.00234,
    cumulative_exponent=-2.11,
    minimum_diameter_m=0.75,
    maximum_diameter_m=2.0,
    height_to_diameter_mean=0.6,
    height_to_diameter_standard_deviation=0.28,
    height_to_diameter_limits=(0.2, 1.0),
)


def wavy_ground(east, north):
    """A smooth synthetic base: slopes and waves of a few metres."""
    return 0.02 * east - 0.01 * north + 1.5 * numpy.sin(east / 37.0) * numpy.cos(north / 23.0)


def triangle_height_by_hand(heights, cell, x, y):
    """Reference: find the triangle of the mesh that contains (x, y) and interpolate on its plane."""
    i, j = int(x // cell), int(y // cell)
    u, v = x / cell - i, y / cell - j
    a, b, c, d = heights[j, i], heights[j, i + 1], heights[j + 1, i], heights[j + 1, i + 1]
    corners = ((0, 0, a), (1, 0, b), (1, 1, d)) if u >= v else ((0, 0, a), (1, 1, d), (0, 1, c))
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = corners
    normal = numpy.cross((x1 - x0, y1 - y0, z1 - z0), (x2 - x0, y2 - y0, z2 - z0))
    return z0 - (normal[0] * (u - x0) + normal[1] * (v - y0)) / normal[2]


def test_height_lookup_is_the_mesh_triangle_plane():
    """The lookup returns the plane of the mesh triangle, and its gradient matches finite differences."""
    random = numpy.random.default_rng(0)
    heights = random.normal(size=(6, 7))
    grid = RegularHeightGrid(0.0, 0.0, 2.0, torch.as_tensor(heights, dtype=torch.float64))
    points = random.uniform(0.0, 10.0, size=(200, 2))
    looked_up = grid.height_at(torch.as_tensor(points[:, 0]), torch.as_tensor(points[:, 1])).numpy()
    expected = [triangle_height_by_hand(heights, 2.0, x, y) for x, y in points]
    assert looked_up == pytest.approx(expected, abs=1e-9)
    x, y = torch.tensor([1.3], dtype=torch.float64), torch.tensor([0.4], dtype=torch.float64)
    slope_x, slope_y = grid.gradient_at(x, y)
    step = 1e-6
    finite_x = (grid.height_at(x + step, y) - grid.height_at(x, y)) / step
    finite_y = (grid.height_at(x, y + step) - grid.height_at(x, y)) / step
    assert float(slope_x) == pytest.approx(float(finite_x), rel=1e-5)
    assert float(slope_y) == pytest.approx(float(finite_y), rel=1e-5)
    assert torch.isfinite(slope_y).all()


def test_grid_triangles_point_up_and_leave_the_hole_open():
    """Every triangle is counter-clockwise from above; the hole removes exactly its cells."""
    grid = RegularHeightGrid(0.0, 0.0, 1.0, torch.zeros(5, 6))
    vertices = grid_vertices(grid)
    full = grid_triangles(5, 6)
    holed = grid_triangles(5, 6, hole=(1, 1, 2, 2))
    assert len(full) == 2 * 4 * 5 and len(holed) == len(full) - 2 * 4
    corner_a, corner_b, corner_c = (vertices[holed[:, k]] for k in range(3))
    normal_z = numpy.cross(corner_b - corner_a, corner_c - corner_a)[:, 2]
    assert (normal_z > 0).all()


def test_terrain_levels_meet_without_a_gap_and_detail_follows_the_statistics():
    """Patch border heights equal the outer mesh; crater and rock counts match the cumulative laws."""
    settings = MeasuredTerrainSettings(
        outer_half_size_m=200.0, outer_cell_size_m=2.0, detail_half_size_m=60.0, detail_cell_size_m=0.25,
        craters=CRATERS, rocks=ROCKS, seed=3,
    )
    terrain = build_measured_terrain(wavy_ground, settings)
    detail = terrain.detail.heights.double()
    edge = torch.cat((detail[0], detail[-1], detail[:, 0], detail[:, -1]))
    count = detail.shape[0]
    lines = torch.as_tensor(-60.0 + 0.25 * numpy.arange(count))
    edge_x = torch.cat((lines, lines, torch.full_like(lines, -60.0), torch.full_like(lines, 60.0)))
    edge_y = torch.cat((torch.full_like(lines, -60.0), torch.full_like(lines, 60.0), lines, lines))
    outer_on_edge = terrain.outer.height_at(edge_x.float(), edge_y.float()).double()
    assert float((edge - outer_on_edge).abs().max()) < 1e-4
    # Inside the patch the detail is really there.
    assert float((detail - terrain.outer.height_at(*torch.meshgrid(lines.float(), lines.float(), indexing="xy")).double()).abs().max()) > 0.2
    area = 120.0**2
    assert len(terrain.craters.diameter) == pytest.approx(CRATERS.expected_count(area), rel=0.2)
    assert len(terrain.rocks.diameter) == pytest.approx(ROCKS.expected_count(area), rel=0.35)
    field = terrain.height_field()
    assert isinstance(field, TerrainHeightField)
    inside = field.height_at(torch.tensor([10.0]), torch.tensor([-5.0]))
    assert float(inside) == pytest.approx(float(terrain.detail.height_at(torch.tensor([10.0]), torch.tensor([-5.0]))))
    outside = field.height_at(torch.tensor([150.0]), torch.tensor([0.0]))
    assert float(outside) == pytest.approx(float(terrain.outer.height_at(torch.tensor([150.0]), torch.tensor([0.0]))))
    meshes = terrain.meshes()
    assert set(meshes) == {"outer", "detail"}


def test_settings_that_cannot_meet_are_rejected():
    """A patch edge off the outer grid lines would leave a gap, so it is refused."""
    with pytest.raises(TerrainBuildError):
        MeasuredTerrainSettings(detail_half_size_m=81.0).check()
    with pytest.raises(TerrainBuildError):
        MeasuredTerrainSettings(detail_cell_size_m=0.3).check()


def test_power_law_sizes_follow_the_cumulative_law():
    """The share of drawn sizes above D matches (D^-b - Dmax^-b) / (Dmin^-b - Dmax^-b)."""
    sizes = draw_power_law_sizes(numpy.random.default_rng(1), 200_000, 1.0, 10.0, 2.0)
    for diameter in (1.5, 3.0, 6.0):
        expected = (diameter**-2 - 10.0**-2) / (1.0 - 10.0**-2)
        assert float((sizes > diameter).mean()) == pytest.approx(expected, rel=0.03)


def test_crater_shapes_and_classes():
    """Bowl depth at the centre, rim at the edge, zero beyond three radii; worn craters have no rim."""
    profile = crater_profile(numpy.array([0.0, 1.0, 2.0, 3.0, 4.0]), depth=1.0, rim_height=0.2)
    assert profile == pytest.approx([-1.0, 0.2, 0.2 * (2.0**-3 - 3.0**-3) / (1 - 3.0**-3), 0.0, 0.0])
    craters = draw_craters(numpy.random.default_rng(2), CRATERS, (0.0, 300.0, 0.0, 300.0))
    ratio = craters.depth / craters.diameter
    assert (ratio >= 0.02).all() and (ratio <= 0.20).all()
    worn = craters.degradation_class == 4
    assert (craters.rim_height[worn] == 0.0).all() and (craters.rim_height[~worn] > 0.0).all()
    assert float(worn.mean()) == pytest.approx(0.5, abs=0.05)
    rocks = draw_rocks(numpy.random.default_rng(3), ROCKS, (0.0, 300.0, 0.0, 300.0))
    assert ((rocks.height / rocks.diameter) >= 0.2).all() and (rocks.diameter <= 2.0).all()


LABEL_TEXT = """PDS_VERSION_ID = PDS3
^IMAGE = ("TEST.TIF", {offset} <BYTES>)
OBJECT = IMAGE_MAP_PROJECTION
  MAP_PROJECTION_TYPE = "EQUIRECTANGULAR"
  A_AXIS_RADIUS = 1737.4 <KM>
  C_AXIS_RADIUS = 1737.4 <KM>
  CENTER_LATITUDE = 20.0 <DEG>
  CENTER_LONGITUDE = 180.0 <DEG>
  MAP_SCALE = 2.0 <METERS/PIXEL>
  LINE_PROJECTION_OFFSET = {line_offset}
  SAMPLE_PROJECTION_OFFSET = {sample_offset}
END_OBJECT = IMAGE_MAP_PROJECTION
OBJECT = IMAGE
  DESCRIPTION = "A test product
    on two lines."
  LINES = {lines}
  LINE_SAMPLES = {samples}
  SAMPLE_TYPE = PC_REAL
  SAMPLE_BITS = 32
END_OBJECT = IMAGE
END
"""


def write_test_product(directory, latitude, longitude, lines=40, samples=50, header_bytes=64):
    """Write a small product whose heights are a plane in map coordinates; return paths and the plane."""
    radius = 1_737_400.0
    map_x = radius * math.radians(longitude - 180.0) * math.cos(math.radians(20.0))
    map_y = radius * math.radians(latitude)
    # Put the site near the middle of the image.
    sample_offset = samples / 2 - map_x / 2.0
    line_offset = lines / 2 + map_y / 2.0
    rows, columns = numpy.mgrid[0:lines, 0:samples]
    pixel_map_x = (columns - sample_offset) * 2.0
    pixel_map_y = (line_offset - rows) * 2.0
    heights = (0.01 * (pixel_map_x - map_x) + 0.02 * (pixel_map_y - map_y) - 1000.0).astype("<f4")
    heights[2, 2] = -3.4028226550889045e38  # one no-data pixel near a corner
    data_path = directory / "TEST.TIF"
    data_path.write_bytes(b"\0" * header_bytes + heights.tobytes())
    label_path = directory / "TEST.LBL"
    label_path.write_text(
        LABEL_TEXT.format(
            offset=header_bytes + 1, line_offset=line_offset, sample_offset=sample_offset, lines=lines, samples=samples
        )
    )
    return label_path, data_path


def test_height_map_reader_on_a_synthetic_product(tmp_path):
    """The reader finds the data, maps local metres to pixels, and refuses no-data and missing files."""
    label_path, data_path = write_test_product(tmp_path, latitude=20.19091, longitude=30.77168)
    label = parse_planetary_data_system_label(label_path.read_text())
    assert label["IMAGE.LINES"] == "40" and "two lines" in label["IMAGE.DESCRIPTION"]
    height_map = LunarHeightMap(label_path, data_path, 20.19091, 30.77168)
    east = numpy.array([0.0, 10.0, -6.0])
    north = numpy.array([0.0, -4.0, 8.0])
    relative = height_map.sample_relative(east, north)
    scale_x = math.cos(math.radians(20.0)) / math.cos(math.radians(20.19091))
    assert relative == pytest.approx(0.01 * east * scale_x + 0.02 * north, abs=1e-3)
    assert height_map.sample_absolute(numpy.zeros(1), numpy.zeros(1))[0] == pytest.approx(-1000.0, abs=1e-3)
    with pytest.raises(HeightMapError, match="no-data"):
        height_map.sample_absolute(numpy.array([(2.3 - 25.0) * 2.0 / scale_x]), numpy.array([(20.0 - 2.3) * 2.0]))
    with pytest.raises(HeightMapError, match="outside"):
        height_map.sample_absolute(numpy.array([1000.0]), numpy.array([0.0]))
    with pytest.raises(HeightMapError, match="download_height_maps"):
        LunarHeightMap(tmp_path / "missing.LBL", data_path, 20.0, 30.0)


def test_apollo17_surface_config_builds_the_populations():
    """The surface config parses into the published laws; the scenario composes and passes the loop check."""
    import yaml

    from lunarsim.env.scenario_loader import default_config_directory, load_scenario
    from lunarsim.slots.surface.measured_terrain import crater_population_from_parameters, rock_population_from_parameters

    parameters = yaml.safe_load((default_config_directory() / "surface" / "apollo17_taurus_littrow.yaml").read_text())
    craters = crater_population_from_parameters(parameters["craters"])
    rocks = rock_population_from_parameters(parameters["rocks"])
    assert craters.cumulative_coefficient_per_m2 == 0.0336 and craters.classes_without_rim == (4,)
    assert sum(craters.class_shares) == pytest.approx(1.0)
    assert rocks.cumulative_exponent == -2.11 and rocks.maximum_diameter_m == 2.0
    assert crater_population_from_parameters({**parameters["craters"], "enabled": False}) is None
    scenario = load_scenario("apollo17_terminal_truth", ["surface.rocks.enabled=false"])
    assert scenario.slots["surface"].binding == "measured_terrain"
    assert scenario.slots["surface"].parameters["rocks"]["enabled"] is False
