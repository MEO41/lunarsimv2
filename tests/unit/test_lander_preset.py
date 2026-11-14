# SPDX-License-Identifier: BSD-3-Clause
"""Lander preset: loads, every number has a source, derived values are consistent."""

import math

import pytest
import yaml

from lunarsim.assets.lander_builder import build_lander_usda
from lunarsim.assets.lander_preset import (
    LanderPresetError,
    lander_preset_from_dictionary,
    lander_preset_sources,
    load_lander_preset,
)
from lunarsim.env.scenario_loader import default_config_directory

PRESET_PATH = default_config_directory() / "landers" / "apollo_lunar_module.yaml"


@pytest.fixture(scope="module")
def preset():
    """Load the Apollo Lunar Module preset once."""
    return load_lander_preset(PRESET_PATH)


def test_every_number_has_a_source_and_none_is_pending():
    """Hard rule of the project: each constant names its public source."""
    data = yaml.safe_load(PRESET_PATH.read_text())
    sources = lander_preset_sources(data)
    assert len(sources) >= 30
    for key, source in sources.items():
        assert len(source) > 15, f"{key} has no real source text"
        assert "PENDING" not in source, f"{key} is still pending verification"


def test_missing_source_is_rejected():
    """An entry without a source cannot be loaded."""
    data = yaml.safe_load(PRESET_PATH.read_text())
    data["mass"]["mass_at_start_kg"] = {"value": 7000.0}
    with pytest.raises(LanderPresetError, match="mass_at_start_kg"):
        lander_preset_from_dictionary(data)


def test_units_and_derived_values(preset):
    """Spot checks in SI units against hand calculations."""
    assert preset.mass_at_start == pytest.approx(16153.2 * 0.45359237, rel=1e-4)
    assert preset.engine_maximum_thrust == pytest.approx(0.65 * 10500 * 4.4482216, rel=1e-4)
    assert preset.maximum_touchdown_tilt == pytest.approx(math.radians(6.0))
    assert preset.usable_descent_propellant_at_start == pytest.approx(349.3 - 43.5)
    assert preset.strut_damping == pytest.approx(2.0 * math.sqrt(78800.0 * 7327.0 / 4.0))
    # The lunar weight must lie inside the throttle range, or hover is impossible.
    lunar_weight = preset.mass_at_start * 1.6242
    assert preset.engine_minimum_thrust < lunar_weight < preset.engine_maximum_thrust


def test_body_plus_legs_reproduce_total_mass_inertia_and_centre_of_mass(preset):
    """Root body + point-mass legs give back the preset's totals about the body-frame origin."""
    assert preset.body_mass + preset.number_of_legs * preset.leg_mass == pytest.approx(preset.mass_at_start)
    body_x, body_y, body_z = preset.body_center_of_mass
    for axis, body_value in enumerate((body_x, body_y, body_z)):
        first_moment = preset.body_mass * body_value + sum(preset.leg_mass * position[axis] for position in preset.leg_positions)
        assert first_moment == pytest.approx(0.0, abs=1e-6)
    assert 0.05 < body_z < 0.15
    legs = preset.leg_positions
    about_x = preset.body_inertia[0] + preset.body_mass * (body_y**2 + body_z**2) + sum(preset.leg_mass * (y * y + z * z) for _, y, z in legs)
    about_y = preset.body_inertia[1] + preset.body_mass * (body_x**2 + body_z**2) + sum(preset.leg_mass * (x * x + z * z) for x, _, z in legs)
    about_z = preset.body_inertia[2] + preset.body_mass * (body_x**2 + body_y**2) + sum(preset.leg_mass * (x * x + y * y) for x, y, _ in legs)
    assert about_x == pytest.approx(preset.inertia_about_x)
    assert about_y == pytest.approx(preset.inertia_about_y)
    assert about_z == pytest.approx(preset.inertia_about_z)


def test_geometry_follows_the_handbook_and_the_centre_of_mass_offset(preset):
    """Stages stack without a gap, the top is 6.011 m above the pads, everything sits on the shifted axis."""
    descent_top = preset.descent_stage_center[2] + 0.5 * preset.descent_stage_height
    ascent_bottom = preset.ascent_stage_center[2] - 0.5 * preset.ascent_stage_height
    assert descent_top == pytest.approx(ascent_bottom)
    top_above_pads = preset.ascent_stage_center[2] + 0.5 * preset.ascent_stage_height + preset.center_of_mass_height_above_footpad_plane
    assert top_above_pads == pytest.approx(6.011, abs=0.002)
    axis_x, axis_y = preset.vehicle_axis_position
    assert (axis_x, axis_y) == pytest.approx((-0.0406, 0.0102))
    centre_x = sum(x for x, _, _ in preset.leg_positions) / preset.number_of_legs
    centre_y = sum(y for _, y, _ in preset.leg_positions) / preset.number_of_legs
    assert (centre_x, centre_y) == pytest.approx((axis_x, axis_y))
    # The lowest point of each dish lies in the footpad plane.
    assert preset.leg_positions[0][2] - preset.footpad_depth == pytest.approx(-preset.center_of_mass_height_above_footpad_plane)
    assert preset.footpad_sphere_radius == pytest.approx((0.47**2 + 0.178**2) / (2 * 0.178))


def test_jets_form_balanced_clusters(preset):
    """16 jets; with the c.g. offset removed all jets give no net force or torque; R-567 arms."""
    assert preset.number_of_jets == 16
    force = preset.jet_force_matrix()
    for axis in range(3):
        assert sum(force[axis]) == pytest.approx(0.0, abs=1e-6)
    torque = preset.jet_torque_matrix()
    for axis in range(3):
        assert max(torque[axis]) > 400.0 and min(torque[axis]) < -400.0
    # A vertical jet turns the vehicle about x with its y arm: 1.679 m from the axis, plus the c.g. offset.
    axis_x, axis_y = preset.vehicle_axis_position
    assert max(torque[0]) == pytest.approx(444.8 * (1.679 + abs(axis_y)), rel=1e-3)
    # A horizontal jet turns the vehicle about z with its arm across the firing axis (1.562 m).
    assert max(abs(value) for value in torque[2]) == pytest.approx(444.8 * (1.562 + 0.0406), rel=2e-2)


def test_usd_text_contains_body_legs_and_struts(preset):
    """The generated USD has one articulation root, 4 dish footpads, 4 prismatic joints, two stage shapes."""
    text = build_lander_usda(preset)
    assert text.count("PhysicsArticulationRootAPI") == 1
    assert text.count("def PhysicsPrismaticJoint") == 4
    assert text.count('def Mesh "footpad"') == 4
    assert text.count('physics:approximation = "convexHull"') == 4
    assert 'def Cylinder "descent_stage"' in text and 'def Cube "ascent_stage"' in text
    assert f"drive:linear:physics:stiffness = {preset.strut_stiffness:.6f}" in text
    assert f"physics:upperLimit = {preset.strut_stroke:.6f}" in text


def test_footpad_dish_mesh_has_the_real_size(preset):
    """The dish mesh is 0.94 m across, 0.178 m deep, and small enough for a PhysX convex hull."""
    from lunarsim.assets.lander_builder import footpad_dish_mesh

    points, counts, indices = footpad_dish_mesh(preset)
    radii = [math.hypot(x, y) for x, y, _ in points]
    heights = [z for _, _, z in points]
    assert max(radii) == pytest.approx(0.47, abs=1e-6)
    assert min(heights) == pytest.approx(-0.178, abs=1e-6)
    assert max(heights) == pytest.approx(0.0, abs=1e-6)
    assert len(points) < 255
    assert sum(counts) == len(indices) and max(indices) == len(points) - 1
