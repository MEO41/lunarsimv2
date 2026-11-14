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
    height = preset.body_center_of_mass_height
    first_moment = preset.body_mass * height + sum(preset.leg_mass * z for _, _, z in preset.leg_positions)
    assert first_moment == pytest.approx(0.0, abs=1e-6)
    assert 0.05 < height < 0.15
    offset_term = preset.body_mass * height**2
    about_x = preset.body_inertia[0] + offset_term + sum(preset.leg_mass * (y * y + z * z) for _, y, z in preset.leg_positions)
    about_y = preset.body_inertia[1] + offset_term + sum(preset.leg_mass * (x * x + z * z) for x, _, z in preset.leg_positions)
    about_z = preset.body_inertia[2] + sum(preset.leg_mass * (x * x + y * y) for x, y, _ in preset.leg_positions)
    assert about_x == pytest.approx(preset.inertia_about_x)
    assert about_y == pytest.approx(preset.inertia_about_y)
    assert about_z == pytest.approx(preset.inertia_about_z)


def test_jets_form_balanced_clusters(preset):
    """16 jets; all jets together give no net force and no net torque; couples exist for each axis."""
    assert preset.number_of_jets == 16
    force = preset.jet_force_matrix()
    torque = preset.jet_torque_matrix()
    for axis in range(3):
        assert sum(force[axis]) == pytest.approx(0.0, abs=1e-6)
        assert sum(torque[axis]) == pytest.approx(0.0, abs=1e-6)
        assert max(torque[axis]) > 400.0 and min(torque[axis]) < -400.0
    # A vertical jet at a 45 degree cluster has a pitch arm of 1.676 / sqrt(2).
    assert max(torque[0]) == pytest.approx(444.8 * 1.676 / math.sqrt(2.0), rel=1e-3)
    # A horizontal jet turns the vehicle about z with the same arm.
    assert max(torque[2]) == pytest.approx(444.8 * 1.676 / math.sqrt(2.0), rel=1e-3)


def test_usd_text_contains_body_legs_and_struts(preset):
    """The generated USD has one articulation root, 4 leg links and 4 prismatic joints with drives."""
    text = build_lander_usda(preset)
    assert text.count("PhysicsArticulationRootAPI") == 1
    assert text.count("def PhysicsPrismaticJoint") == 4
    assert text.count('def Sphere "footpad"') == 4
    assert f"drive:linear:physics:stiffness = {preset.strut_stiffness:.6f}" in text
    assert f"physics:upperLimit = {preset.strut_stroke:.6f}" in text
