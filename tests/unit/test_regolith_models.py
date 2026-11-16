# SPDX-License-Identifier: BSD-3-Clause
"""Bekker footpad soil and the L1 plume model (milestone 5)."""

import math

import torch

from lunarsim.core.message_types import SurfaceGeometry
from lunarsim.terrain.regolith_models import PSI_PER_INCH, BekkerSoil, FootpadSoilContact, PlumeModel

INCH = 0.0254
PAD_RADIUS = 0.5 * 37.0 * INCH
PAD_DEPTH = 7.0 * INCH
SPHERE_RADIUS = (PAD_RADIUS**2 + PAD_DEPTH**2) / (2.0 * PAD_DEPTH)


def contact(soil: BekkerSoil | None = None) -> FootpadSoilContact:
    """Return the soil contact of one environment with one LM footpad."""
    return FootpadSoilContact(soil or BekkerSoil(), PAD_RADIUS, SPHERE_RADIUS, PAD_DEPTH, 1, 1, "cpu")


def test_linear_soil_force_matches_the_dish_formula():
    """For n = 1 the dish force is close to k pi R z^2 (exact for a paraboloid; a sphere sits a little higher)."""
    pads = contact()
    for sinkage in (0.01, 0.03, 0.08):
        expected = BekkerSoil().modulus * math.pi * SPHERE_RADIUS * sinkage**2
        assert abs(float(pads.loading_force(torch.tensor([sinkage]))) - expected) / expected < 0.05


def test_static_sinkage_under_a_quarter_of_the_lm_weight_is_about_four_centimetres():
    """One pad holding 7,327 kg * 1.62 / 4 sinks about 4.4 cm in the Apollo 11 soil (2.5 psi/in)."""
    load = 7327.0 * 1.62 / 4.0
    pads = contact()
    depths = torch.linspace(0.0, 0.1, 2001)
    sinkage = float(depths[torch.searchsorted(pads.loading_force(depths), torch.tensor(load))])
    assert 0.04 < sinkage < 0.05


def test_pad_pressed_then_lifted_does_not_spring_back():
    """After 3 cm of sinkage, lifting the pad 1 cm leaves no force; pressing back to 3 cm restores it."""
    pads = contact()
    geometry = SurfaceGeometry(site_origin=torch.zeros(1, 3))
    still = torch.zeros(1, 1, 3)
    loaded, _ = pads.forces(torch.tensor([[[0.0, 0.0, -0.03]]]), still, geometry)
    lifted, sinkage = pads.forces(torch.tensor([[[0.0, 0.0, -0.02]]]), still, geometry)
    again, _ = pads.forces(torch.tensor([[[0.0, 0.0, -0.03]]]), still, geometry)
    assert float(loaded[0, 0, 2]) > 1000.0
    assert float(lifted[0, 0, 2]) == 0.0 and abs(float(sinkage[0, 0]) - 0.02) < 1e-6
    assert abs(float(again[0, 0, 2]) - float(loaded[0, 0, 2])) < 1e-3


def test_friction_opposes_sliding_and_is_bounded_by_the_coefficient():
    """A sliding pressed pad gets a sideways force against its motion, at most 0.4 of the normal force."""
    pads = contact()
    geometry = SurfaceGeometry(site_origin=torch.zeros(1, 3))
    force, _ = pads.forces(torch.tensor([[[0.0, 0.0, -0.03]]]), torch.tensor([[[0.5, 0.0, 0.0]]]), geometry)
    assert float(force[0, 0, 0]) < 0.0
    assert abs(float(force[0, 0, 0])) <= 0.4 * float(force[0, 0, 2]) + 1e-6


def test_design_soil_preset_passes_through_its_two_points():
    """The LM design soil gives 1 psi at 4 in and 12 psi at 24 in."""
    soil = BekkerSoil.from_parameters({"preset": "lm_design"})
    for depth_in, pressure_psi in ((4.0, 1.0), (24.0, 12.0)):
        pressure = soil.modulus * (depth_in * INCH) ** soil.exponent
        assert abs(pressure / 6894.757 - pressure_psi) < 1e-6


def test_plume_dust_starts_below_100_feet_and_saturates_near_the_ground():
    """No dust above 30.5 m, full dust at 3 m with hover thrust; the force decays with height."""
    plume = PlumeModel()
    thrust = torch.full((3,), plume.reference_thrust_n)
    heights = torch.tensor([40.0, 3.0, 16.0])
    nozzle = torch.stack((torch.zeros(3), torch.zeros(3), heights), dim=1)
    down = torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1)
    result = plume.evaluate(nozzle, down, thrust, 1.5, nozzle + torch.tensor([0.0, 0.0, 1.0]), SurfaceGeometry(site_origin=torch.zeros(3, 3)))
    assert float(result["dust_density"][0]) == 0.0 and float(result["dust_density"][1]) == 1.0
    assert 0.0 < float(result["dust_density"][2]) < 1.0
    assert float(result["force"][1, 2]) > float(result["force"][2, 2]) > float(result["force"][0, 2]) >= 0.0
    # Upright over flat ground the force passes through the centre of mass: no torque.
    assert float(result["torque"].abs().max()) < 1e-6


def test_tilted_plume_gives_a_torque_and_disabled_plume_gives_nothing():
    """A tilted thrust axis moves the centre of pressure sideways; enabled false returns zeros."""
    plume = PlumeModel()
    nozzle = torch.tensor([[0.0, 0.0, 2.0]])
    tilted = torch.nn.functional.normalize(torch.tensor([[0.2, 0.0, -1.0]]), dim=1)
    result = plume.evaluate(nozzle, tilted, torch.tensor([plume.reference_thrust_n]), 1.5, torch.tensor([[0.0, 0.0, 3.0]]), SurfaceGeometry(site_origin=torch.zeros(1, 3)))
    assert abs(float(result["torque"][0, 1])) > 0.0
    off = PlumeModel.from_parameters({"enabled": False}).evaluate(nozzle, tilted, torch.tensor([1.0e4]), 1.5, nozzle, SurfaceGeometry(site_origin=torch.zeros(1, 3)))
    assert float(off["force"].abs().sum()) == 0.0 and float(off["dust_density"][0]) == 0.0
