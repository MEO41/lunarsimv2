# SPDX-License-Identifier: BSD-3-Clause
"""L2 plume grains (CPU, milestone 5): size bins, gas-speed fractions, ballistic flight, mass bookkeeping, the particle cap."""

import math
from pathlib import Path

import torch
import yaml

from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.terrain.plume_grain_model import PlumeGrainModel, gas_speed_fractions, size_bins

PARAMETERS = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "configs" / "surface" / "regolith_plume_l2.yaml"))["grains"]
HOVER_THRUST = 11869.7
EXHAUST_VELOCITY = 305.0 * 9.80665


def flat_ground(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return a flat ground at height zero."""
    return torch.zeros_like(x)


def test_size_bins_are_octaves_whose_mass_fractions_sum_to_one():
    """Seven octave bins from 4 to 512 microns; mass fractions positive and summing to one."""
    diameters, fractions = size_bins(PARAMETERS)
    assert diameters.shape[0] == 7
    assert abs(float(fractions.sum()) - 1.0) < 1e-5 and bool((fractions >= 0).all())
    assert bool((diameters[1:] / diameters[:-1] - 2.0).abs().max() < 1e-4)


def test_small_grains_reach_more_of_the_gas_speed():
    """alpha falls with the diameter and stays in (0, 1]."""
    diameters, _ = size_bins(PARAMETERS)
    alpha = gas_speed_fractions(diameters, PARAMETERS)
    assert bool((alpha[1:] <= alpha[:-1] + 1e-6).all())
    assert float(alpha.min()) > 0.0 and float(alpha.max()) <= 1.0 + 1e-6


def test_a_grain_flies_a_ballistic_arc():
    """Without thrust nothing erodes; a placed grain moves as x = x0 + v t, z = z0 + vz t - g t^2 / 2."""
    model = PlumeGrainModel(PARAMETERS, 1, "cpu")
    model.position[0, 0] = torch.tensor([1.0, 2.0, 5.0])
    model.velocity[0, 0] = torch.tensor([30.0, -10.0, 4.0])
    model.alive[0, 0] = True
    model.mass[0, 0] = 1e-3
    nozzle = torch.tensor([[0.0, 0.0, 10.0]])
    dt, steps = 0.02, 25
    for _ in range(steps):
        state = model.step(nozzle, torch.zeros(1), torch.zeros(1), EXHAUST_VELOCITY, flat_ground, dt)
    time = dt * steps
    expected = torch.tensor([1.0 + 30.0 * time, 2.0 - 10.0 * time, 5.0 + 4.0 * time - 0.5 * MOON_SURFACE_GRAVITY * time**2])
    assert torch.allclose(model.position[0, 0], expected, atol=1e-4)
    assert float(state.eroded_mass[0]) == 0.0 and not bool(state.eroding[0])


def test_eroded_mass_is_carried_by_the_new_particles():
    """At hover thrust 5 m up, the mass on the particles launched in one step equals the eroded mass."""
    model = PlumeGrainModel(PARAMETERS, 2, "cpu")
    nozzle = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 12.0]])
    state = model.step(nozzle, torch.zeros(2), torch.full((2,), HOVER_THRUST), EXHAUST_VELOCITY, flat_ground, 0.02)
    carried = torch.where(model.alive, model.mass, 0.0).sum(dim=1)
    assert bool((state.eroded_mass > 0).all())
    assert torch.allclose(carried, state.eroded_mass, rtol=1e-4)
    assert torch.allclose(state.ejecta_mass_rate * 0.02, state.eroded_mass, rtol=1e-5)


def test_the_particle_cap_reuses_the_oldest_particles():
    """With room for two steps of launches, the count stays at the cap and the youngest particles are kept."""
    parameters = dict(PARAMETERS, particles_per_lander=56)  # 7 bins x 4 per step = 28 per step
    model = PlumeGrainModel(parameters, 1, "cpu")
    nozzle = torch.tensor([[0.0, 0.0, 30.0]])
    for _ in range(5):
        state = model.step(nozzle, torch.zeros(1), torch.full((1,), HOVER_THRUST), EXHAUST_VELOCITY, flat_ground, 0.02)
    assert int(state.particles[0]) <= 56
    assert float(model.age[0][model.alive[0]].max()) < 2 * 0.02 + 1e-6


def test_the_layer_coefficient_gives_the_reference_gas_density():
    """The fitted layer thickness gives 2e-5 kg/m^3 at the reference point of the Apollo-like CFD (3.8 m out, nozzle 3.05 m up, hover)."""
    mass_flow = HOVER_THRUST / EXHAUST_VELOCITY
    radius, height = 3.7762, 3.05
    thickness = PARAMETERS["layer_thickness_coefficient"] * math.hypot(radius, height)
    density = mass_flow / (2.0 * math.pi * radius * thickness * PARAMETERS["gas_speed_mps"])
    assert abs(density / PARAMETERS["reference_gas_density_kg_per_m3"] - 1.0) < 0.02
