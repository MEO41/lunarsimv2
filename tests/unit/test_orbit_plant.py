# SPDX-License-Identifier: BSD-3-Clause
"""Orbit plant (milestone 6): the design's test "two-body energy and angular momentum stay constant;
circular period = 2 pi sqrt(a^3 / mu)" (CPU, torch float64)."""

import math

import torch
from helpers import make_binding_context

from lunarsim.assets.lander_preset import lander_preset_from_dictionary
from lunarsim.core.frames import convert_truth_state
from lunarsim.core.message_types import Frame, MessageKey, SurfaceGeometry, TruthState, Wrench
from lunarsim.core.physical_constants import MOON_GRAVITATIONAL_PARAMETER, MOON_MEAN_RADIUS
from lunarsim.env.scenario_loader import load_scenario
from lunarsim.slots.plant.orbit_float64 import OrbitFloat64Plant


class _World:
    """Test-side writer of the start state, the wrenches and the surface geometry; reader of the truth."""

    provides = frozenset({MessageKey.START_STATE, MessageKey.ACTUATOR_WRENCH, MessageKey.SURFACE_WRENCH, MessageKey.SURFACE_GEOMETRY})
    requires = frozenset({MessageKey.TRUTH_STATE})
    reads_truth = True


def make_plant(frame: str, count: int = 2):
    """Return (plant, bus, writer) with the Apollo LM preset and zero wrenches."""
    context = make_binding_context(count)
    context.shared_data["lander_preset"] = lander_preset_from_dictionary(load_scenario("s1_terminal_truth", []).lander)
    plant = OrbitFloat64Plant({"frame": frame}, context)
    world, bus = _World(), context.bus
    zero = Wrench.zeros(count)
    bus.write(MessageKey.ACTUATOR_WRENCH, zero, world)
    bus.write(MessageKey.SURFACE_WRENCH, zero, world)
    bus.write(MessageKey.SURFACE_GEOMETRY, SurfaceGeometry(site_origin=torch.zeros(count, 3)), world)
    return plant, bus, world


def circular_start(radius: float, count: int = 2) -> TruthState:
    """Return MCI circular-orbit start states in the x-y plane, float64."""
    speed = math.sqrt(MOON_GRAVITATIONAL_PARAMETER / radius)
    dtype = torch.float64
    return TruthState(
        frame=Frame.MOON_CENTERED_INERTIAL, time=torch.zeros(count, dtype=dtype),
        position=torch.tensor([[radius, 0.0, 0.0]] * count, dtype=dtype), velocity=torch.tensor([[0.0, speed, 0.0]] * count, dtype=dtype),
        orientation_quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count, dtype=dtype),
        angular_velocity=torch.tensor([[0.0, 0.0, 0.01]] * count, dtype=dtype), mass=torch.full((count,), 14000.0, dtype=dtype),
        contact=torch.zeros(count, 0, dtype=dtype),
    )


def run(plant, bus, world, seconds: float, dt: float) -> TruthState:
    """Integrate ``seconds`` with zero thrust and return the truth."""
    for _ in range(int(round(seconds / dt))):
        plant.apply_wrench(bus, dt)
    plant.step(bus, dt)
    return bus.read(MessageKey.TRUTH_STATE, world)


def test_circular_orbit_closes_after_one_period_and_keeps_energy_and_momentum():
    """After 2 pi sqrt(a^3/mu) the lander is back at the start; energy and angular momentum hold to 1e-9."""
    radius = MOON_MEAN_RADIUS + 111120.0
    plant, bus, world = make_plant("MCI")
    start = circular_start(radius)
    bus.write(MessageKey.START_STATE, start, world)
    plant.reset(torch.arange(2), torch.Generator())
    period = 2.0 * math.pi * math.sqrt(radius**3 / MOON_GRAVITATIONAL_PARAMETER)
    steps = 2000
    truth = run(plant, bus, world, period, period / steps)
    assert float((truth.position - start.position).norm(dim=1).max()) < 1.0  # metres after 7,100 s
    energy = 0.5 * (truth.velocity**2).sum(dim=1) - MOON_GRAVITATIONAL_PARAMETER / truth.position.norm(dim=1)
    energy0 = 0.5 * (start.velocity**2).sum(dim=1) - MOON_GRAVITATIONAL_PARAMETER / start.position.norm(dim=1)
    assert float(((energy - energy0) / energy0).abs().max()) < 1e-9
    momentum = torch.cross(truth.position, truth.velocity, dim=1)
    momentum0 = torch.cross(start.position, start.velocity, dim=1)
    assert float(((momentum - momentum0).norm(dim=1) / momentum0.norm(dim=1)).max()) < 1e-9
    # A free spin about a principal axis stays constant (no torque).
    assert torch.allclose(truth.angular_velocity, start.angular_velocity, atol=1e-12)


def test_moon_fixed_frame_gives_the_same_trajectory_as_the_inertial_frame():
    """Integrated in MCMF (with Coriolis and centrifugal terms), the orbit matches the MCI one."""
    radius = MOON_MEAN_RADIUS + 15240.0
    inertial_plant, inertial_bus, inertial_world = make_plant("MCI")
    fixed_plant, fixed_bus, fixed_world = make_plant("MCMF")
    start = circular_start(radius)
    inertial_bus.write(MessageKey.START_STATE, start, inertial_world)
    fixed_bus.write(MessageKey.START_STATE, start, fixed_world)
    inertial_plant.reset(torch.arange(2), torch.Generator())
    fixed_plant.reset(torch.arange(2), torch.Generator())
    a = run(inertial_plant, inertial_bus, inertial_world, 600.0, 0.5)
    b = convert_truth_state(run(fixed_plant, fixed_bus, fixed_world, 600.0, 0.5), Frame.MOON_CENTERED_INERTIAL)
    assert float((a.position - b.position).norm(dim=1).max()) < 1e-3
    assert float((a.velocity - b.velocity).norm(dim=1).max()) < 1e-6


def test_thrust_along_the_velocity_raises_the_orbit_and_burns_mass():
    """A prograde force adds energy, and a mass flow lowers the mass."""
    radius = MOON_MEAN_RADIUS + 111120.0
    plant, bus, world = make_plant("MCI", count=1)
    start = circular_start(radius, count=1)
    # Body z along +y (the velocity): rotate +90 deg about -x.
    start.orientation_quaternion = torch.tensor([[math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0]], dtype=torch.float64)
    bus.write(MessageKey.START_STATE, start, world)
    plant.reset(torch.arange(1), torch.Generator())
    bus.write(MessageKey.ACTUATOR_WRENCH, Wrench(force=torch.tensor([[0.0, 0.0, 4.0e4]]), torque=torch.zeros(1, 3), mass_flow_rate=torch.tensor([13.0])), world)
    truth = run(plant, bus, world, 10.0, 0.1)
    energy = 0.5 * (truth.velocity**2).sum(dim=1) - MOON_GRAVITATIONAL_PARAMETER / truth.position.norm(dim=1)
    energy0 = 0.5 * (start.velocity**2).sum(dim=1) - MOON_GRAVITATIONAL_PARAMETER / start.position.norm(dim=1)
    assert float(energy - energy0) > 0.0
    assert abs(float(truth.mass[0]) - (14000.0 - 130.0)) < 1e-6
