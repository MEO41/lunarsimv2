# SPDX-License-Identifier: BSD-3-Clause
"""Physics validation of the analytic plant (CPU), test T1 of the analytic-to-PhysX study.

The known answers and tolerances are those of the PhysX validation tests in ``tests/physics``
(free fall, hover, reaction-control torque, contact rest), so the two simulators pass the same
checks. Where the analytic model is exact by construction the tolerance is tighter, and the
docstring says why.
"""

import math

import pytest
import torch
from analytic_environment_support import make_actions, make_environment, read_truth, set_start_states

from lunarsim.agents.classical.jet_selection import JetCoupleTable
from lunarsim.core.binding import SlotName
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.core.rotation_math import quaternion_from_axis_angle, tilt_angle_from_vertical

# ---- free fall ----

START_ALTITUDE = 50.0
FALL_TIME = 2.0


def test_free_fall_follows_the_semi_implicit_euler_parabola():
    """z(t) = z0 - g t^2 / 2 - g t dt / 2 within 1 mm, and v = -g t within 1 mm/s (as the PhysX test).

    The plant integrates like PhysX's PGS solver (speed first, then position with the new
    speed), so it falls g t dt / 2 further than the exact parabola: 8.1 mm at 2 s.
    """
    environment = make_environment("s1_terminal_truth", ["num_envs=2", "actuators.initial_throttle=0.0", "metrics=none"])
    set_start_states(environment, altitude=START_ALTITUDE)
    start_height = float(read_truth(environment).position[0, 2])
    step = environment.clock.control_step_duration
    for _ in range(round(FALL_TIME / step)):
        environment.step(make_actions(environment))
    truth = read_truth(environment)
    dt = environment.clock.physics_step_duration
    g = MOON_SURFACE_GRAVITY
    expected = start_height - 0.5 * g * FALL_TIME**2 - 0.5 * g * FALL_TIME * dt
    assert float(truth.position[0, 2]) == pytest.approx(expected, abs=1e-3)
    assert abs(float(truth.velocity[0, 2]) + g * FALL_TIME) < 1e-3
    assert float(truth.contact.abs().max()) == 0.0


# ---- hover while the mass drops ----

HOVER_TIME = 20.0


@pytest.fixture(scope="module")
def hover_result():
    """Hover with thrust = m g for 20 s (environment 0) and hold the start-mass thrust (environment 1)."""
    environment = make_environment("s1_terminal_truth", ["num_envs=2", "metrics=none"])
    preset = environment.bindings[SlotName.PLANT].preset
    observations = set_start_states(environment, altitude=30.0)
    start = read_truth(environment)
    start_height, start_mass = start.position[:, 2].clone(), start.mass.clone()
    expected_burn = 0.0
    for _ in range(int(HOVER_TIME / environment.clock.control_step_duration)):
        mass = observations["policy"][:, 13].clone()
        mass[1] = start_mass[1]
        throttle = mass * MOON_SURFACE_GRAVITY / preset.engine_rated_thrust
        expected_burn += float(throttle[0]) * preset.engine_rated_thrust / preset.engine_exhaust_velocity * 0.02
        observations, _, _, _, _ = environment.step(make_actions(environment, throttle=throttle))
    return start_height, start_mass, read_truth(environment), expected_burn, preset


def test_hover_holds_altitude(hover_result):
    """With thrust = m g the altitude drifts less than 2 cm and the speed stays below 5 mm/s."""
    start_height, _, end, _, _ = hover_result
    assert abs(float(end.position[0, 2] - start_height[0])) < 0.02
    assert abs(float(end.velocity[0, 2])) < 0.005


def test_mass_drops_by_thrust_over_exhaust_velocity(hover_result):
    """The truth mass falls by the integral of thrust / (Isp g0), within 0.1 %."""
    _, start_mass, end, expected_burn, _ = hover_result
    burned = float(start_mass[0] - end.mass[0])
    assert burned == pytest.approx(expected_burn, rel=1e-3)
    assert burned > 70.0


def test_constant_thrust_makes_the_lighter_lander_rise(hover_result):
    """Control case: thrust for the start mass lifts the lander by g mdot t^3 / (6 m0), within 5 %."""
    start_height, start_mass, end, _, preset = hover_result
    mass_flow = float(start_mass[1]) * MOON_SURFACE_GRAVITY / preset.engine_exhaust_velocity
    expected_rise = MOON_SURFACE_GRAVITY * mass_flow * HOVER_TIME**3 / (6.0 * float(start_mass[1]))
    assert float(end.position[1, 2] - start_height[1]) == pytest.approx(expected_rise, rel=0.05)


# ---- reaction-control torque ----

FIRING_TIME = 1.0
COAST_TIME = 0.1


@pytest.fixture(scope="module")
def spin_result():
    """Fire the six jet couples (plus and minus about each body axis) for 1 s, then coast 0.1 s."""
    environment = make_environment("s1_terminal_truth", ["num_envs=6", "actuators.initial_throttle=0.0", "metrics=none"])
    preset = environment.bindings[SlotName.PLANT].preset
    table = JetCoupleTable(preset, device=environment.device)
    axis_commands = torch.zeros(6, 3)
    for axis in range(3):
        axis_commands[2 * axis, axis] = 1.0
        axis_commands[2 * axis + 1, axis] = -1.0
    jets = table.jets_for_axis_commands(axis_commands)
    set_start_states(environment, altitude=50.0)
    step = environment.clock.control_step_duration
    for _ in range(round(FIRING_TIME / step)):
        environment.step(make_actions(environment, jets=jets))
    for _ in range(round(COAST_TIME / step)):
        environment.step(make_actions(environment))
    return read_truth(environment), table, preset


def test_body_rate_is_torque_times_time_over_inertia(spin_result):
    """Each couple spins the lander about its own axis at torque * t / I, within 0.5 %."""
    truth, table, preset = spin_result
    inertia = (preset.inertia_about_x, preset.inertia_about_y, preset.inertia_about_z)
    for axis in range(3):
        expected = float(table.couple_torque[axis]) * FIRING_TIME / inertia[axis]
        assert float(truth.angular_velocity[2 * axis, axis]) == pytest.approx(expected, rel=5e-3)
        assert float(truth.angular_velocity[2 * axis + 1, axis]) == pytest.approx(-expected, rel=5e-3)


def test_no_cross_axis_rate_and_no_net_force(spin_result):
    """The other two body rates stay below 0.5 % of the main rate; the velocity is pure free fall."""
    truth, _, _ = spin_result
    for axis in range(3):
        for row in (2 * axis, 2 * axis + 1):
            main_rate = abs(float(truth.angular_velocity[row, axis]))
            others = [other for other in range(3) if other != axis]
            assert float(truth.angular_velocity[row, others].abs().max()) < 5e-3 * main_rate
    fall_speed = MOON_SURFACE_GRAVITY * (FIRING_TIME + COAST_TIME)
    assert float((truth.velocity[:, 2] + fall_speed).abs().max()) < 0.002
    assert float(truth.velocity[:, :2].abs().max()) < 0.002


# ---- contact rest ----

GROUP_SIZE = 4
GENTLE, HARD, SLIDING, TILTED = (slice(index * GROUP_SIZE, (index + 1) * GROUP_SIZE) for index in range(4))
SETTLE_TIME = 12.0


@pytest.fixture(scope="module")
def touchdown():
    """Drop the four groups of the PhysX contact-rest test (gentle, hard, sliding, tilted) and record 12 s."""
    count = 4 * GROUP_SIZE
    environment = make_environment("s1_terminal_truth", [f"num_envs={count}", "actuators.initial_throttle=0.0", "metrics=none"])
    plant = environment.bindings[SlotName.PLANT]
    preset = plant.preset
    altitude = torch.full((count,), 0.05)
    altitude[GENTLE] = 0.3
    altitude[TILTED] = 0.05 + preset.footpad_radius_from_thrust_axis * math.sin(math.radians(6.0))
    vertical_speed = torch.zeros(count)
    vertical_speed[HARD] = -3.0
    vertical_speed[SLIDING] = -2.1
    vertical_speed[TILTED] = -2.1
    set_start_states(environment, altitude=altitude, vertical_speed=vertical_speed)
    start = environment.bindings[SlotName.START]
    position, velocity, orientation = start.fixed_start_states
    velocity[SLIDING, 0] = 1.2
    tilt_axis = torch.tensor([0.0, 1.0, 0.0]).repeat(GROUP_SIZE, 1)
    orientation[TILTED] = quaternion_from_axis_angle(tilt_axis, torch.full((GROUP_SIZE,), math.radians(6.0)))
    start.set_fixed_start_states(position, velocity, orientation)
    environment.reset()

    history = {"position": [], "speed": [], "rate": [], "leg_forces": [], "strut": [], "tilt": []}
    for _ in range(int(SETTLE_TIME / environment.clock.control_step_duration)):
        environment.step(make_actions(environment, throttle=0.0))
        truth = read_truth(environment)
        history["position"].append(truth.position.clone())
        history["speed"].append(truth.velocity.norm(dim=1))
        history["rate"].append(truth.angular_velocity.norm(dim=1))
        history["leg_forces"].append(truth.contact.clone())
        history["strut"].append(plant.strut_compression().clone())
        history["tilt"].append(tilt_angle_from_vertical(truth.orientation_quaternion))
    history = {name: torch.stack(values) for name, values in history.items()}
    weight = preset.mass_at_start * MOON_SURFACE_GRAVITY
    return history, preset, weight


def test_no_bounce(touchdown):
    """After first contact, the footpads never leave the ground in any group."""
    history, _, weight = touchdown
    total_force = history["leg_forces"].sum(dim=2)
    in_contact = total_force > 0.01 * weight
    assert bool(in_contact.any(dim=0).all()), "some lander never touched the ground"
    after_first_contact = in_contact.cumsum(dim=0) > 0
    lost_contact = after_first_contact & (total_force < 0.001 * weight)
    assert int(lost_contact.sum()) == 0


def test_sum_of_leg_forces_equals_weight(touchdown):
    """Mean total leg force over the last 2 s is m g within 0.5 %, in every environment."""
    history, _, weight = touchdown
    mean_total = history["leg_forces"][-100:].sum(dim=2).mean(dim=0)
    assert float((mean_total / weight - 1.0).abs().max()) < 0.005


def test_symmetric_landings_share_the_load_equally(touchdown):
    """Groups A and B: each leg carries m g / 4 within 3 % (the c.g. offset moves about 2 % of the load)."""
    history, _, weight = touchdown
    mean_per_leg = history["leg_forces"][-100:].mean(dim=0)
    for group in (GENTLE, HARD):
        assert float((mean_per_leg[group] / (0.25 * weight) - 1.0).abs().max()) < 0.03


def test_strut_compression_matches_the_spring(touchdown):
    """Groups A and B: rest compression is each leg's static share of the body weight / stiffness within 2 %."""
    history, preset, _ = touchdown
    body_weight = preset.body_mass * MOON_SURFACE_GRAVITY
    axis_x, axis_y = preset.vehicle_axis_position
    body_x, body_y, _ = preset.body_center_of_mass
    radius = preset.footpad_radius_from_thrust_axis
    shares = [
        0.25 + ((body_x - axis_x) * (leg_x - axis_x) + (body_y - axis_y) * (leg_y - axis_y)) / (2.0 * radius**2)
        for leg_x, leg_y, _ in preset.leg_positions
    ]
    expected = torch.tensor(shares) * body_weight / preset.strut_stiffness
    rest_compression = history["strut"][-100:].mean(dim=0)
    for group in (GENTLE, HARD):
        assert float((rest_compression[group] / expected - 1.0).abs().max()) < 0.02
    assert float(history["strut"].max()) < 0.95 * preset.strut_stroke


def test_lander_comes_to_rest_without_jitter(touchdown):
    """All groups: no drift, no height jitter, small velocity noise, upright (the PhysX test's limits)."""
    history, _, weight = touchdown
    position = history["position"]
    drift = (position[-1] - position[-200]).norm(dim=1)
    assert float(drift.max()) < 1e-3
    assert float(position[-100:, :, 2].std(dim=0).max()) < 1e-4
    assert float(history["speed"][-100:].max()) < 0.01
    assert float(history["rate"][-100:].max()) < math.radians(0.2)
    assert float(history["tilt"][-1].max()) < math.radians(0.5)
    assert float(history["leg_forces"][-100:].sum(dim=2).std(dim=0).max()) < 0.03 * weight


def test_sliding_landing_stops_by_friction(touchdown):
    """Group C: the lander slides between 0.02 m and 1.2 m and stops."""
    history, _, _ = touchdown
    slide = history["position"][-1, SLIDING, 0] - history["position"][0, SLIDING, 0]
    assert float(slide.min()) > 0.02
    assert float(slide.max()) < 1.2


# ---- inertia bookkeeping ----


def test_inertia_matches_the_preset_at_the_start_mass_and_falls_with_the_body_mass():
    """At the start mass the inertia is the preset's; burning propellant lowers only the body's share."""
    environment = make_environment("s1_terminal_truth", ["num_envs=1"])
    plant = environment.bindings[SlotName.PLANT]
    preset = plant.preset
    at_start = plant.inertia(torch.tensor([preset.mass_at_start]))[0]
    assert at_start.tolist() == pytest.approx([preset.inertia_about_x, preset.inertia_about_y, preset.inertia_about_z], rel=1e-5)
    lighter = plant.inertia(torch.tensor([preset.mass_at_start - 100.0]))[0]
    assert bool((lighter < at_start).all())
    assert bool((lighter > plant.legs_inertia).all())
