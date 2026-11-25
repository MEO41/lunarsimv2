# SPDX-License-Identifier: BSD-3-Clause
"""Test T8 support (CPU): ideal actuators and the point-mass plant of the model ablation."""

import math

import pytest
import torch
from analytic_environment_support import make_actions, make_environment, read_truth, set_start_states

from lunarsim.core.binding import SlotName
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.core.rotation_math import body_up_axis_in_parent, tilt_angle_from_vertical

POINT_MASS = "analytic_point_mass"
END_AT_CONTACT = "metrics.end_at_first_contact=true"


def hover_throttle(environment) -> float:
    """Return the throttle fraction whose thrust balances the start weight."""
    preset = environment.bindings[SlotName.PLANT].preset
    return preset.mass_at_start * MOON_SURFACE_GRAVITY / preset.engine_rated_thrust


def tilted_direction(count: int, tilt_degrees: float) -> torch.Tensor:
    """Return thrust directions tilted by ``tilt_degrees`` towards site +x, (count, 3)."""
    tilt = math.radians(tilt_degrees)
    return torch.tensor([math.sin(tilt), 0.0, math.cos(tilt)]).repeat(count, 1)


# ---- ideal actuators ----


def test_ideal_actuators_remove_the_response_limits():
    """With actuators.ideal_response=true the engine has no lag, slew limit or tail-off, the jets no minimum pulse or delay."""
    environment = make_environment("s1_terminal_truth", ["num_envs=2", "actuators.ideal_response=true"])
    actuators = environment.bindings[SlotName.ACTUATORS]
    assert actuators.engine.lag_time_constant == 0.0
    assert math.isinf(actuators.engine.maximum_throttle_rate)
    assert actuators.engine.shutdown_decay_time == 0.0
    assert actuators.jets.minimum_pulse_duration == 0.0
    standard = make_environment("s1_terminal_truth", ["num_envs=2"]).bindings[SlotName.ACTUATORS]
    assert standard.engine.lag_time_constant > 0.0 and standard.jets.minimum_pulse_duration > 0.0


def test_ideal_engine_reaches_a_throttle_step_in_one_control_step():
    """A step from hover to 55 % throttle is flown at once with ideal actuators, but not with the standard ones."""
    fractions = {}
    for ideal in (False, True):
        environment = make_environment(
            "s1_terminal_truth", ["num_envs=1", f"actuators.ideal_response={str(ideal).lower()}"]
        )
        set_start_states(environment, altitude=50.0)
        environment.step(make_actions(environment, throttle=0.55))
        fractions[ideal] = float(environment.bindings[SlotName.ACTUATORS].engine.thrust_fraction[0])
    assert fractions[True] == pytest.approx(0.55, abs=1e-5)
    assert fractions[False] < 0.5


# ---- point mass ----


def test_point_mass_needs_end_at_first_contact():
    """The point mass has no legs, so the environment refuses it unless the episode ends at the first contact."""
    with pytest.raises(ValueError, match="end_at_first_contact"):
        make_environment("s1_terminal_truth", ["num_envs=2"], plant=POINT_MASS)


def test_point_mass_refuses_to_fly_without_a_commanded_direction():
    """Without the training wrapper's direction the plant stops with an error instead of flying a wrong attitude."""
    environment = make_environment("s1_terminal_truth", ["num_envs=2", END_AT_CONTACT], plant=POINT_MASS)
    set_start_states(environment, altitude=50.0)
    with pytest.raises(RuntimeError, match="commanded thrust direction"):
        environment.step(make_actions(environment, throttle=0.3))


def test_point_mass_points_the_thrust_at_once_and_has_no_rate():
    """After one control step the thrust axis lies along a 10 degree command, the body rate is zero, and the lander accelerates towards +x."""
    environment = make_environment("s1_terminal_truth", ["num_envs=3", END_AT_CONTACT], plant=POINT_MASS)
    set_start_states(environment, altitude=80.0, tilt_degrees=3.0)
    direction = tilted_direction(3, 10.0)
    environment.set_commanded_thrust_direction(direction)
    environment.step(make_actions(environment, throttle=hover_throttle(environment)))
    truth = read_truth(environment)
    assert torch.allclose(body_up_axis_in_parent(truth.orientation_quaternion), direction, atol=1e-5)
    assert float(truth.angular_velocity.abs().max()) == 0.0
    assert bool((truth.velocity[:, 0] > 0.0).all())


def test_point_mass_hovers_upright_at_hover_throttle():
    """Upright at the hover throttle the vertical speed stays near zero for 2 s (the mass falls slowly)."""
    environment = make_environment(
        "s1_terminal_truth", ["num_envs=2", END_AT_CONTACT, "actuators.ideal_response=true"], plant=POINT_MASS
    )
    set_start_states(environment, altitude=60.0)
    upright = tilted_direction(2, 0.0)
    for _ in range(int(2.0 / environment.clock.control_step_duration)):
        environment.set_commanded_thrust_direction(upright)
        environment.step(make_actions(environment, throttle=hover_throttle(environment)))
    truth = read_truth(environment)
    assert float(truth.velocity[:, 2].abs().max()) < 0.05
    assert float(truth.velocity[:, :2].abs().max()) < 1e-6


def test_point_mass_falls_freely_stops_on_the_ground_and_ends_the_episode():
    """With the engine off it falls at lunar gravity; at the ground the first contact ends the episode with the approach speed.

    Ideal actuators, so the running start engine stops at once (the standard engine tails off).
    """
    environment = make_environment(
        "s1_terminal_truth", ["num_envs=1", END_AT_CONTACT, "actuators.ideal_response=true"], plant=POINT_MASS
    )
    set_start_states(environment, altitude=2.0)
    upright = tilted_direction(1, 0.0)
    step_time = environment.clock.control_step_duration
    terminated = torch.zeros(1, dtype=torch.bool)
    steps = 0
    while not bool(terminated[0]) and steps < 200:
        environment.set_commanded_thrust_direction(upright)
        _, _, terminated, _, _ = environment.step(make_actions(environment, throttle=0.0))
        steps += 1
    assert bool(terminated[0])
    expected_fall_time = math.sqrt(2.0 * 2.0 / MOON_SURFACE_GRAVITY)
    assert steps * step_time == pytest.approx(expected_fall_time, abs=2 * step_time)
    records = environment.bindings[SlotName.METRICS].drain_episode_records()
    assert bool(records["had_contact"][0])
    assert float(records["touchdown_vertical_speed_mps"][0]) == pytest.approx(
        -MOON_SURFACE_GRAVITY * expected_fall_time, abs=0.1
    )


LAGGED = "plant.attitude_time_constant_s=1.3"


def test_lagged_point_mass_closes_63_percent_of_a_tilt_step_after_one_time_constant():
    """With a 1.3 s lag a 10 degree command is reached to 63.2 % after 1.3 s, with a body rate about the turn axis."""
    environment = make_environment(
        "s1_terminal_truth", ["num_envs=1", END_AT_CONTACT, LAGGED, "actuators.ideal_response=true"], plant=POINT_MASS
    )
    set_start_states(environment, altitude=80.0)
    command = tilted_direction(1, 10.0)
    step_time = environment.clock.control_step_duration
    tilts, rates = [], []
    for _ in range(int(round(1.3 / step_time))):
        environment.set_commanded_thrust_direction(command)
        environment.step(make_actions(environment, throttle=hover_throttle(environment)))
        truth = read_truth(environment)
        tilts.append(math.degrees(float(tilt_angle_from_vertical(truth.orientation_quaternion)[0])))
        rates.append(truth.angular_velocity[0].clone())
    assert tilts[-1] == pytest.approx(10.0 * (1.0 - math.exp(-1.0)), abs=0.1)
    assert all(later > earlier for earlier, later in zip(tilts, tilts[1:], strict=False))
    # Turning towards +x is a rotation about +y; the rate decays as the gap closes.
    assert float(rates[0][1]) > 0.0 and abs(float(rates[0][0])) < 1e-6 and abs(float(rates[0][2])) < 1e-6
    assert float(rates[0][1]) == pytest.approx(math.radians(10.0) / 1.3, rel=0.05)
    assert float(rates[-1][1]) < float(rates[0][1])


def test_zero_time_constant_is_the_attitude_at_once():
    """attitude_time_constant_s=0 (the default) flies exactly as option A: thrust axis on the command, zero rate."""
    environment = make_environment(
        "s1_terminal_truth", ["num_envs=2", END_AT_CONTACT, "plant.attitude_time_constant_s=0"], plant=POINT_MASS
    )
    set_start_states(environment, altitude=80.0)
    command = tilted_direction(2, 7.0)
    environment.set_commanded_thrust_direction(command)
    environment.step(make_actions(environment, throttle=hover_throttle(environment)))
    truth = read_truth(environment)
    assert torch.allclose(body_up_axis_in_parent(truth.orientation_quaternion), command, atol=1e-5)
    assert float(truth.angular_velocity.abs().max()) == 0.0


def test_negative_time_constant_is_refused():
    """A negative time constant has no meaning and stops the build."""
    with pytest.raises(ValueError, match="attitude_time_constant_s"):
        make_environment(
            "s1_terminal_truth", ["num_envs=1", END_AT_CONTACT, "plant.attitude_time_constant_s=-1"], plant=POINT_MASS
        )


def test_lagged_point_mass_with_zero_rate_report_turns_but_reports_no_rate():
    """body_rate_report=zero keeps the 1.3 s lag of the thrust axis but reports a zero body rate."""
    environment = make_environment(
        "s1_terminal_truth",
        ["num_envs=1", END_AT_CONTACT, LAGGED, "plant.body_rate_report=zero", "actuators.ideal_response=true"],
        plant=POINT_MASS,
    )
    set_start_states(environment, altitude=80.0)
    command = tilted_direction(1, 10.0)
    for _ in range(10):
        environment.set_commanded_thrust_direction(command)
        environment.step(make_actions(environment, throttle=hover_throttle(environment)))
    truth = read_truth(environment)
    assert math.degrees(float(tilt_angle_from_vertical(truth.orientation_quaternion)[0])) > 1.0
    assert float(truth.angular_velocity.abs().max()) == 0.0


def test_point_mass_at_once_with_held_turn_rate_reports_the_turn_over_one_decision():
    """body_rate_report=held_turn_rate: the attitude jumps to a 10 degree command and reports 10 degrees per 0.1 s about +y until the next turn."""
    environment = make_environment(
        "s1_terminal_truth", ["num_envs=2", END_AT_CONTACT, "plant.body_rate_report=held_turn_rate"], plant=POINT_MASS
    )
    set_start_states(environment, altitude=80.0)
    command = tilted_direction(2, 10.0)
    for _ in range(3):
        environment.set_commanded_thrust_direction(command)
        environment.step(make_actions(environment, throttle=hover_throttle(environment)))
        truth = read_truth(environment)
        assert torch.allclose(body_up_axis_in_parent(truth.orientation_quaternion), command, atol=1e-5)
        assert float(truth.angular_velocity[0, 1]) == pytest.approx(math.radians(10.0) / 0.1, rel=1e-3)
        assert abs(float(truth.angular_velocity[0, 0])) < 1e-5


def test_unknown_body_rate_report_is_refused():
    """A body_rate_report value other than physical, zero or held_turn_rate stops the build."""
    with pytest.raises(ValueError, match="body_rate_report"):
        make_environment(
            "s1_terminal_truth", ["num_envs=1", END_AT_CONTACT, "plant.body_rate_report=sometimes"], plant=POINT_MASS
        )


@pytest.mark.parametrize("time_constant", [0.0, 1.3])
def test_ppo_trains_two_iterations_on_the_point_mass(time_constant):
    """rsl_rl trains on the point mass, at once and lagged: the wrapper hands the commanded direction over every control step."""
    from rsl_rl.runners import OnPolicyRunner

    from lunarsim.agents.rsl_rl.lander_vector_environment import LanderVectorEnvironment
    from lunarsim.agents.rsl_rl.training_configuration import runner_configuration

    environment = make_environment(
        "s1_terminal_truth",
        [
            "num_envs=16",
            "agent=ppo_mlp",
            "training=landing_ppo",
            END_AT_CONTACT,
            f"plant.attitude_time_constant_s={time_constant}",
        ],
        plant=POINT_MASS,
    )
    parameters = environment.scenario.slots["agent"].parameters
    vector_environment = LanderVectorEnvironment(environment, parameters)
    configuration = runner_configuration(parameters["ppo"], 0, "smoke", has_critic_group=True)
    configuration["max_iterations"] = 2
    runner = OnPolicyRunner(vector_environment, configuration, log_dir=None, device="cpu")
    runner.learn(num_learning_iterations=2)
    truth = read_truth(environment)
    assert bool(torch.isfinite(truth.position).all())
    if time_constant == 0.0:
        assert float(truth.angular_velocity.abs().max()) == 0.0
