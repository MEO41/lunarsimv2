# SPDX-License-Identifier: BSD-3-Clause
"""Physics validation (CPU): propellant use and actuator timing.

Known answers:
- mass flow = thrust / (Isp g0)
- delta-v of a burn = Isp g0 ln(m0 / m1) (the rocket equation)
"""

import math

import pytest
import torch

from lunarsim.core.physical_constants import STANDARD_GRAVITY
from lunarsim.slots.actuators.propulsion_models import DescentEngineModel, ReactionControlPulseModel

RATED_THRUST = 46706.0
SPECIFIC_IMPULSE = 305.0
EXHAUST_VELOCITY = SPECIFIC_IMPULSE * STANDARD_GRAVITY
PHYSICS_STEP = 0.005


def make_engine(count=1, lag=0.08, rate=0.85, propellant=1.0e6):
    """Return an engine at 30 % steady throttle with a large propellant load."""
    engine = DescentEngineModel(count, RATED_THRUST, 0.10, 0.65, lag, rate, 0.05, EXHAUST_VELOCITY)
    engine.reset(torch.arange(count), 0.30, propellant)
    return engine


def test_mass_flow_equals_thrust_over_exhaust_velocity():
    """Steady state: mdot = F / (Isp g0). Tolerance 1e-6 relative: only float32 rounding."""
    engine = make_engine()
    thrust, mass_flow = engine.step(torch.tensor([0.30]), PHYSICS_STEP)
    assert float(thrust) == pytest.approx(0.30 * RATED_THRUST, rel=1e-6)
    assert float(mass_flow) == pytest.approx(0.30 * RATED_THRUST / EXHAUST_VELOCITY, rel=1e-6)


def test_delta_v_matches_the_rocket_equation():
    """Integrate dv = F/m dt with the model's mass flow for a 60 s burn.

    Tolerance 0.01 %: the explicit integration at 5 ms has a truncation error of about
    mdot dt / (2 m), which is 1e-6 here; float64 is used for the sum so rounding is not the limit.
    """
    engine = make_engine()
    mass = 7327.0
    delta_v = 0.0
    for _ in range(int(60.0 / PHYSICS_STEP)):
        thrust, mass_flow = engine.step(torch.tensor([0.30]), PHYSICS_STEP)
        mass_after = mass - float(mass_flow) * PHYSICS_STEP
        delta_v += float(thrust) / (0.5 * (mass + mass_after)) * PHYSICS_STEP
        mass = mass_after
    expected = EXHAUST_VELOCITY * math.log(7327.0 / mass)
    assert delta_v == pytest.approx(expected, rel=1e-4)
    assert 7327.0 - mass == pytest.approx(0.30 * RATED_THRUST / EXHAUST_VELOCITY * 60.0, rel=1e-4)


def test_throttle_is_clamped_to_the_throttleable_range_and_cuts_off():
    """Commands are held inside 10..65 % of rated thrust; a command below 5 % shuts down."""
    engine = make_engine(count=3, lag=0.0, rate=1.0e6)
    thrust, _ = engine.step(torch.tensor([1.0, 0.07, 0.01]), PHYSICS_STEP)
    assert thrust.tolist() == pytest.approx([0.65 * RATED_THRUST, 0.10 * RATED_THRUST, 0.0], rel=1e-6)


def test_thrust_follows_a_first_order_lag():
    """After one time constant the thrust has covered 63.2 % of a step (slew limit switched off).

    Tolerance 0.5 % of the step: the model integrates the lag exactly, so only rounding remains.
    """
    engine = make_engine(lag=0.08, rate=1.0e6)
    for _ in range(int(0.08 / PHYSICS_STEP)):
        engine.step(torch.tensor([0.60]), PHYSICS_STEP)
    expected = 0.30 + (0.60 - 0.30) * (1.0 - math.exp(-1.0))
    assert float(engine.thrust_fraction) == pytest.approx(expected, abs=0.005 * 0.30)


def test_throttle_command_slews_at_the_rate_limit():
    """With no lag, the thrust fraction rises at 0.85 per second."""
    engine = make_engine(lag=0.0, rate=0.85)
    for _ in range(int(0.2 / PHYSICS_STEP)):
        engine.step(torch.tensor([0.65]), PHYSICS_STEP)
    assert float(engine.thrust_fraction) == pytest.approx(0.30 + 0.85 * 0.2, abs=1e-4)


def test_engine_stops_when_propellant_runs_out_and_burns_exactly_the_load():
    """The engine burns all of its propellant, not more, then gives no thrust."""
    engine = make_engine(propellant=2.0)
    total_burned = 0.0
    for _ in range(400):
        _, mass_flow = engine.step(torch.tensor([0.30]), PHYSICS_STEP)
        total_burned += float(mass_flow) * PHYSICS_STEP
    thrust, _ = engine.step(torch.tensor([0.30]), PHYSICS_STEP)
    assert total_burned == pytest.approx(2.0, rel=1e-5)
    assert float(thrust) == 0.0


def make_jets(valve_delay=0.007, minimum_pulse=0.014):
    """Return a pulse model for one environment with three jets."""
    return ReactionControlPulseModel(1, 3, minimum_pulse, valve_delay, PHYSICS_STEP)


def run_control_steps(jets, duties, substeps=4):
    """Run one control step per duty row and return the open fractions of all substeps."""
    fractions = []
    for duty in duties:
        jets.begin_control_step(torch.tensor([duty]), substeps * PHYSICS_STEP)
        for _ in range(substeps):
            fractions.append(jets.step()[0].clone())
    return torch.stack(fractions)


def test_pulse_impulse_is_exact_and_short_pulses_are_dropped():
    """Open time summed over substeps equals the commanded on-time; on-times below 14 ms give nothing.

    Jet 0: duty 1.0 -> 20 ms. Jet 1: duty 0.75 -> 15 ms (kept). Jet 2: duty 0.5 -> 10 ms (dropped).
    """
    fractions = run_control_steps(make_jets(), [[1.0, 0.75, 0.5], [0.0, 0.0, 0.0]])
    open_time = fractions.sum(dim=0) * PHYSICS_STEP
    assert open_time.tolist() == pytest.approx([0.020, 0.015, 0.0], abs=1e-7)


def test_valve_delay_shifts_the_pulse_by_seven_milliseconds():
    """With a 7 ms delay and 5 ms substeps, a pulse that starts at t=0 opens 40 % into substep 2."""
    fractions = run_control_steps(make_jets(), [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])[:, 0]
    # substep: 0 (0-5 ms) closed, 1 (5-10 ms) open for 3 ms, 2..4 open, 5 open for 2 ms, then closed
    assert fractions.tolist() == pytest.approx([0.0, 0.6, 1.0, 1.0, 1.0, 0.4, 0.0, 0.0], abs=1e-6)


def test_continuous_firing_keeps_the_valve_open_across_control_steps():
    """Duty 1.0 in consecutive control steps gives a valve that stays fully open."""
    fractions = run_control_steps(make_jets(), [[1.0, 0.0, 0.0]] * 3)[:, 0]
    assert fractions[2:].tolist() == pytest.approx([1.0] * 10, abs=1e-6)


def test_shutdown_thrust_falls_in_a_straight_line_over_the_decay_time():
    """After a shutdown the thrust falls linearly to zero in 0.5 s; the impulse is half of F0 times 0.5 s."""
    engine = DescentEngineModel(1, RATED_THRUST, 0.10, 0.65, 0.08, 0.85, 0.05, EXHAUST_VELOCITY, shutdown_decay_time=0.5)
    engine.reset(torch.arange(1), 0.30, 1.0e6)
    start_thrust = 0.30 * RATED_THRUST
    impulse, thrust_at = 0.0, {}
    for step in range(200):  # 1 s
        thrust, _ = engine.step(torch.tensor([0.0]), PHYSICS_STEP)
        impulse += float(thrust[0]) * PHYSICS_STEP
        thrust_at[round((step + 1) * PHYSICS_STEP, 3)] = float(thrust[0])
    assert thrust_at[0.25] == pytest.approx(0.5 * start_thrust, rel=0.02)
    assert thrust_at[0.5] == pytest.approx(0.0, abs=0.01 * start_thrust)
    assert thrust_at[1.0] == 0.0
    assert impulse == pytest.approx(0.5 * start_thrust * 0.5, rel=1e-3)


def test_engine_restarts_during_the_tail_off():
    """A new throttle command during the tail-off brings the engine back."""
    engine = DescentEngineModel(1, RATED_THRUST, 0.10, 0.65, 0.08, 0.85, 0.05, EXHAUST_VELOCITY, shutdown_decay_time=0.5)
    engine.reset(torch.arange(1), 0.30, 1.0e6)
    for _ in range(20):  # 0.1 s off
        engine.step(torch.tensor([0.0]), PHYSICS_STEP)
    for _ in range(400):  # 2 s on
        thrust, _ = engine.step(torch.tensor([0.30]), PHYSICS_STEP)
    assert float(thrust[0]) == pytest.approx(0.30 * RATED_THRUST, rel=1e-3)
