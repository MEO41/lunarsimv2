# SPDX-License-Identifier: BSD-3-Clause
"""Physics validation (GPU): hover while mass drops.

Known answer: thrust = m g keeps the altitude constant while propellant burns, and the mass
drops by the integral of thrust / (Isp g0).

Environment 0 commands thrust = m(t) g from the mass in its observation (hover).
Environment 1 holds the thrust for the start mass (control case): as its mass drops it must
rise by about g mdot t^3 / (6 m0). This proves that the mass really drops inside PhysX.

Tolerances and why:
- Hover drift below 2 cm and speed below 5 mm/s over 20 s. The command uses the mass at the
  start of each 20 ms control step, during which 0.08 kg burns, so the thrust is 0.13 N too
  high: 1.7e-5 m/s^2, or 3.5 mm over 20 s. The 0.08 s thrust lag adds less than that.
- Mass drop within 0.1 % of thrust / (Isp g0) integrated over time: only float32 summation.
- Control-case rise within 5 % of the cubic formula, which is itself a first-order
  approximation (it neglects terms of order mdot t / m0 = 1 %).
"""

import pytest
from isaac_test_support import make_actions, make_environment, read_truth, set_start_states

from lunarsim.core.binding import SlotName
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY

HOVER_TIME = 20.0


@pytest.fixture(scope="module")
def hover_result():
    """Run the hover and the control case for 20 s."""
    environment = make_environment("s1_terminal_truth", ["num_envs=2", "metrics=none"])
    preset = environment.bindings[SlotName.PLANT].preset
    observations = set_start_states(environment, altitude=30.0)
    start = read_truth(environment)
    start_height, start_mass = start.position[:, 2].clone(), start.mass.clone()
    expected_burn = 0.0
    for _ in range(int(HOVER_TIME / environment.clock.control_step_duration)):
        mass = observations["policy"][:, 13].clone()
        mass[1] = start_mass[1]  # control case: thrust for the start mass, never updated
        throttle = mass * MOON_SURFACE_GRAVITY / preset.engine_rated_thrust
        expected_burn += float(throttle[0]) * preset.engine_rated_thrust / preset.engine_exhaust_velocity * 0.02
        observations, _, _, _, _ = environment.step(make_actions(environment, throttle=throttle))
    end = read_truth(environment)
    yield start_height, start_mass, end, expected_burn, preset
    environment.close()


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
    assert burned > 70.0  # about 3.98 kg/s for 20 s


def test_constant_thrust_makes_the_lighter_lander_rise(hover_result):
    """Control case: thrust for the start mass lifts the lander by g mdot t^3 / (6 m0), within 5 %."""
    start_height, start_mass, end, _, preset = hover_result
    mass_flow = float(start_mass[1]) * MOON_SURFACE_GRAVITY / preset.engine_exhaust_velocity
    expected_rise = MOON_SURFACE_GRAVITY * mass_flow * HOVER_TIME**3 / (6.0 * float(start_mass[1]))
    assert float(end.position[1, 2] - start_height[1]) == pytest.approx(expected_rise, rel=0.05)
