# SPDX-License-Identifier: BSD-3-Clause
"""Physics validation (GPU): reaction-control torque.

Known answer: angular acceleration = torque / inertia, per axis.

Six landers float in free fall with the engine off. Each fires one jet couple (positive or
negative torque about body x, y or z) for 1 s, then coasts for 0.1 s so the valve delay has
passed. The body rate must be torque * 1 s / inertia about that axis.

Tolerances and why:
- Rate about the fired axis within 0.5 % of torque * t / I. The pulse model delivers the exact
  impulse; the remaining effects are the mass (and so inertia) that the jets burn in 1 s
  (0.65 kg of 7,327 kg, 0.01 %) and float32 integration over 220 steps.
- Rates about the other two axes below 0.5 % of the fired-axis rate: the couples are pure and
  the inertia tensor is diagonal, so only gyroscopic coupling of numerical noise remains.
- No change of linear velocity beyond free fall (2 mm/s): a couple has no net force.
"""

import pytest
import torch
from isaac_test_support import make_actions, make_environment, read_truth, set_start_states

from lunarsim.agents.classical.jet_selection import JetCoupleTable
from lunarsim.core.binding import SlotName
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY

FIRING_TIME = 1.0
COAST_TIME = 0.1


@pytest.fixture(scope="module")
def spin_result():
    """Fire the six couples and return the final truth, the couple table and the preset."""
    environment = make_environment("s1_terminal_truth", ["num_envs=6", "actuators.initial_throttle=0.0"])
    preset = environment.bindings[SlotName.PLANT].preset
    table = JetCoupleTable(preset, device=environment.device)
    axis_commands = torch.zeros(6, 3, device=environment.device)
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
    yield read_truth(environment), table, preset
    environment.close()


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
