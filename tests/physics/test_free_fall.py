# SPDX-License-Identifier: BSD-3-Clause
"""Physics validation (GPU): free fall.

Known answer: z(t) = z0 - 1/2 g t^2 with g = GM / R^2 = 1.6242 m/s^2 (the design table rounds
this to 1.62).

Tolerances and why:
- Height: 3 mm after 2 s (400 PhysX steps of 5 ms). A step-by-step check showed that PhysX
  moves the articulation root with the mean velocity of the step, which reproduces the exact
  parabola (the first steps match to the last float32 digit). What remains is float32 rounding:
  a height near 54 m is stored in steps of 3.8 micrometres, and the measured error grows to
  1.1 mm in 2 s. 3 mm leaves a factor of three.
- Speed: |v + g t| < 1 mm/s. The measured error is 0.1 mm/s, again float32 accumulation.
"""

import pytest
import torch
from isaac_test_support import make_actions, make_environment, read_truth, set_start_states

from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY

START_ALTITUDE = 50.0
FALL_TIME = 2.0


@pytest.fixture(scope="module")
def fallen_state():
    """Drop 16 landers from rest with the engine off for 2 s and return the start and end truth."""
    environment = make_environment("s1_terminal_truth", ["num_envs=16", "actuators.initial_throttle=0.0"])
    set_start_states(environment, altitude=START_ALTITUDE)
    start = read_truth(environment)
    start_height = start.position[:, 2].clone()
    for _ in range(int(FALL_TIME / environment.clock.control_step_duration)):
        environment.step(make_actions(environment, throttle=0.0))
    end = read_truth(environment)
    result = (start_height, end, environment.clock.physics_step_duration)
    yield result
    environment.close()


def test_height_follows_the_parabola(fallen_state):
    """z matches z0 - 1/2 g t^2 within 3 mm after 2 s, in all 16 environments."""
    start_height, end, _ = fallen_state
    exact = start_height - 0.5 * MOON_SURFACE_GRAVITY * FALL_TIME**2
    assert float((end.position[:, 2] - exact).abs().max()) < 0.003


def test_speed_is_g_times_t(fallen_state):
    """v_z = -g t within 1 mm/s; no horizontal speed appears."""
    _, end, _ = fallen_state
    assert float((end.velocity[:, 2] + MOON_SURFACE_GRAVITY * FALL_TIME).abs().max()) < 0.001
    assert float(end.velocity[:, :2].abs().max()) < 1e-4
    assert float(end.position[:, :2].abs().max()) < 1e-3


def test_no_rotation_and_no_mass_change_in_free_fall(fallen_state):
    """Attitude stays upright, body rates stay zero, and mass is constant with the engine off."""
    _, end, _ = fallen_state
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=end.position.device)
    assert torch.allclose(end.orientation_quaternion, identity.expand(16, 4), atol=1e-5)
    assert float(end.angular_velocity.abs().max()) < 1e-5
    assert float((end.mass - end.mass[0]).abs().max()) == 0.0
    assert float(end.contact.abs().max()) == 0.0
