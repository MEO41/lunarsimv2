# SPDX-License-Identifier: BSD-3-Clause
"""Physics validation (GPU): free fall.

Known answer: z(t) = z0 - 1/2 g t^2 with g = GM / R^2 = 1.6242 m/s^2 (the design table rounds
this to 1.62).

Tolerances and why:
- Against the exact formula: 2 cm after 2 s. With the PGS solver PhysX integrates with
  semi-implicit Euler (first the speed, then the position with the new speed), which falls
  1/2 g t dt further than the exact parabola: 8.1 mm at t = 2 s with dt = 5 ms.
- Against the integrator's own closed form (z0 - 1/2 g t^2 - 1/2 g t dt): 1 mm. What is left
  is float32 rounding: a height near 54 m is stored in steps of 3.8 micrometres, over 400
  steps. (With the TGS solver PhysX follows the exact parabola instead; this was measured
  during the contact study.)
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
    """z matches z0 - 1/2 g t^2 within 2 cm, and the integrator's closed form within 1 mm."""
    start_height, end, physics_step = fallen_state
    exact = start_height - 0.5 * MOON_SURFACE_GRAVITY * FALL_TIME**2
    integrator = exact - 0.5 * MOON_SURFACE_GRAVITY * FALL_TIME * physics_step
    assert float((end.position[:, 2] - exact).abs().max()) < 0.02
    assert float((end.position[:, 2] - integrator).abs().max()) < 0.001


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
