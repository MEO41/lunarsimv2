# SPDX-License-Identifier: BSD-3-Clause
"""Physics validation (GPU): the footpad friction is the surface's coefficient (0.4).

Known answer: a lander that slides on flat ground with speed v stops after v^2 / (2 mu g).

Eight landers start standing on the flat ground (footpads just touching), engine off, upright,
with horizontal speeds of 0.5 to 1.5 m/s. With mu = 0.4 (NASA TN D-6850) and g = 1.6242 m/s^2
the slide is 0.077 m to 1.73 m.

Tolerance and why: within 15 % of the formula. The legs carry the weight through springs, so
the normal force swings a little while the lander settles in the first tenths of a second; the
slide also includes the short time before the footpads are loaded. A wrong combination of the
two materials would give mu = 0.2 and twice the distance, which this test catches.
"""

import pytest
import torch
from isaac_test_support import make_actions, make_environment, read_truth, set_start_states

from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY

COUNT = 8
FRICTION = 0.4


@pytest.fixture(scope="module")
def slides():
    """Let the landers slide for 6 s and return start speeds and slide distances."""
    environment = make_environment(
        "s1_terminal_truth", [f"num_envs={COUNT}", "metrics=none", "actuators.initial_throttle=0.0"]
    )
    speeds = torch.linspace(0.5, 1.5, COUNT, device=environment.device)
    set_start_states(environment, altitude=0.0)
    start = environment.bindings["start"]
    position, velocity, orientation = start.fixed_start_states
    velocity[:, 0] = speeds
    start.set_fixed_start_states(position, velocity, orientation)
    environment.reset()
    first = read_truth(environment).position[:, 0].clone()
    for _ in range(int(6.0 / environment.clock.control_step_duration)):
        environment.step(make_actions(environment, throttle=0.0))
    truth = read_truth(environment)
    yield speeds, truth.position[:, 0] - first, truth.velocity.norm(dim=1)
    environment.close()


def test_slide_distance_matches_the_friction_coefficient(slides):
    """Each slide is v^2 / (2 * 0.4 * g) within 15 %, and every lander has stopped."""
    speeds, distance, final_speed = slides
    expected = speeds**2 / (2.0 * FRICTION * MOON_SURFACE_GRAVITY)
    print("slide", distance.tolist(), "expected", expected.tolist())
    assert float(final_speed.max()) < 0.02
    assert torch.allclose(distance, expected, rtol=0.15)
