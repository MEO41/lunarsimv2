# SPDX-License-Identifier: BSD-3-Clause
"""Controller build-up step 2 (GPU): hover.

The vertical loop holds a footpad altitude of 30 m while the mass drops. Eight landers start
between 27 m and 33 m with vertical speeds between -2 and +1 m/s and tilts up to 5 degrees.

Pass criteria and why:
- From 15 s to 60 s the altitude error stays below 5 cm and the vertical speed below 3 cm/s.
  The loop is a critically damped second-order system with a natural frequency of 1 rad/s, so
  a 3 m error has decayed to under 1 mm after 15 s. The thrust law uses the estimated mass, so
  no steady error builds up while propellant burns; what is left is the 0.08 s thrust lag and
  the small vertical force changes when attitude jets fire.
- The mass drops by more than 200 kg in 60 s (about 4 kg/s of hover flow), which shows that
  the hold really ran against a changing mass.
- Tilt below 1 degree after 15 s (the attitude loop of step 1 runs at the same time). The loop
  fires minimum 14 ms pulses inside a rate dead band, so the tilt cycles. With the jet arms of
  MIT R-567 (1.68 m along each axis, 42 % longer than the 1.19 m of milestone 1a) one pulse
  turns the lander more, and the measured cycle reaches 0.65 degrees.
"""

import math

import pytest
import torch
from isaac_test_support import build_agent, make_environment, read_truth, set_start_states

from lunarsim.core.message_types import Observation
from lunarsim.core.rotation_math import tilt_angle_from_vertical

TARGET_ALTITUDE = 30.0


@pytest.fixture(scope="module")
def history():
    """Run the hover-hold agent for 60 s and record altitude, vertical speed, tilt and mass."""
    environment = make_environment("s1_terminal_truth", ["num_envs=8", "agent=hover_hold", "metrics=none"])
    agent = build_agent(environment)
    device = environment.device
    observations = set_start_states(
        environment,
        altitude=torch.linspace(27.0, 33.0, 8, device=device),
        vertical_speed=torch.linspace(-2.0, 1.0, 8, device=device),
        tilt_degrees=torch.linspace(0.0, 5.0, 8, device=device),
    )
    step = environment.clock.control_step_duration
    cg_height = agent.preset.center_of_mass_height_above_footpad_plane
    record = {"altitude": [], "vertical_speed": [], "tilt": [], "mass": []}
    for _ in range(int(60.0 / step)):
        command = agent.compute_command(Observation(policy=observations["policy"]), step)
        observations, _, _, _, _ = environment.step(command.to_action_tensor())
        truth = read_truth(environment)
        record["altitude"].append(truth.position[:, 2] - cg_height)
        record["vertical_speed"].append(truth.velocity[:, 2].clone())
        record["tilt"].append(tilt_angle_from_vertical(truth.orientation_quaternion))
        record["mass"].append(truth.mass.clone())
    yield {name: torch.stack(values) for name, values in record.items()}, step
    environment.close()


def test_altitude_is_held_while_mass_drops(history):
    """Altitude error below 5 cm and vertical speed below 3 cm/s from 15 s to 60 s."""
    record, step = history
    late = slice(int(15.0 / step), None)
    print("ALTITUDE_ERROR_MAX_M", float((record["altitude"][late] - TARGET_ALTITUDE).abs().max()))
    print("VERTICAL_SPEED_MAX_MPS", float(record["vertical_speed"][late].abs().max()))
    assert float((record["altitude"][late] - TARGET_ALTITUDE).abs().max()) < 0.05
    assert float(record["vertical_speed"][late].abs().max()) < 0.03


def test_mass_really_dropped_and_attitude_stayed_upright(history):
    """More than 200 kg burned in 60 s; tilt below 1 degree after 15 s."""
    record, step = history
    burned = record["mass"][0] - record["mass"][-1]
    assert float(burned.min()) > 200.0
    print("TILT_MAX_DEG", math.degrees(float(record["tilt"][int(15.0 / step) :].max())))
    assert float(record["tilt"][int(15.0 / step) :].max()) < math.radians(1.0)
