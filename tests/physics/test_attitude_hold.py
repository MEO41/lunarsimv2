# SPDX-License-Identifier: BSD-3-Clause
"""Controller build-up step 1 (GPU): attitude hold.

The jets bring the lander upright from a small tilt while the engine gives hover thrust.
Eight landers start with tilts of 1, 2, 4, 6, 8, 10, 10 and 10 degrees.

Pass criteria and why:
- Tilt below 0.6 degrees after 12 s and for the following 8 s. The controller has a dead band
  of 0.3 degrees per axis (0.42 degrees for two axes together), and one minimum jet pulse
  changes the rate by about 0.1 deg/s, so the lander drifts slowly inside the dead band.
- Body rate below 0.3 deg/s in the last 8 s: three minimum pulses.
- Peak body rate below 6 deg/s: the controller caps the wanted rate at 5 deg/s.
- Jet propellant below 2 kg in 20 s. Measured: 0.13 kg (1 degree) to 1.16 kg (10 degrees).
  Righting 10 degrees needs about 1.5 s of one couple (4 jets at 0.16 kg/s); the rest is
  dead-band keeping. A chattering controller would burn about 6 kg (this was seen before the
  rate dead band was added).
"""

import math

import pytest
import torch
from isaac_test_support import build_agent, make_environment, read_truth, set_start_states

from lunarsim.core.message_types import Observation
from lunarsim.core.rotation_math import tilt_angle_from_vertical

START_TILTS = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 10.0, 10.0]


@pytest.fixture(scope="module")
def history():
    """Run the attitude-hold agent for 20 s and record tilt, rate and jet use."""
    environment = make_environment("s1_terminal_truth", ["num_envs=8", "agent=attitude_hold", "metrics=none"])
    agent = build_agent(environment)
    tilts = torch.tensor(START_TILTS, device=environment.device)
    observations = set_start_states(environment, altitude=40.0, tilt_degrees=tilts)
    step = environment.clock.control_step_duration
    record = {"tilt": [], "rate": [], "jets_open": []}
    for _ in range(int(20.0 / step)):
        command = agent.compute_command(Observation(policy=observations["policy"]), step)
        observations, _, _, _, _ = environment.step(command.to_action_tensor())
        truth = read_truth(environment)
        record["tilt"].append(tilt_angle_from_vertical(truth.orientation_quaternion))
        record["rate"].append(truth.angular_velocity.norm(dim=1))
        record["jets_open"].append(command.reaction_control_jets.sum(dim=1))
    preset = agent.preset
    yield {name: torch.stack(values) for name, values in record.items()}, preset, step
    environment.close()


def test_lander_becomes_upright_and_stays_upright(history):
    """From up to 10 degrees, the tilt is below 0.6 degrees after 12 s and stays there."""
    record, _, step = history
    late = record["tilt"][int(12.0 / step) :]
    print("LATE_TILT_MAX_DEG", [round(math.degrees(float(value)), 3) for value in late.max(dim=0).values])
    assert float(late.max()) < math.radians(0.6)


def test_rates_stay_bounded(history):
    """Peak rate below 6 deg/s; residual rate below 0.3 deg/s."""
    record, _, step = history
    assert float(record["rate"].max()) < math.radians(6.0)
    assert float(record["rate"][int(12.0 / step) :].max()) < math.radians(0.3)


def test_jet_propellant_use_is_small(history):
    """Each lander burns less than 2 kg of jet propellant in 20 s."""
    record, preset, step = history
    mass_flow_per_jet = preset.jet_thrust / preset.jet_exhaust_velocity
    burned = record["jets_open"].sum(dim=0) * step * mass_flow_per_jet
    print("JET_PROPELLANT_KG", [round(float(value), 2) for value in burned])
    assert float(burned.max()) < 2.0
