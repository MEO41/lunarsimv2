# SPDX-License-Identifier: BSD-3-Clause
"""Engine cut-off of the classical agents (CPU): probe mode with delay, backup height, height mode."""

import math

import pytest
import torch

from lunarsim.agents.classical.landing_agents import VerticalDescentAgent
from lunarsim.assets.lander_preset import load_lander_preset
from lunarsim.core.message_types import NavigationStateLayout, Observation
from lunarsim.env.scenario_loader import default_config_directory

CONTROL_STEP = 0.02
PRESET = load_lander_preset(default_config_directory() / "landers" / "apollo_lunar_module.yaml")


def make_agent(count=1, **parameters):
    """Return a vertical-descent agent on the CPU with the given parameters."""
    agent = VerticalDescentAgent(parameters, count, PRESET.number_of_jets, "cpu", {"lander_preset": PRESET})
    agent.reset(torch.arange(count), torch.Generator())
    return agent


def observation(height, probe_contact, count=1, roll_rate=0.0):
    """Return an upright, slowly descending observation at the given height and contact-light state."""
    state = torch.zeros(count, NavigationStateLayout.DIMENSION)
    state[:, NavigationStateLayout.ANGULAR_VELOCITY] = torch.tensor([roll_rate, 0.0, 0.0])
    state[:, NavigationStateLayout.ORIENTATION_QUATERNION] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    state[:, NavigationStateLayout.VELOCITY] = torch.tensor([0.0, 0.0, -0.8])
    state[:, NavigationStateLayout.MASS] = PRESET.mass_at_start
    state[:, NavigationStateLayout.HEIGHT_ABOVE_GROUND] = height
    state[:, NavigationStateLayout.SURFACE_PROBE_CONTACT] = probe_contact
    return Observation(policy=state)


def steps_until_cut(agent, heights, lights):
    """Feed observations and return the index of the first step with zero throttle (or None)."""
    for index, (height, light) in enumerate(zip(heights, lights)):
        command = agent.compute_command(observation(height, light), CONTROL_STEP)
        if float(command.throttle[0]) == 0.0:
            return index
    return None


def test_probe_mode_cuts_the_engine_one_second_after_the_contact_light():
    """The light comes on at step 10; with a 1 s delay the throttle is zero from step 60 on."""
    agent = make_agent()
    lights = [0.0] * 10 + [1.0] * 100
    assert steps_until_cut(agent, [1.5] * 110, lights) == 60


def test_light_that_goes_off_again_still_counts():
    """A probe that bends away after touching does not stop the countdown."""
    agent = make_agent(engine_stop_delay_after_probe_contact_s=0.2)
    lights = [0.0] * 5 + [1.0] + [0.0] * 50
    assert steps_until_cut(agent, [1.5] * 56, lights) == 15


def test_zero_delay_cuts_at_once_and_jets_close():
    """With no delay the engine stops on the step that sees the light, and the jets are closed."""
    agent = make_agent(engine_stop_delay_after_probe_contact_s=0.0)
    command = agent.compute_command(observation(1.6, 1.0), CONTROL_STEP)
    assert float(command.throttle[0]) == 0.0
    assert float(command.reaction_control_jets.abs().sum()) == 0.0


def test_backup_height_cuts_without_a_light():
    """No probe signal: the engine runs down to the backup height, then stops."""
    agent = make_agent(backup_engine_cutoff_height_m=0.1)
    heights = [0.5 - 0.05 * k for k in range(12)]
    assert steps_until_cut(agent, heights, [0.0] * 12) == 9  # 0.05 m is the first height below 0.1 m


def test_height_mode_ignores_the_light():
    """The old rule: only the height straight below matters, the light is ignored."""
    agent = make_agent(engine_cutoff_signal="height_above_ground")
    heights = [1.0 - 0.1 * k for k in range(10)]
    assert steps_until_cut(agent, heights, [1.0] * 10) == 8  # 0.2 m is the first height below 0.25 m


def test_reset_rearms_the_engine_and_forgets_the_light():
    """After a reset the countdown starts again from the next light."""
    agent = make_agent(engine_stop_delay_after_probe_contact_s=0.1)
    assert steps_until_cut(agent, [1.5] * 10, [1.0] * 10) == 5
    agent.reset(torch.arange(1), torch.Generator())
    assert float(agent.compute_command(observation(1.5, 0.0), CONTROL_STEP).throttle[0]) > 0.0


def test_unknown_signal_is_rejected():
    """A misspelt cut-off signal stops the agent from being built."""
    with pytest.raises(ValueError):
        make_agent(engine_cutoff_signal="probe")


def test_turning_lander_keeps_the_engine_until_steady_or_the_backup_height():
    """After the delay a lander turning at 3 deg/s keeps the engine; it stops when the rate is gone."""
    agent = make_agent(engine_stop_delay_after_probe_contact_s=0.1)
    turning = math.radians(3.0)
    for _ in range(20):
        command = agent.compute_command(observation(1.5, 1.0, roll_rate=turning), CONTROL_STEP)
        assert float(command.throttle[0]) > 0.0
    assert float(agent.compute_command(observation(1.4, 1.0), CONTROL_STEP).throttle[0]) == 0.0
    agent = make_agent(engine_stop_delay_after_probe_contact_s=0.1)
    assert float(agent.compute_command(observation(-0.01, 1.0, roll_rate=turning), CONTROL_STEP).throttle[0]) == 0.0
