# SPDX-License-Identifier: BSD-3-Clause
"""Policy action adapter (CPU): throttle mapping, attitude and axis modes, observation scaling."""

import pytest
import torch

from lunarsim.agents.rsl_rl.policy_action_adapter import PolicyActionAdapter
from lunarsim.assets.lander_preset import load_lander_preset
from lunarsim.core.message_types import NavigationStateLayout
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.env.scenario_loader import default_config_directory

PRESET = load_lander_preset(default_config_directory() / "landers" / "apollo_lunar_module.yaml")


def upright_observation(count=1, roll_rate=0.0):
    """Return an upright observation (only orientation and rates matter for the jets)."""
    state = torch.zeros(count, NavigationStateLayout.DIMENSION)
    state[:, NavigationStateLayout.ORIENTATION_QUATERNION] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    state[:, NavigationStateLayout.ANGULAR_VELOCITY] = torch.tensor([roll_rate, 0.0, 0.0])
    return state


def test_throttle_is_off_at_minus_one_hover_at_zero_maximum_at_one():
    """-1 -> 0 (engine off), 0 -> hover at the start mass, +1 -> 0.65."""
    adapter = PolicyActionAdapter(PRESET, "cpu")
    hover = PRESET.mass_at_start * MOON_SURFACE_GRAVITY / PRESET.engine_rated_thrust
    actions = torch.tensor([[-1.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    command = adapter.command_from_policy_actions(actions, upright_observation(4), 0.02)
    assert command.throttle.tolist() == pytest.approx([0.0, hover, 0.65, 0.65])


def test_attitude_mode_fires_jets_to_reach_the_wanted_tilt():
    """Upright with zero action: no jets. Full +x tilt: the y-axis couple fires."""
    adapter = PolicyActionAdapter(PRESET, "cpu")
    quiet = adapter.command_from_policy_actions(torch.zeros(1, 3), upright_observation(), 0.02)
    assert float(quiet.reaction_control_jets.sum()) == 0.0
    tilt = adapter.command_from_policy_actions(torch.tensor([[0.0, 1.0, 0.0]]), upright_observation(), 0.02)
    assert float(tilt.reaction_control_jets.sum()) > 0.0


def test_axis_mode_has_four_actions_and_uses_the_couple_table():
    """axis_commands: action 1 = +1 fires the positive x couple at full duty."""
    adapter = PolicyActionAdapter(PRESET, "cpu", action_mode="axis_commands")
    assert adapter.action_dimension == 4
    command = adapter.command_from_policy_actions(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), upright_observation(), 0.02)
    expected = adapter.jet_table.jets_for_axis_commands(torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.equal(command.reaction_control_jets, expected)


def test_observation_scaling_and_unknown_mode():
    """Mass 7300 kg scales to 1.0, height 20 m to 1.0; an unknown mode is rejected."""
    adapter = PolicyActionAdapter(PRESET, "cpu")
    state = upright_observation()
    state[:, NavigationStateLayout.MASS] = 7300.0
    state[:, NavigationStateLayout.HEIGHT_ABOVE_GROUND] = 20.0
    scaled = adapter.scale_observation(state)
    assert float(scaled[0, NavigationStateLayout.MASS]) == pytest.approx(1.0)
    assert float(scaled[0, NavigationStateLayout.HEIGHT_ABOVE_GROUND]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        PolicyActionAdapter(PRESET, "cpu", action_mode="jets")
