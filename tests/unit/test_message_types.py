# SPDX-License-Identifier: BSD-3-Clause
"""Message types: shapes, batch dimension first, dtype checks."""

import pytest
import torch
from helpers import make_truth_state

from lunarsim.core.message_types import Command, Frame, Measurement, NavEstimate, NavigationStateLayout, Wrench


def test_truth_state_reports_batch_size_and_frame():
    """A valid TruthState keeps N as the first dimension of every field."""
    state = make_truth_state(number_of_environments=7)
    assert state.number_of_environments == 7
    assert state.frame is Frame.SITE
    assert state.contact.shape == (7, 4)


def test_truth_state_rejects_wrong_shape_and_missing_frame():
    """Wrong shapes and a string frame are errors, not silent bugs."""
    state = make_truth_state()
    with pytest.raises(ValueError):
        type(state)(**{**state.__dict__, "position": torch.zeros(4, 2)})
    with pytest.raises(ValueError):
        type(state)(**{**state.__dict__, "mass": torch.zeros(5)})
    with pytest.raises(TypeError):
        type(state)(**{**state.__dict__, "frame": "SITE"})


def test_measurement_checks_valid_flag_and_covariance():
    """``valid`` must be bool (N,), covariance (N, d) or (N, d, d)."""
    Measurement(torch.zeros(3, 2), torch.ones(3, dtype=torch.bool), torch.zeros(3), torch.zeros(3, 2))
    Measurement(torch.zeros(3, 2), torch.ones(3, dtype=torch.bool), torch.zeros(3), torch.zeros(3, 2, 2))
    with pytest.raises(ValueError):
        Measurement(torch.zeros(3, 2), torch.ones(3), torch.zeros(3), torch.zeros(3, 2))
    with pytest.raises(ValueError):
        Measurement(torch.zeros(3, 2), torch.ones(3, dtype=torch.bool), torch.zeros(3), torch.zeros(3, 5))


def test_command_action_tensor_round_trip_and_clamp():
    """The flat action layout is throttle, 2 gimbal angles, jets; values are clamped to 0..1."""
    actions = torch.tensor([[1.5, 0.1, -0.1, -1.0, 0.5, 2.0]])
    command = Command.from_action_tensor(actions)
    assert command.throttle.tolist() == [1.0]
    assert command.reaction_control_jets.tolist() == [[0.0, 0.5, 1.0]]
    assert command.to_action_tensor().shape == (1, Command.action_dimension(number_of_jets=3))
    assert torch.equal(Command.from_action_tensor(command.to_action_tensor()).gimbal, command.gimbal)


def test_wrench_fills_zero_mass_flow():
    """A wrench without a mass flow gets zeros of shape (N,)."""
    wrench = Wrench(force=torch.ones(5, 3), torque=torch.zeros(5, 3))
    assert wrench.mass_flow_rate.shape == (5,)
    assert float(wrench.mass_flow_rate.abs().sum()) == 0.0
    with pytest.raises(ValueError):
        Wrench(force=torch.ones(5, 3), torque=torch.zeros(5, 2))


def test_navigation_layout_covers_all_columns_once():
    """The layout slices tile 0..DIMENSION without gaps or overlap."""
    layout = NavigationStateLayout
    columns = []
    for part in (
        layout.POSITION,
        layout.VELOCITY,
        layout.ORIENTATION_QUATERNION,
        layout.ANGULAR_VELOCITY,
        layout.MASS,
        layout.HEIGHT_ABOVE_GROUND,
        layout.SURFACE_PROBE_CONTACT,
    ):
        columns += list(range(part.start, part.stop))
    assert columns == list(range(layout.DIMENSION))
    NavEstimate(mean=torch.zeros(2, layout.DIMENSION), covariance=torch.zeros(2, layout.DIMENSION), health=torch.zeros(2, 0))
