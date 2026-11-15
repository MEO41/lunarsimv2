# SPDX-License-Identifier: BSD-3-Clause
"""Policies with memory (CPU, milestone 3): attention masking, exports equal the trained models."""

import pytest
import torch

pytest.importorskip("rsl_rl")
from rsl_rl.models import MLPModel, RNNModel  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from lunarsim.agents.rsl_rl.decision_history import DecisionHistory  # noqa: E402
from lunarsim.agents.rsl_rl.history_models import (  # noqa: E402
    HISTORY_GROUP,
    HISTORY_MASK_GROUP,
    HistoryAttentionModel,
    export_policy_module,
)

STEP_DIMENSION, HISTORY, ACTIONS, COUNT = 20, 8, 3, 5
DISTRIBUTION = {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"}


def observations(count=COUNT, real_steps=HISTORY):
    """Return random history groups with the first HISTORY - real_steps steps padded."""
    history = torch.randn(count, HISTORY, STEP_DIMENSION)
    mask = torch.ones(count, HISTORY)
    mask[:, : HISTORY - real_steps] = 0.0
    history = history * mask.unsqueeze(-1)
    return TensorDict({"policy": torch.randn(count, STEP_DIMENSION), HISTORY_GROUP: history.reshape(count, -1), HISTORY_MASK_GROUP: mask}, batch_size=[count])


def attention_model():
    """Return a small attention model with a few normaliser updates."""
    torch.manual_seed(0)
    model = HistoryAttentionModel(
        observations(), {"actor": [HISTORY_GROUP, HISTORY_MASK_GROUP]}, "actor", ACTIONS, hidden_dims=[32],
        distribution_cfg=dict(DISTRIBUTION), history_length=HISTORY, model_dimension=16, heads=4, layers=2, feedforward_dimension=32,
    )
    for _ in range(3):
        model.update_normalization(observations(real_steps=5))
    return model.eval()


def test_padded_steps_do_not_change_the_output():
    """Changing only the padded steps leaves the attention output unchanged."""
    model = attention_model()
    obs = observations(real_steps=3)
    changed = obs.clone()
    history = changed[HISTORY_GROUP].reshape(COUNT, HISTORY, STEP_DIMENSION)
    history[:, :5] = 100.0 * torch.randn(COUNT, 5, STEP_DIMENSION)
    changed[HISTORY_GROUP] = history.reshape(COUNT, -1)
    with torch.no_grad():
        assert torch.allclose(model(obs), model(changed), atol=1e-5)


def test_attention_export_equals_the_model():
    """The traced attention export gives the model's deterministic output."""
    model = attention_model()
    exported = export_policy_module(model, "attention")
    obs = observations(real_steps=4)
    with torch.no_grad():
        assert torch.allclose(exported(obs[HISTORY_GROUP], obs[HISTORY_MASK_GROUP]), model(obs), atol=1e-5)


def test_gru_export_keeps_one_hidden_state_per_lander():
    """Five steps through the GRU export equal five steps of the rsl_rl model, for five landers at once."""
    torch.manual_seed(1)
    obs = observations()
    model = RNNModel(obs, {"actor": ["policy"]}, "actor", ACTIONS, hidden_dims=[32], obs_normalization=True,
                     distribution_cfg=dict(DISTRIBUTION), rnn_type="gru", rnn_hidden_dim=16, rnn_num_layers=1).eval()
    exported = export_policy_module(model, "gru")
    hidden = torch.zeros(1, COUNT, 16)
    model.reset()
    for _ in range(5):
        step = TensorDict({"policy": torch.randn(COUNT, STEP_DIMENSION)}, batch_size=[COUNT])
        with torch.no_grad():
            expected = model(step)
            actions, hidden = exported(step["policy"], hidden)
        assert torch.allclose(actions, expected, atol=1e-5)


def test_mlp_export_equals_the_model():
    """The MLP export gives the model's deterministic output."""
    obs = observations()
    model = MLPModel(obs, {"actor": ["policy"]}, "actor", ACTIONS, hidden_dims=[32], obs_normalization=True, distribution_cfg=dict(DISTRIBUTION)).eval()
    with torch.no_grad():
        assert torch.allclose(export_policy_module(model, "mlp")(obs["policy"]), model(obs), atol=1e-6)


def test_decision_history_clears_restarted_environments():
    """A restart wipes the history: only the newest step is real afterwards."""
    history = DecisionHistory(4, 2, "cpu")
    for value in range(5):
        history.push(torch.full((2, 3), float(value)), torch.tensor([False, value == 4]))
    steps, mask = history.groups()
    assert mask[0].tolist() == [1, 1, 1, 1] and mask[1].tolist() == [0, 0, 0, 1]
    assert steps[0].reshape(4, 3)[:, 0].tolist() == [1, 2, 3, 4]
