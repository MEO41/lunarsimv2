# SPDX-License-Identifier: BSD-3-Clause
"""Policy networks with memory for milestone 3 (design: ``ppo_gru``, ``ppo_attention``), and exports.

- :class:`HistoryAttentionModel` (an rsl_rl ``MLPModel``): reads the observation groups
  ``policy_history`` (the last ``history_length`` scaled observations, oldest first, flattened)
  and ``policy_history_mask`` (1 for real steps, 0 for padding before the episode start). Each
  step is normalised per feature (running mean and variance of real steps only), projected to
  ``model_dimension``, given a learned position embedding, and passed through a transformer
  encoder (``layers`` layers, ``heads`` heads) that ignores padded steps. The newest step's output
  is the latent of the usual MLP head and Gaussian action distribution, so PPO, the distribution
  and the training loop are exactly those of the MLP and GRU policies.
- The GRU policy is rsl_rl's own ``RNNModel`` with ``rnn_type: gru``.
- ``export_*``: TorchScript modules for flying many landers at once (rsl_rl's own GRU export keeps
  one hidden state for a batch of one): the MLP export maps an observation to an action; the GRU
  export maps (observation, hidden state) to (action, new hidden state); the attention export
  maps (history, mask) to an action.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.normalization import EmpiricalNormalization
from tensordict import TensorDict
from torch import nn

HISTORY_GROUP = "policy_history"
HISTORY_MASK_GROUP = "policy_history_mask"


class HistoryAttentionModel(MLPModel):
    """Transformer encoder over the observation history, then the MLP head of rsl_rl."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128,),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        history_length: int = 32,
        model_dimension: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward_dimension: int = 128,
    ) -> None:
        """Build the per-step normaliser, the projection, the position embedding and the encoder."""
        self.latent_dim = model_dimension
        self.history_length = history_length
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, False, distribution_cfg)
        if self.obs_groups != [HISTORY_GROUP, HISTORY_MASK_GROUP]:
            raise ValueError(f"HistoryAttentionModel needs the groups [{HISTORY_GROUP}, {HISTORY_MASK_GROUP}], got {self.obs_groups}")
        self.step_dimension = (self.obs_dim - history_length) // history_length
        self.obs_normalization = obs_normalization
        self.step_normalizer = EmpiricalNormalization(self.step_dimension) if obs_normalization else nn.Identity()
        self.input_projection = nn.Linear(self.step_dimension, model_dimension)
        self.position_embedding = nn.Parameter(0.02 * torch.randn(history_length, model_dimension))
        layer = nn.TransformerEncoderLayer(
            model_dimension, heads, feedforward_dimension, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def _get_latent_dim(self) -> int:
        """Return the width of the MLP head's input: the encoder's model dimension."""
        return self.latent_dim

    def split(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the history (N, H, d) and the mask of real steps (N, H) bool."""
        history = obs[HISTORY_GROUP].reshape(-1, self.history_length, self.step_dimension)
        mask = obs[HISTORY_MASK_GROUP] > 0.5
        return history, mask

    def encode(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the newest step's encoder output (N, model dimension)."""
        tokens = self.input_projection(self.step_normalizer(history)) + self.position_embedding
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        return encoded[:, -1]

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        """Return the latent of the MLP head."""
        history, mask = self.split(obs)
        return self.encode(history, mask)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update the per-step normaliser with the real (not padded) steps only."""
        if self.obs_normalization:
            history, mask = self.split(obs)
            self.step_normalizer.update(history[mask])

    def as_jit(self) -> nn.Module:
        """Return the TorchScript-ready attention export."""
        return AttentionPolicyExport(self)


class MultilayerPerceptronPolicyExport(nn.Module):
    """observation (N, d) -> deterministic action (N, a)."""

    def __init__(self, model: MLPModel) -> None:
        """Copy the normaliser, the network and the deterministic output of an MLP model."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = model.distribution.as_deterministic_output_module()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the mean action."""
        return self.deterministic_output(self.mlp(self.obs_normalizer(observation)))


class RecurrentPolicyExport(nn.Module):
    """(observation (N, d), hidden state (L, N, h)) -> (action (N, a), new hidden state)."""

    def __init__(self, model) -> None:
        """Copy the normaliser, the GRU, the network and the deterministic output of an RNN model."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = model.distribution.as_deterministic_output_module()

    def forward(self, observation: torch.Tensor, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean action and the new hidden state."""
        output, new_hidden_state = self.rnn(self.obs_normalizer(observation).unsqueeze(0), hidden_state)
        return self.deterministic_output(self.mlp(output.squeeze(0))), new_hidden_state


class AttentionPolicyExport(nn.Module):
    """(history (N, H * d), mask (N, H)) -> deterministic action (N, a)."""

    def __init__(self, model: HistoryAttentionModel) -> None:
        """Copy every part of the attention model."""
        super().__init__()
        self.history_length = model.history_length
        self.step_dimension = model.step_dimension
        self.step_normalizer = copy.deepcopy(model.step_normalizer)
        self.input_projection = copy.deepcopy(model.input_projection)
        self.position_embedding = nn.Parameter(model.position_embedding.detach().clone())
        self.encoder = copy.deepcopy(model.encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = model.distribution.as_deterministic_output_module()

    def forward(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the mean action."""
        steps = history.reshape(history.shape[0], self.history_length, self.step_dimension)
        tokens = self.input_projection(self.step_normalizer(steps)) + self.position_embedding
        encoded = self.encoder(tokens, src_key_padding_mask=mask < 0.5)
        return self.deterministic_output(self.mlp(encoded[:, -1]))


def export_policy_module(actor, kind: str) -> torch.jit.ScriptModule:
    """Return the TorchScript export (on the CPU) of a trained actor of the given kind (mlp, gru, attention).

    The export modules copy only the parts they need (normaliser, networks, deterministic
    output); the trained actor itself is not copied, because after training it holds tensors
    that cannot be copied (recurrent hidden state, distribution parameters).
    """
    with torch.no_grad():
        if kind == "mlp":
            return torch.jit.script(MultilayerPerceptronPolicyExport(actor).to("cpu").eval())
        if kind == "gru":
            return torch.jit.script(RecurrentPolicyExport(actor).to("cpu").eval())
        if kind == "attention":
            module = AttentionPolicyExport(actor).to("cpu").eval()
            history = torch.zeros(2, module.history_length * module.step_dimension)
            mask = torch.ones(2, module.history_length)
            mask[0, :5] = 0.0
            return torch.jit.trace(module, (history, mask), check_trace=False)
    raise ValueError(f"unknown policy kind '{kind}'")
