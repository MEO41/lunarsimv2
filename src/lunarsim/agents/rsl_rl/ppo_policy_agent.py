# SPDX-License-Identifier: BSD-3-Clause
"""``agent: ppo_mlp`` - flies a policy trained by ``scripts/train_ppo.py`` (milestone 1b).

The policy file is the exported actor (TorchScript, with rsl_rl's observation normaliser inside):
``runs/<training run>/policy.pt``. It gets the scaled observation and returns the mean action
(no exploration noise). The mapping to a Command is the one used in training
(:class:`~lunarsim.agents.rsl_rl.policy_action_adapter.PolicyActionAdapter`).

Parameters: ``policy_path`` (required for flying; relative paths are taken from the repository
root), ``action_mode``, ``maximum_tilt_deg`` and ``policy_period_steps`` (must match the
training run), and ``zero_body_rate_in_policy_input`` (default false; when true, the body rate
the policy sees is set to zero before scaling, while the attitude controller still gets the real
rate - an evaluation-only check of the observation channel asked for by the simulated review of
the transfer-study paper, item R20). The ``ppo:`` section of the agent file holds the training settings and is not used here.
"""

from __future__ import annotations

from pathlib import Path

import torch

from lunarsim.agents.agent_base import AgentBase
from lunarsim.agents.rsl_rl.decision_history import DecisionHistory
from lunarsim.agents.rsl_rl.policy_action_adapter import PolicyActionAdapter
from lunarsim.agents.rsl_rl.training_configuration import policy_kind
from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import Command, NavigationStateLayout, Observation
from lunarsim.core.registry import register_binding

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


@register_binding(SlotName.AGENT, "ppo_mlp")
class ProximalPolicyOptimizationMultilayerPerceptronAgent(AgentBase):
    """Runs an exported PPO actor on the observation."""

    def __init__(self, parameters, number_of_environments, number_of_jets, device, shared_data=None) -> None:
        """Load the exported policy and build the action adapter."""
        super().__init__(parameters, number_of_environments, number_of_jets, device, shared_data)
        self.adapter = PolicyActionAdapter.from_agent_parameters(self.shared_data["lander_preset"], device, self.parameters)
        policy_path = self.parameters.get("policy_path")
        if not policy_path:
            raise ValueError("agent ppo_mlp needs policy_path=<runs/.../policy.pt> to fly")
        path = Path(policy_path)
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        self.policy = torch.jit.load(str(path), map_location=device)
        self.policy.eval()
        # The policy decides every policy_period_steps control steps, as in training; between
        # decisions the attitude controller keeps flying the held action.
        self.policy_period_steps = int(self.parameters.get("policy_period_steps", 1))
        self.zero_body_rate_in_policy_input = bool(self.parameters.get("zero_body_rate_in_policy_input", False))
        self.steps_since_decision = torch.zeros(number_of_environments, dtype=torch.long, device=device)
        self.held_actions = torch.zeros(number_of_environments, self.adapter.action_dimension, device=device)
        ppo = dict(self.parameters.get("ppo") or {})
        self.kind = policy_kind(ppo)
        self.restarted = torch.ones(number_of_environments, dtype=torch.bool, device=device)
        if self.kind == "gru":
            recurrent = ppo["recurrent"]
            self.hidden_state = torch.zeros(int(recurrent["rnn_num_layers"]), number_of_environments, int(recurrent["rnn_hidden_dim"]), device=device)
        if self.kind == "attention":
            self.history = DecisionHistory(int(ppo["attention"]["history_length"]), number_of_environments, device)

    def reset(self, env_ids, gen) -> None:
        """Make the next control step a decision step; clear the memory (hidden state or history)."""
        self.steps_since_decision[env_ids] = 0
        self.restarted[env_ids] = True
        if self.kind == "gru":
            self.hidden_state[:, env_ids] = 0.0

    def compute_command(self, observation: Observation, control_step_duration: float) -> Command:
        """Return the Command of the policy's mean action (new at decision steps, held between them)."""
        decide = self.steps_since_decision % self.policy_period_steps == 0
        if bool(decide.any()):
            scaled = self.adapter.scale_observation(self._policy_input(observation.policy))
            with torch.no_grad():
                if self.kind == "gru":
                    actions, hidden_state = self.policy(scaled, self.hidden_state)
                    self.hidden_state = torch.where(decide.view(1, -1, 1), hidden_state, self.hidden_state)
                elif self.kind == "attention":
                    # Every environment's history advances only at its own decision steps.
                    previous_steps, previous_mask = (None, None) if self.history.steps is None else (self.history.steps.clone(), self.history.mask.clone())
                    self.history.push(scaled, self.restarted & decide)
                    if previous_steps is not None:
                        self.history.steps = torch.where(decide.view(-1, 1, 1), self.history.steps, previous_steps)
                        self.history.mask = torch.where(decide.view(-1, 1), self.history.mask, previous_mask)
                    actions = self.policy(*self.history.groups())
                else:
                    actions = self.policy(scaled)
            self.held_actions = torch.where(decide.unsqueeze(1), actions, self.held_actions)
            self.restarted &= ~decide
        self.steps_since_decision += 1
        return self.adapter.command_from_policy_actions(self.held_actions, observation.policy, control_step_duration)

    def _policy_input(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the raw observation the policy sees: unchanged, or with a zero body rate when ``zero_body_rate_in_policy_input`` is set."""
        if not self.zero_body_rate_in_policy_input:
            return observation
        masked = observation.clone()
        masked[:, NavigationStateLayout.ANGULAR_VELOCITY] = 0.0
        return masked
