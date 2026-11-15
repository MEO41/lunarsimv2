# SPDX-License-Identifier: BSD-3-Clause
"""``agent: ppo_mlp`` - flies a policy trained by ``scripts/train_ppo.py`` (milestone 1b).

The policy file is the exported actor (TorchScript, with rsl_rl's observation normaliser inside):
``runs/<training run>/policy.pt``. It gets the scaled observation and returns the mean action
(no exploration noise). The mapping to a Command is the one used in training
(:class:`~lunarsim.agents.rsl_rl.policy_action_adapter.PolicyActionAdapter`).

Parameters: ``policy_path`` (required for flying; relative paths are taken from the repository
root), ``action_mode``, ``maximum_tilt_deg`` and ``policy_period_steps`` (must match the
training run). The ``ppo:`` section of the agent file holds the training settings and is not used here.
"""

from __future__ import annotations

from pathlib import Path

import torch

from lunarsim.agents.agent_base import AgentBase
from lunarsim.agents.rsl_rl.policy_action_adapter import PolicyActionAdapter
from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import Command, Observation
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
        self.steps_since_decision = torch.zeros(number_of_environments, dtype=torch.long, device=device)
        self.held_actions = torch.zeros(number_of_environments, self.adapter.action_dimension, device=device)

    def reset(self, env_ids, gen) -> None:
        """Make the next control step of the given environments a decision step."""
        self.steps_since_decision[env_ids] = 0

    def compute_command(self, observation: Observation, control_step_duration: float) -> Command:
        """Return the Command of the policy's mean action (new at decision steps, held between them)."""
        decide = self.steps_since_decision % self.policy_period_steps == 0
        if bool(decide.any()):
            with torch.no_grad():
                actions = self.policy(self.adapter.scale_observation(observation.policy))
            self.held_actions = torch.where(decide.unsqueeze(1), actions, self.held_actions)
        self.steps_since_decision += 1
        return self.adapter.command_from_policy_actions(self.held_actions, observation.policy, control_step_duration)
