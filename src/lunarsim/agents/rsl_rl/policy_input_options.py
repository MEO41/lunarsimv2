# SPDX-License-Identifier: BSD-3-Clause
"""Options that change what the PPO policy sees, shared by training and evaluation.

Both are observation-channel controls of the analytic-to-PhysX transfer study, asked for by the
second simulated review of the paper (items R8 and R9, 2026-11-26). Training
(:class:`~lunarsim.agents.rsl_rl.lander_vector_environment.LanderVectorEnvironment`) and flying
(``agent: ppo_mlp``) build the policy input with the same functions, so a policy trained with an
option sees the same input when it is evaluated with it.

- ``zero_body_rate_in_policy_input`` (default false): the three body-rate values of the policy
  observation are set to zero before scaling. Only the policy input changes: the reward, the
  critic, the touchdown metric and the attitude controller still get the plant's rate.
- ``last_action_in_policy_input`` (default false): the policy's previous action, clipped to
  [-1, 1], is appended to the scaled observation (zero at the first decision of an episode), so
  the policy knows the direction it last commanded.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.message_types import NavigationStateLayout


class PolicyInputOptions:
    """Reads the two options from the agent parameters and applies them to a policy input."""

    def __init__(self, zero_body_rate: bool = False, last_action: bool = False) -> None:
        """Store which of the two options are on."""
        self.zero_body_rate = bool(zero_body_rate)
        self.last_action = bool(last_action)

    @classmethod
    def from_agent_parameters(cls, parameters: dict) -> PolicyInputOptions:
        """Build the options from an ``agent: ppo_mlp`` parameter dictionary."""
        return cls(
            zero_body_rate=parameters.get("zero_body_rate_in_policy_input", False),
            last_action=parameters.get("last_action_in_policy_input", False),
        )

    def raw_policy_observation(self, observation: Tensor) -> Tensor:
        """Return the unscaled observation the policy sees: unchanged, or with a zero body rate."""
        if not self.zero_body_rate:
            return observation
        masked = observation.clone()
        masked[:, NavigationStateLayout.ANGULAR_VELOCITY] = 0.0
        return masked

    def append_last_action(self, scaled_observation: Tensor, last_actions: Tensor) -> Tensor:
        """Return the scaled observation, followed by the clipped previous action when that option is on."""
        if not self.last_action:
            return scaled_observation
        return torch.cat((scaled_observation, last_actions.clamp(-1.0, 1.0).to(scaled_observation.dtype)), dim=1)
