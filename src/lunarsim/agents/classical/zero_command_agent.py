# SPDX-License-Identifier: BSD-3-Clause
"""``agent: zero`` - always commands zero throttle and no jets. Used by the null loop."""

import torch

from lunarsim.agents.agent_base import AgentBase
from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import Command, Observation
from lunarsim.core.registry import register_binding


@register_binding(SlotName.AGENT, "zero")
class ZeroCommandAgent(AgentBase):
    """Outputs an all-zero command."""

    def compute_command(self, observation: Observation, control_step_duration: float) -> Command:
        """Return zero throttle, zero gimbal and all jets off."""
        count = observation.policy.shape[0]
        return Command(
            throttle=torch.zeros(count, device=self.device),
            gimbal=torch.zeros(count, 2, device=self.device),
            reaction_control_jets=torch.zeros(count, self.number_of_jets, device=self.device),
        )
