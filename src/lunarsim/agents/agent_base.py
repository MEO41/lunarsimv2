# SPDX-License-Identifier: BSD-3-Clause
"""Base class of agents that run outside the environment."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import Command, MessageKey, Observation


class AgentBase:
    """An agent maps an Observation to a Command. It never sees TruthState.

    The agent follows the same declaration as a binding (slot, requires, provides, reads_truth),
    so the registry can check the loop, but it is called by the run script and not by the
    environment.
    """

    slot: str = SlotName.AGENT
    registry_name: str = ""
    requires: frozenset[str] = frozenset({MessageKey.OBSERVATION})
    provides: frozenset[str] = frozenset({MessageKey.COMMAND})
    reads_truth: bool = False

    def __init__(
        self,
        parameters: dict[str, Any],
        number_of_environments: int,
        number_of_jets: int,
        device: torch.device | str,
        shared_data: dict[str, Any] | None = None,
    ) -> None:
        """Store the scenario parameters and the sizes the agent needs to build commands."""
        self.parameters = dict(parameters)
        self.number_of_environments = number_of_environments
        self.number_of_jets = number_of_jets
        self.device = device
        self.shared_data = shared_data or {}

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Reset the internal state of the given environments."""

    def compute_command(self, observation: Observation, control_step_duration: float) -> Command:
        """Return the command for the current observation."""
        raise NotImplementedError
