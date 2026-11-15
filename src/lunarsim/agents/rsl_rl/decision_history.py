# SPDX-License-Identifier: BSD-3-Clause
"""History of decision-step observations for the attention policy (milestone 3); pure torch.

Shared by the training wrapper (``lander_vector_environment``) and the flying agent
(``ppo_policy_agent``), so both build exactly the same input.
"""

from __future__ import annotations

import torch


class DecisionHistory:
    """The last ``length`` decision-step observations per environment, oldest first, with a mask."""

    def __init__(self, length: int, count: int, device) -> None:
        """Start empty (all padding)."""
        self.length, self.count, self.device = length, count, device
        self.steps: torch.Tensor | None = None
        self.mask = torch.zeros(count, length, device=device)

    def push(self, observation: torch.Tensor, restarted: torch.Tensor) -> None:
        """Clear restarted environments, then append ``observation`` (N, d) as the newest step."""
        if self.steps is None:
            self.steps = torch.zeros(self.count, self.length, observation.shape[1], device=self.device)
        keep = (~restarted).float()
        self.steps = self.steps * keep.view(-1, 1, 1)
        self.mask = self.mask * keep.view(-1, 1)
        self.steps = torch.cat((self.steps[:, 1:], observation.unsqueeze(1)), dim=1)
        self.mask = torch.cat((self.mask[:, 1:], torch.ones(self.count, 1, device=self.device)), dim=1)

    def groups(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the flattened history (N, length * d) and the mask (N, length)."""
        return self.steps.reshape(self.count, -1), self.mask.clone()
