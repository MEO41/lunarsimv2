# SPDX-License-Identifier: BSD-3-Clause
"""``metrics: none`` - no episode ends by an event; only the time limit ends episodes."""

import torch

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import EpisodeEvents, MessageKey
from lunarsim.core.registry import register_binding


@register_binding(SlotName.METRICS, "none")
class NoMetrics(BindingBase):
    """Reports that no environment has terminated."""

    provides = frozenset({MessageKey.EPISODE_EVENTS})

    def step(self, bus, dt: float) -> None:
        """Write an all-False termination message."""
        never = torch.zeros(self.number_of_environments, dtype=torch.bool, device=self.device)
        bus.write(MessageKey.EPISODE_EVENTS, EpisodeEvents(terminated=never, truncated=never.clone()), self)
