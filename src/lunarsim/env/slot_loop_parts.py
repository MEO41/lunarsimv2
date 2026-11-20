# SPDX-License-Identifier: BSD-3-Clause
"""Parts of the slot loop that every environment class shares, with no Isaac Lab import.

``LanderEnv`` (PhysX, inside Isaac Lab) and ``AnalyticLanderEnvironment`` (pure torch) run the
same slot loop. The reset order, the ports through which an environment reads and writes the
bus, and the critic's truth vector live here once, so the two environments cannot drift apart.
"""

from __future__ import annotations

import torch

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey

# Slots that reset after the start slot, in this order.
# The surface comes first: it may move the site of an environment, and the plant places the
# lander relative to that site.
SLOT_RESET_ORDER = (
    SlotName.SURFACE,
    SlotName.PLANT,
    SlotName.SENSORS,
    SlotName.FAULTS,
    SlotName.NAVIGATION,
    SlotName.ACTUATORS,
    SlotName.METRICS,
)


class _AgentPort:
    """Lets the environment write the agent's Command to the bus on the agent's behalf."""

    slot = SlotName.AGENT
    requires = frozenset({MessageKey.OBSERVATION})
    provides = frozenset({MessageKey.COMMAND})
    reads_truth = False


class _EpisodeEventsReader:
    """Lets the environment read the metrics output to fill the done flags."""

    requires = frozenset({MessageKey.EPISODE_EVENTS})
    provides = frozenset()
    reads_truth = False


class _TrainingTruthReader:
    """Training-side reader of TruthState (reward and critic group); its truth reads are logged."""

    requires = frozenset({MessageKey.TRUTH_STATE, MessageKey.EPISODE_EVENTS})
    provides = frozenset()
    reads_truth = True


def truth_observation_vector(truth) -> torch.Tensor:
    """Return the critic's view of the truth in the NavigationStateLayout order, (N, 16)."""
    return torch.cat(
        (
            truth.position,
            truth.velocity,
            truth.orientation_quaternion,
            truth.angular_velocity,
            truth.mass.unsqueeze(1),
            truth.footpad_height_above_ground.unsqueeze(1),
            truth.surface_probe_contact.unsqueeze(1),
        ),
        dim=1,
    ).to(torch.float32)
