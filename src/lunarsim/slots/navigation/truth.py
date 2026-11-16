# SPDX-License-Identifier: BSD-3-Clause
"""``nav: truth`` - the navigation estimate is the true state (the truth bypass)."""

import torch

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import MessageKey, NavEstimate, NavigationStateLayout, TruthState
from lunarsim.core.registry import register_binding


@register_binding(SlotName.NAVIGATION, "truth")
class TruthNavigation(BindingBase):
    """Copies TruthState into NavEstimate with zero covariance.

    This is the only binding on the flight-software side that may read TruthState.
    """

    requires = frozenset({MessageKey.TRUTH_STATE})
    provides = frozenset({MessageKey.NAVIGATION_ESTIMATE})
    reads_truth = True
    health_dimension = 0  # no sensors, no health columns

    def step(self, bus, dt: float) -> None:
        """Write the estimate: position, velocity, quaternion, body rates, mass, height above ground, probe contact."""
        truth: TruthState = bus.read(MessageKey.TRUTH_STATE, self)
        bus.write(MessageKey.NAVIGATION_ESTIMATE, self.truth_estimate(truth), self)

    @staticmethod
    def truth_estimate(truth: TruthState) -> NavEstimate:
        """Return the true state in NavigationStateLayout with zero covariance."""
        mean = torch.cat(
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
        return NavEstimate(
            mean=mean,
            covariance=torch.zeros(mean.shape[0], NavigationStateLayout.DIMENSION, device=mean.device),
            health=torch.zeros(mean.shape[0], 0, device=mean.device),
        )
