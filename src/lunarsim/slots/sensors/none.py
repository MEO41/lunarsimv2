# SPDX-License-Identifier: BSD-3-Clause
"""``sensors: none`` - the sensor chain is not built (truth navigation is used instead)."""

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.registry import register_binding


@register_binding(SlotName.SENSORS, "none")
class NoSensors(BindingBase):
    """Produces no measurements."""
