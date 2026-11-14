# SPDX-License-Identifier: BSD-3-Clause
"""``faults: none`` - measurements pass through unchanged."""

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.registry import register_binding


@register_binding(SlotName.FAULTS, "none")
class NoFaults(BindingBase):
    """Changes nothing."""
