# SPDX-License-Identifier: BSD-3-Clause
"""``actuators: none`` - the command is ignored; the lander gets no force."""

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import MessageKey, Wrench
from lunarsim.core.registry import register_binding


@register_binding(SlotName.ACTUATORS, "none")
class NoActuators(BindingBase):
    """Writes a zero wrench every substep."""

    requires = frozenset({MessageKey.COMMAND})
    provides = frozenset({MessageKey.ACTUATOR_WRENCH})
    number_of_jets = 0

    def step(self, bus, dt: float) -> None:
        """Write a zero actuator wrench."""
        bus.write(MessageKey.ACTUATOR_WRENCH, Wrench.zeros(self.number_of_environments, self.device), self)
