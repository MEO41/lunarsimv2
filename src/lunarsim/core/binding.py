# SPDX-License-Identifier: BSD-3-Clause
"""The one interface that every binding implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
from torch import Tensor

if TYPE_CHECKING:
    from lunarsim.core.message_bus import MessageBus
    from lunarsim.core.simulation_clock import SimulationClock


class SlotName:
    """The nine slot names, spelled as in scenario files."""

    START = "start"
    PLANT = "plant"
    SURFACE = "surface"
    SENSORS = "sensors"
    FAULTS = "faults"
    NAVIGATION = "nav"
    AGENT = "agent"
    ACTUATORS = "actuators"
    METRICS = "metrics"

    ALL = (START, PLANT, SURFACE, SENSORS, FAULTS, NAVIGATION, AGENT, ACTUATORS, METRICS)
    # Slots that live inside LanderEnv. The agent sits outside, because training libraries own it.
    INSIDE_ENVIRONMENT = (START, PLANT, SURFACE, SENSORS, FAULTS, NAVIGATION, ACTUATORS, METRICS)


@runtime_checkable
class Binding(Protocol):
    """Interface from the design document.

    ``slot`` is one of the nine slot names. ``requires`` and ``provides`` are the message keys
    that the binding reads and writes. ``reads_truth`` must be True for a binding that reads
    ``TruthState``.
    """

    slot: str
    requires: frozenset[str]
    provides: frozenset[str]
    reads_truth: bool

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Reset the internal state of the given environments, drawing only from ``gen``."""
        ...

    def step(self, bus: MessageBus, dt: float) -> None:
        """Read the required messages from the bus and write the provided ones."""
        ...


@dataclass
class BindingContext:
    """What the environment hands to every binding when it builds it."""

    number_of_environments: int
    device: torch.device | str
    clock: SimulationClock
    bus: MessageBus
    # Objects that only exist inside a running simulator (articulation, sensors, the env itself).
    # Bindings that are pure torch never touch this.
    simulation_handles: dict[str, Any] = field(default_factory=dict)
    # Scenario-wide data that several slots share (for example the lander preset).
    shared_data: dict[str, Any] = field(default_factory=dict)


class BindingBase:
    """Convenience base class with the default behaviour of a binding.

    Subclasses set the class attributes and override :meth:`reset` and :meth:`step`.
    """

    slot: str = ""
    registry_name: str = ""
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset()
    reads_truth: bool = False

    def __init__(self, parameters: dict[str, Any], context: BindingContext) -> None:
        """Store the scenario parameters of this binding and the shared context."""
        self.parameters = dict(parameters)
        self.context = context
        self.number_of_environments = context.number_of_environments
        self.device = context.device

    def setup_scene(self, environment: Any) -> None:
        """Create simulator objects (assets, terrain). Called once, before the simulation starts."""

    def initialize(self, environment: Any) -> None:
        """Allocate tensors that need the running simulation. Called once, after it starts."""

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Reset the internal state of the given environments."""

    def begin_control_step(self, bus: MessageBus, dt: float) -> None:
        """Called once per control step in ``_pre_physics_step``, before the PhysX substeps."""

    def step(self, bus: MessageBus, dt: float) -> None:
        """Read the required messages from the bus and write the provided ones."""
