# SPDX-License-Identifier: BSD-3-Clause
"""The binding registry and the checks that run before every run."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey


class RegistryError(ValueError):
    """A scenario names a binding that does not exist, or a registration is wrong."""


class SlotLoopError(ValueError):
    """The loop of bindings is not closed, or a binding breaks the truth-access rule."""


# Slots that model the physical world. They work with the true state by nature.
# The faults slot models the physical world (a sensor that breaks at some altitude), so it may
# read the truth like the sensors (decision 2026-11-15, milestone 3).
TRUTH_SIDE_SLOTS = frozenset(
    {SlotName.START, SlotName.PLANT, SlotName.SURFACE, SlotName.SENSORS, SlotName.FAULTS, SlotName.METRICS}
)
# The only bindings on the flight-software side that may read TruthState: ``nav: truth`` and its
# S6 variant with the true hazard map (design S6: "Set nav: truth with the true hazard map").
TRUTH_NAVIGATION_NAME = "truth"
TRUTH_NAVIGATION_NAMES = frozenset({TRUTH_NAVIGATION_NAME, "truth_with_hazard_map"})

# Order in which the slots run during one control step, starting at a reset.
# The plant appears twice: it first reports the state, and later takes the wrench.
SLOT_STEP_ORDER = (
    SlotName.START,
    SlotName.PLANT,
    SlotName.SENSORS,
    SlotName.FAULTS,
    SlotName.NAVIGATION,
    "observation_builder",  # part of the environment: NavEstimate -> Observation
    SlotName.AGENT,
    SlotName.ACTUATORS,
    SlotName.SURFACE,
    SlotName.PLANT,
    SlotName.METRICS,
)


# Keys that their provider writes during initialisation and at every reset, before any step
# runs (the surface's site origins and ground lookup). Readers anywhere in the step may use them.
KEYS_WRITTEN_BEFORE_THE_FIRST_STEP = frozenset({MessageKey.SURFACE_GEOMETRY})


class ObservationBuilderDeclaration:
    """Declares what the environment's observation builder reads and writes, for the loop check."""

    slot = "observation_builder"
    registry_name = "environment"
    requires = frozenset({MessageKey.NAVIGATION_ESTIMATE})
    provides = frozenset({MessageKey.OBSERVATION})
    reads_truth = False


_BINDING_CLASSES: dict[str, dict[str, type]] = {slot: {} for slot in SlotName.ALL}


def register_binding(slot: str, name: str) -> Callable[[type], type]:
    """Class decorator: register a binding class under ``slot`` and ``name``.

    ``name`` is the name used in scenario files, for example ``nav: truth``.
    """

    def decorator(binding_class: type) -> type:
        """Store the class in the registry and stamp its slot and name on it."""
        if slot not in _BINDING_CLASSES:
            raise RegistryError(f"Unknown slot '{slot}'. Slots are {SlotName.ALL}")
        if name in _BINDING_CLASSES[slot] and _BINDING_CLASSES[slot][name] is not binding_class:
            raise RegistryError(f"Binding '{name}' is already registered in slot '{slot}'")
        binding_class.slot = slot
        binding_class.registry_name = name
        _BINDING_CLASSES[slot][name] = binding_class
        return binding_class

    return decorator


def get_binding_class(slot: str, name: str) -> type:
    """Return the binding class registered under ``slot`` and ``name``."""
    if slot not in _BINDING_CLASSES:
        raise RegistryError(f"Unknown slot '{slot}'. Slots are {SlotName.ALL}")
    if name not in _BINDING_CLASSES[slot]:
        raise RegistryError(
            f"No binding '{name}' in slot '{slot}'. Registered: {sorted(_BINDING_CLASSES[slot]) or 'none'}"
        )
    return _BINDING_CLASSES[slot][name]


def registered_binding_names(slot: str) -> list[str]:
    """Return the sorted names of all bindings registered in ``slot``."""
    return sorted(_BINDING_CLASSES[slot])


def check_slot_loop(bindings: Mapping[str, Any]) -> None:
    """Check one binding per slot before a run. Raise :class:`SlotLoopError` on the first problem.

    ``bindings`` maps each of the nine slot names to a binding class or instance. Three checks:

    1. Every slot has a binding, and the binding says it belongs to that slot.
    2. Requires/provides: walking the step order, every key that a binding requires has a
       provider earlier in the walk. A slot that appears twice (the plant) provides from its
       first position and has its requirements checked at its last position. Keys in
       ``KEYS_WRITTEN_BEFORE_THE_FIRST_STEP`` count as available from the start.
    3. Truth access: a binding that requires ``TruthState`` sets ``reads_truth=True``, and only
       truth-side slots (start, plant, surface, sensors, metrics) and ``nav: truth`` may set it.
    """
    all_bindings = dict(bindings)
    all_bindings["observation_builder"] = ObservationBuilderDeclaration

    for slot in SlotName.ALL:
        if slot not in all_bindings:
            raise SlotLoopError(f"Slot '{slot}' has no binding")
        if all_bindings[slot].slot != slot:
            raise SlotLoopError(
                f"Binding '{_describe(all_bindings[slot])}' is bound to slot '{slot}' "
                f"but declares slot '{all_bindings[slot].slot}'"
            )

    last_position = {slot: index for index, slot in enumerate(SLOT_STEP_ORDER)}
    available_keys: set[str] = {
        key for binding in all_bindings.values() for key in binding.provides if key in KEYS_WRITTEN_BEFORE_THE_FIRST_STEP
    }
    for index, slot in enumerate(SLOT_STEP_ORDER):
        binding = all_bindings[slot]
        if index == last_position[slot]:
            missing_keys = set(binding.requires) - available_keys - set(binding.provides)
            if missing_keys:
                raise SlotLoopError(
                    f"Binding '{_describe(binding)}' requires {sorted(missing_keys)}, "
                    f"but no binding earlier in the step provides it"
                )
        available_keys |= set(binding.provides)

    for slot in SlotName.ALL:
        binding = all_bindings[slot]
        requires_truth = MessageKey.TRUTH_STATE in binding.requires
        if requires_truth and not binding.reads_truth:
            raise SlotLoopError(
                f"Truth leak: binding '{_describe(binding)}' requires TruthState but has reads_truth=False"
            )
        if binding.reads_truth:
            is_truth_navigation = (
                slot == SlotName.NAVIGATION and getattr(binding, "registry_name", "") in TRUTH_NAVIGATION_NAMES
            )
            if slot not in TRUTH_SIDE_SLOTS and not is_truth_navigation:
                raise SlotLoopError(
                    f"Truth leak: binding '{_describe(binding)}' in slot '{slot}' sets reads_truth=True. "
                    f"Only truth-side slots {sorted(TRUTH_SIDE_SLOTS)} and {sorted(TRUTH_NAVIGATION_NAMES)} navigation may read TruthState"
                )


def _describe(binding: Any) -> str:
    """Return ``slot: name`` of a binding for error messages."""
    return f"{getattr(binding, 'slot', '?')}: {getattr(binding, 'registry_name', type(binding).__name__)}"
