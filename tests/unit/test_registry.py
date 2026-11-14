# SPDX-License-Identifier: BSD-3-Clause
"""Registry: lookup, requires/provides check, truth-leak check."""

import pytest

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey
from lunarsim.core.registry import (
    RegistryError,
    SlotLoopError,
    check_slot_loop,
    get_binding_class,
    register_binding,
)


def _declaration(slot, requires=(), provides=(), reads_truth=False, registry_name="test"):
    """Build a minimal binding declaration class for the loop check."""
    return type(
        f"Declared_{slot}",
        (),
        {
            "slot": slot,
            "registry_name": registry_name,
            "requires": frozenset(requires),
            "provides": frozenset(provides),
            "reads_truth": reads_truth,
        },
    )


def _good_loop():
    """Return a closed loop of declarations that mirrors scenario S1."""
    key = MessageKey
    return {
        SlotName.START: _declaration(SlotName.START, provides=[key.START_STATE]),
        SlotName.PLANT: _declaration(
            SlotName.PLANT,
            requires=[key.START_STATE, key.ACTUATOR_WRENCH, key.SURFACE_WRENCH],
            provides=[key.TRUTH_STATE],
        ),
        SlotName.SURFACE: _declaration(SlotName.SURFACE, provides=[key.SURFACE_WRENCH]),
        SlotName.SENSORS: _declaration(SlotName.SENSORS),
        SlotName.FAULTS: _declaration(SlotName.FAULTS),
        SlotName.NAVIGATION: _declaration(
            SlotName.NAVIGATION,
            requires=[key.TRUTH_STATE],
            provides=[key.NAVIGATION_ESTIMATE],
            reads_truth=True,
            registry_name="truth",
        ),
        SlotName.AGENT: _declaration(SlotName.AGENT, requires=[key.OBSERVATION], provides=[key.COMMAND]),
        SlotName.ACTUATORS: _declaration(SlotName.ACTUATORS, requires=[key.COMMAND], provides=[key.ACTUATOR_WRENCH]),
        SlotName.METRICS: _declaration(
            SlotName.METRICS, requires=[key.TRUTH_STATE], provides=[key.EPISODE_EVENTS], reads_truth=True
        ),
    }


def test_closed_loop_passes():
    """The S1-shaped loop passes all three checks."""
    check_slot_loop(_good_loop())


def test_missing_slot_fails():
    """Every one of the nine slots needs a binding."""
    loop = _good_loop()
    del loop[SlotName.FAULTS]
    with pytest.raises(SlotLoopError, match="faults"):
        check_slot_loop(loop)


def test_missing_provider_fails():
    """A required key without an earlier provider stops the run."""
    loop = _good_loop()
    loop[SlotName.NAVIGATION] = _declaration(
        SlotName.NAVIGATION, requires=[MessageKey.MEASUREMENT_PREFIX + "imu"], provides=[MessageKey.NAVIGATION_ESTIMATE]
    )
    with pytest.raises(SlotLoopError, match="measurement.imu"):
        check_slot_loop(loop)


def test_provider_later_in_the_step_does_not_count():
    """The actuators cannot read a message that the metrics write after them."""
    loop = _good_loop()
    loop[SlotName.ACTUATORS] = _declaration(
        SlotName.ACTUATORS, requires=[MessageKey.COMMAND, MessageKey.EPISODE_EVENTS], provides=[MessageKey.ACTUATOR_WRENCH]
    )
    with pytest.raises(SlotLoopError, match="episode_events"):
        check_slot_loop(loop)


def test_truth_reader_without_flag_fails():
    """Requiring TruthState without reads_truth=True is a leak."""
    loop = _good_loop()
    loop[SlotName.FAULTS] = _declaration(SlotName.FAULTS, requires=[MessageKey.TRUTH_STATE])
    with pytest.raises(SlotLoopError, match="Truth leak"):
        check_slot_loop(loop)


@pytest.mark.parametrize("slot", [SlotName.FAULTS, SlotName.AGENT, SlotName.ACTUATORS, SlotName.NAVIGATION])
def test_flight_side_slot_may_not_set_reads_truth(slot):
    """Faults, agent, actuators and any navigation other than 'truth' may not read TruthState."""
    loop = _good_loop()
    original = loop[slot]
    loop[slot] = _declaration(
        slot,
        requires=set(original.requires) | {MessageKey.TRUTH_STATE},
        provides=original.provides,
        reads_truth=True,
        registry_name="ekf",
    )
    with pytest.raises(SlotLoopError, match="Truth leak"):
        check_slot_loop(loop)


def test_binding_in_wrong_slot_fails():
    """A binding class declared for one slot cannot be bound to another."""
    loop = _good_loop()
    loop[SlotName.SENSORS] = _declaration(SlotName.FAULTS)
    with pytest.raises(SlotLoopError, match="declares slot"):
        check_slot_loop(loop)


def test_register_and_look_up_binding():
    """The decorator stamps slot and name; unknown names and slots fail with a clear error."""

    @register_binding(SlotName.FAULTS, "registry_test_binding")
    class RegistryTestBinding:
        requires = frozenset()
        provides = frozenset()
        reads_truth = False

    assert get_binding_class(SlotName.FAULTS, "registry_test_binding") is RegistryTestBinding
    assert RegistryTestBinding.slot == SlotName.FAULTS
    with pytest.raises(RegistryError, match="No binding"):
        get_binding_class(SlotName.FAULTS, "does_not_exist")
    with pytest.raises(RegistryError, match="Unknown slot"):
        get_binding_class("not_a_slot", "x")
