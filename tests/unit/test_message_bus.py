# SPDX-License-Identifier: BSD-3-Clause
"""MessageBus: write/read, missing key, declared-interface guard, truth guard."""

import pytest
from helpers import make_truth_state

from lunarsim.core.message_bus import MessageBus, MissingMessageError, TruthAccessError, UndeclaredMessageError
from lunarsim.core.message_types import MessageKey


class _Writer:
    requires = frozenset()
    provides = frozenset({MessageKey.TRUTH_STATE, "thing"})
    reads_truth = False


class _HonestReader:
    requires = frozenset({MessageKey.TRUTH_STATE, "thing"})
    provides = frozenset()
    reads_truth = True


class _LeakyReader:
    requires = frozenset({MessageKey.TRUTH_STATE})
    provides = frozenset()
    reads_truth = False


def test_write_then_read_returns_latest_message():
    """The bus keeps the latest message of each key."""
    bus = MessageBus()
    bus.write("thing", 1, _Writer())
    bus.write("thing", 2, _Writer())
    assert bus.read("thing", _HonestReader()) == 2
    assert bus.has("thing") and not bus.has("other")


def test_read_of_missing_key_fails():
    """Reading a key that nobody wrote is an error."""
    with pytest.raises(MissingMessageError):
        MessageBus().read("thing", _HonestReader())


def test_undeclared_read_and_write_fail():
    """A binding may only touch keys in its requires/provides sets."""
    bus = MessageBus()
    with pytest.raises(UndeclaredMessageError):
        bus.write("not_declared", 1, _Writer())
    bus.write("thing", 1, _Writer())
    with pytest.raises(UndeclaredMessageError):
        bus.read("thing", _LeakyReader())


def test_truth_guard_blocks_reader_without_flag_and_logs_honest_reader():
    """Runtime guard of goal G3: only reads_truth=True bindings get TruthState."""
    bus = MessageBus()
    bus.write(MessageKey.TRUTH_STATE, make_truth_state(), _Writer())
    with pytest.raises(TruthAccessError):
        bus.read(MessageKey.TRUTH_STATE, _LeakyReader())
    bus.read(MessageKey.TRUTH_STATE, _HonestReader())
    assert bus.truth_read_log == {"_HonestReader": 1}
