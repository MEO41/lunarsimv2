# SPDX-License-Identifier: BSD-3-Clause
"""The message bus: the latest message of each key, for all environments."""

from __future__ import annotations

from typing import Any

from lunarsim.core.message_types import MessageKey


class TruthAccessError(RuntimeError):
    """A binding without ``reads_truth=True`` tried to read ``TruthState``."""


class UndeclaredMessageError(RuntimeError):
    """A binding read or wrote a key that is not in its ``requires`` or ``provides`` set."""


class MissingMessageError(KeyError):
    """A binding read a key that nobody has written yet."""


class MessageBus:
    """Holds the latest message of each key.

    Every read and write names the binding that does it. The bus uses this to enforce, at run
    time, the same rules that the registry checks before the run:

    - a binding reads only keys in its ``requires`` set and writes only keys in its ``provides`` set;
    - only a binding with ``reads_truth=True`` reads ``TruthState``.
    """

    def __init__(self) -> None:
        """Create an empty bus."""
        self._messages: dict[str, Any] = {}
        self._truth_read_log: dict[str, int] = {}

    def write(self, key: str, message: Any, writer: Any) -> None:
        """Store ``message`` under ``key``. ``writer`` is the binding that provides it."""
        if key not in writer.provides:
            raise UndeclaredMessageError(
                f"{type(writer).__name__} wrote '{key}' but does not list it in 'provides' {sorted(writer.provides)}"
            )
        self._messages[key] = message

    def read(self, key: str, reader: Any) -> Any:
        """Return the latest message under ``key``. ``reader`` is the binding that requires it."""
        if key not in reader.requires:
            raise UndeclaredMessageError(
                f"{type(reader).__name__} read '{key}' but does not list it in 'requires' {sorted(reader.requires)}"
            )
        if key == MessageKey.TRUTH_STATE:
            if not getattr(reader, "reads_truth", False):
                raise TruthAccessError(
                    f"{type(reader).__name__} read TruthState without reads_truth=True. This is a truth leak."
                )
            reader_name = type(reader).__name__
            self._truth_read_log[reader_name] = self._truth_read_log.get(reader_name, 0) + 1
        if key not in self._messages:
            raise MissingMessageError(f"'{key}' was read by {type(reader).__name__} before anyone wrote it")
        return self._messages[key]

    def has(self, key: str) -> bool:
        """Return True if a message is stored under ``key``."""
        return key in self._messages

    @property
    def truth_read_log(self) -> dict[str, int]:
        """Return how many times each binding class has read ``TruthState`` (for the leak report)."""
        return dict(self._truth_read_log)
