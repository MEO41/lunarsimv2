# SPDX-License-Identifier: BSD-3-Clause
"""Message latency, executor jitter and the reply deadline of a bridged node (S10 timing test).

Design S10: "Timing test: add the real message latency and executor jitter. Measure the change in
touchdown metrics", with ``timeout_ms: 20``. The simulator waits for every reply (so the node
always processes every request in order), and then decides in simulation time whether the reply
was on time:

    delay = measured round trip + added latency + jitter

where the added latency is fixed and the jitter is drawn per message from a seeded normal
distribution, cut at zero (model choices; the parameters are ``added_latency_ms`` and
``jitter_ms``). A reply whose delay is above ``timeout_ms`` misses its deadline: the binding
keeps its previous output for this control step, as a flight loop with a deadline does. With
``timeout_ms: null`` every reply is used (the parity test).

The measured round trip is real wall time of this machine, so it varies between runs. For a
repeatable timing test set ``count_measured_round_trip: false``: only the seeded added latency
and jitter decide the misses, and the measured times are still recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy


@dataclass
class DeadlineModel:
    """Decides per reply whether it met the deadline; keeps the timing statistics."""

    timeout_ms: float | None = None
    added_latency_ms: float = 0.0
    jitter_ms: float = 0.0
    count_measured_round_trip: bool = True
    seed: int = 0
    measured_round_trips_ms: list[float] = field(default_factory=list)
    delays_ms: list[float] = field(default_factory=list)
    misses: int = 0

    def __post_init__(self) -> None:
        """Create the seeded jitter stream."""
        self.random = numpy.random.default_rng(self.seed)

    @classmethod
    def from_parameters(cls, parameters: dict, seed: int) -> DeadlineModel:
        """Build the model from a bridge binding's parameters (``timeout_ms``, ``added_latency_ms``, ``jitter_ms``, ``count_measured_round_trip``)."""
        timeout = parameters.get("timeout_ms")
        return cls(
            timeout_ms=None if timeout is None else float(timeout),
            added_latency_ms=float(parameters.get("added_latency_ms", 0.0)),
            jitter_ms=float(parameters.get("jitter_ms", 0.0)),
            count_measured_round_trip=bool(parameters.get("count_measured_round_trip", True)),
            seed=int(seed),
        )

    def reply_is_on_time(self, measured_round_trip_s: float) -> bool:
        """Record one reply's timing and return True if it met the deadline."""
        measured = 1e3 * measured_round_trip_s
        jitter = max(0.0, float(self.random.normal(0.0, self.jitter_ms))) if self.jitter_ms > 0.0 else 0.0
        delay = (measured if self.count_measured_round_trip else 0.0) + self.added_latency_ms + jitter
        self.measured_round_trips_ms.append(measured)
        self.delays_ms.append(delay)
        on_time = self.timeout_ms is None or delay <= self.timeout_ms
        self.misses += 0 if on_time else 1
        return on_time

    def summary(self) -> dict[str, float]:
        """Return the timing statistics so far: replies, misses, measured round trip and delay percentiles (ms)."""
        measured = numpy.asarray(self.measured_round_trips_ms) if self.measured_round_trips_ms else numpy.zeros(1)
        delays = numpy.asarray(self.delays_ms) if self.delays_ms else numpy.zeros(1)
        return {
            "replies": len(self.delays_ms),
            "deadline_misses": self.misses,
            "measured_round_trip_ms_median": float(numpy.median(measured)),
            "measured_round_trip_ms_p99": float(numpy.percentile(measured, 99)),
            "measured_round_trip_ms_max": float(measured.max()),
            "delay_ms_median": float(numpy.median(delays)),
            "delay_ms_p99": float(numpy.percentile(delays, 99)),
        }

    def take_episode_counts(self) -> tuple[int, int]:
        """Return (replies, misses) since the last call and start counting again (per-episode records)."""
        replies, misses = len(self.delays_ms) - getattr(self, "_replies_mark", 0), self.misses - getattr(self, "_misses_mark", 0)
        self._replies_mark, self._misses_mark = len(self.delays_ms), self.misses
        return replies, misses
