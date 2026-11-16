# SPDX-License-Identifier: BSD-3-Clause
"""Parts shared by the two bridge bindings (``nav: ros2_bridge``, ``agent: ros2_bridge``)."""

from __future__ import annotations

import atexit
import pickle
import weakref
from typing import Any

import torch


def picklable_shared_data(shared_data: dict[str, Any]) -> dict[str, Any]:
    """Return the entries of the scenario's shared data that can travel to a node (others stay behind)."""
    kept = {}
    for key, value in shared_data.items():
        try:
            pickle.dumps(value)
        except Exception:
            continue
        kept[key] = value
    return kept


def inner_binding_selection(parameters: dict, slot_label: str) -> tuple[str, dict]:
    """Split the bridge's ``node`` parameters into the inner binding name and its parameters."""
    node = dict(parameters.get("node") or {})
    name = node.pop("binding", None)
    if not name:
        raise ValueError(f"{slot_label}: ros2_bridge needs node.binding (the binding the node runs)")
    return str(name), node


def close_at_exit(owner: Any) -> None:
    """Close the owner's transport when the program ends (the node process stops with it)."""
    reference = weakref.ref(owner)

    def close() -> None:
        """Close the transport if the owner still exists."""
        bridge = reference()
        if bridge is not None:
            bridge.close()

    atexit.register(close)


class EpisodeTimingRecords:
    """Per-environment counts of replies and deadline misses, kept per episode for the run records."""

    def __init__(self, prefix: str, count: int) -> None:
        """Allocate the counters of ``count`` environments; columns start with ``prefix``."""
        self.prefix = prefix
        self.episode_number = torch.full((count,), -1, dtype=torch.long)
        self.replies = torch.zeros(count, dtype=torch.long)
        self.misses = torch.zeros(count, dtype=torch.long)
        self.round_trip_maximum_ms = torch.zeros(count)
        self.records: list[dict[str, torch.Tensor]] = []

    def add(self, replies_on_time: bool, round_trip_ms: float, environment_ids: torch.Tensor | None = None) -> None:
        """Count one reply for the given environments (default: all)."""
        ids = slice(None) if environment_ids is None else environment_ids.cpu()
        self.replies[ids] += 1
        self.misses[ids] += 0 if replies_on_time else 1
        self.round_trip_maximum_ms[ids] = torch.clamp_min(self.round_trip_maximum_ms[ids], round_trip_ms)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Store the finished episodes of ``env_ids`` and start new counts."""
        ids = env_ids.cpu()
        finished = ids[self.episode_number[ids] >= 0]
        if len(finished) > 0:
            self.records.append({
                "environment_index": finished.clone(),
                "episode_number": self.episode_number[finished].clone(),
                f"{self.prefix}_replies": self.replies[finished].clone(),
                f"{self.prefix}_deadline_misses": self.misses[finished].clone(),
                f"{self.prefix}_round_trip_maximum_ms": self.round_trip_maximum_ms[finished].clone(),
            })
        self.episode_number[ids] += 1
        self.replies[ids] = 0
        self.misses[ids] = 0
        self.round_trip_maximum_ms[ids] = 0.0

    def drain(self) -> dict[str, torch.Tensor]:
        """Return and forget the stored episode records (columns concatenated)."""
        if not self.records:
            return {}
        merged = {name: torch.cat([record[name] for record in self.records]) for name in self.records[0]}
        self.records = []
        return merged
