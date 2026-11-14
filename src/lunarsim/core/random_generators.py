# SPDX-License-Identifier: BSD-3-Clause
"""One random stream per slot, all derived from one root seed."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import torch


def derive_slot_seed(root_seed: int, slot_name: str) -> int:
    """Return the seed of one slot: a hash of the root seed and the slot name.

    A hash (and not ``root_seed + index``) is used so that adding or removing a slot never
    changes the seed of another slot.
    """
    digest = hashlib.sha256(f"lunarsim:{int(root_seed)}:{slot_name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


class SlotRandomGenerators:
    """Owns one ``torch.Generator`` per slot.

    A draw in one slot never moves the stream of another slot. Thus, a change to the agent does
    not change the fault schedule or the start states.
    """

    def __init__(self, root_seed: int, slot_names: Iterable[str], device: torch.device | str = "cpu") -> None:
        """Create and seed one generator per slot name on ``device``."""
        self.root_seed = int(root_seed)
        self.device = device
        self._generators: dict[str, torch.Generator] = {}
        self._seeds: dict[str, int] = {}
        for slot_name in slot_names:
            seed = derive_slot_seed(self.root_seed, slot_name)
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            self._generators[slot_name] = generator
            self._seeds[slot_name] = seed

    def generator_for(self, slot_name: str) -> torch.Generator:
        """Return the generator of ``slot_name``."""
        return self._generators[slot_name]

    def seeds(self) -> dict[str, int]:
        """Return the seed of every slot (written to the run manifest)."""
        return dict(self._seeds)
