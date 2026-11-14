# SPDX-License-Identifier: BSD-3-Clause
"""Per-slot generators: same seed gives the same draws; slots do not disturb each other."""

import torch

from lunarsim.core.binding import SlotName
from lunarsim.core.random_generators import SlotRandomGenerators, derive_slot_seed


def _draw(generators: SlotRandomGenerators, slot: str, count: int = 8) -> torch.Tensor:
    """Draw ``count`` uniform numbers from the generator of ``slot``."""
    return torch.rand(count, generator=generators.generator_for(slot))


def test_same_root_seed_gives_same_draws():
    """Two generator sets with the same root seed produce identical streams."""
    first = SlotRandomGenerators(42, SlotName.ALL)
    second = SlotRandomGenerators(42, SlotName.ALL)
    for slot in SlotName.ALL:
        assert torch.equal(_draw(first, slot), _draw(second, slot))


def test_different_root_seed_or_slot_gives_different_draws():
    """Streams differ between root seeds and between slots."""
    assert not torch.equal(
        _draw(SlotRandomGenerators(1, SlotName.ALL), SlotName.START),
        _draw(SlotRandomGenerators(2, SlotName.ALL), SlotName.START),
    )
    generators = SlotRandomGenerators(1, SlotName.ALL)
    assert not torch.equal(_draw(generators, SlotName.START), _draw(generators, SlotName.FAULTS))


def test_draws_in_one_slot_do_not_move_another_slot():
    """Extra draws in the agent slot leave the faults stream unchanged."""
    quiet = SlotRandomGenerators(7, SlotName.ALL)
    busy = SlotRandomGenerators(7, SlotName.ALL)
    _draw(busy, SlotName.AGENT, count=1000)
    assert torch.equal(_draw(quiet, SlotName.FAULTS), _draw(busy, SlotName.FAULTS))


def test_slot_seed_does_not_depend_on_which_other_slots_exist():
    """Removing a slot from the set does not change the seed of the others."""
    full = SlotRandomGenerators(7, SlotName.ALL)
    partial = SlotRandomGenerators(7, [SlotName.START])
    assert full.seeds()[SlotName.START] == partial.seeds()[SlotName.START] == derive_slot_seed(7, SlotName.START)
