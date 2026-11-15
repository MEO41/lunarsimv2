# SPDX-License-Identifier: BSD-3-Clause
"""The held-out fault test grid of the failure study (milestone 3, paper 1).

Every combination of fault type (8), target sensor (2) and onset altitude (5, 10, 20, 30, 40 m)
is one cell; environment ``i`` gets cell ``i mod 80``, so 4,096 environments give about 51
landings per cell. Sizes sit in the middle of their configured ranges (``size_fraction`` 0.5).
With the evaluation seed (1000, not used in training) the start states, signs, event times and
delays are fixed too, so every agent meets exactly the same test.
"""

from __future__ import annotations

import torch

from lunarsim.slots.faults.fault_schedule import FAULT_CODE, FAULT_TYPES, TARGET_NAMES

GRID_ONSET_ALTITUDES_M = (5.0, 10.0, 20.0, 30.0, 40.0)
GRID_SIZE_FRACTION = 0.5


def grid_cells() -> list[tuple[str, str, float]]:
    """Return every (fault type, target, onset altitude) cell of the grid."""
    return [
        (fault, target, onset)
        for fault in FAULT_TYPES
        if fault != "none"
        for target in TARGET_NAMES
        for onset in GRID_ONSET_ALTITUDES_M
    ]


def apply_fault_grid(schedule, count: int, device) -> None:
    """Give environment ``i`` the grid cell ``i mod (number of cells)`` (``set_fixed_faults``)."""
    cells = grid_cells()
    chosen = [cells[index % len(cells)] for index in range(count)]
    schedule.set_fixed_faults(
        torch.tensor([FAULT_CODE[fault] for fault, _, _ in chosen], device=device),
        torch.tensor([TARGET_NAMES.index(target) for _, target, _ in chosen], device=device),
        torch.tensor([onset for _, _, onset in chosen], device=device),
        torch.full((count,), GRID_SIZE_FRACTION, device=device),
    )
