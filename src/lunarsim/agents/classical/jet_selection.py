# SPDX-License-Identifier: BSD-3-Clause
"""Choose reaction-control jets that give a pure torque about one body axis.

This uses only the lander design (jet positions and directions from the preset), never the
true state, so both the classical agent and the tests may use it.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.assets.lander_preset import LanderPreset


class JetCoupleTable:
    """For each body axis and sign: the jets that fire together as a pure couple.

    Torque about body x or y uses the four vertical jets with the right sign of torque (two
    push up on one side, two push down on the other). Torque about body z uses the four
    horizontal jets with the right sign. In both cases the net force is zero and the torques
    about the other two axes cancel.
    """

    def __init__(self, preset: LanderPreset, device: torch.device | str = "cpu") -> None:
        """Build the selection masks and the torque of each couple."""
        force = torch.tensor(preset.jet_force_matrix(), device=device)  # (3, n_jets)
        torque = torch.tensor(preset.jet_torque_matrix(), device=device)  # (3, n_jets)
        is_vertical_jet = force[2].abs() > 0.5 * preset.jet_thrust
        number_of_jets = force.shape[1]
        # masks[axis, 0] = jets for positive torque, masks[axis, 1] = jets for negative torque
        self.masks = torch.zeros(3, 2, number_of_jets, device=device)
        for axis in range(3):
            candidates = is_vertical_jet if axis < 2 else ~is_vertical_jet
            self.masks[axis, 0] = (candidates & (torque[axis] > 1e-6)).float()
            self.masks[axis, 1] = (candidates & (torque[axis] < -1e-6)).float()
        # Torque about its own axis when a whole couple fires, N m (the same for both signs).
        self.couple_torque = torch.stack([(self.masks[axis, 0] * torque[axis]).sum() for axis in range(3)])
        self.net_force_check = torch.stack(
            [(self.masks[axis, sign] * force).sum(dim=1).abs().max() for axis in range(3) for sign in range(2)]
        ).max()

    def jets_for_axis_commands(self, axis_commands: Tensor) -> Tensor:
        """Turn per-axis commands in [-1, 1], shape (N, 3), into jet duties, shape (N, n_jets).

        A positive command fires the positive couple of that axis with that duty; a negative
        command fires the negative couple.
        """
        positive = axis_commands.clamp(0.0, 1.0)  # (N, 3)
        negative = (-axis_commands).clamp(0.0, 1.0)
        duties = positive @ self.masks[:, 0, :] + negative @ self.masks[:, 1, :]
        return duties.clamp(0.0, 1.0)
