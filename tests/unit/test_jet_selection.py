# SPDX-License-Identifier: BSD-3-Clause
"""Jet couples: each axis command gives a pure torque about that axis."""

import pytest
import torch

from lunarsim.agents.classical.jet_selection import JetCoupleTable
from lunarsim.assets.lander_preset import load_lander_preset
from lunarsim.env.scenario_loader import default_config_directory


@pytest.fixture(scope="module")
def preset():
    """Load the Apollo Lunar Module preset."""
    return load_lander_preset(default_config_directory() / "landers" / "apollo_lunar_module.yaml")


def test_each_couple_is_four_jets_with_pure_torque(preset):
    """Every couple has 4 jets, zero net force and torque only about its own axis."""
    table = JetCoupleTable(preset)
    force = torch.tensor(preset.jet_force_matrix())
    torque = torch.tensor(preset.jet_torque_matrix())
    for axis in range(3):
        for sign_index, sign in enumerate((1.0, -1.0)):
            mask = table.masks[axis, sign_index]
            assert int(mask.sum()) == 4
            assert float((force * mask).sum(dim=1).abs().max()) < 1e-3
            net_torque = (torque * mask).sum(dim=1)
            assert float(net_torque[axis]) == pytest.approx(sign * float(table.couple_torque[axis]), rel=1e-5)
            others = [other for other in range(3) if other != axis]
            assert float(net_torque[others].abs().max()) < 1e-2


def test_axis_commands_map_to_duties(preset):
    """Sign picks the couple, magnitude sets the duty, and axes combine."""
    table = JetCoupleTable(preset)
    duties = table.jets_for_axis_commands(torch.tensor([[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert torch.equal(duties[0], table.masks[0, 0])
    assert torch.allclose(duties[1], 0.5 * table.masks[0, 1])
    assert float(duties[2].sum()) == 0.0
