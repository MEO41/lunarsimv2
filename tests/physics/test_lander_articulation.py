# SPDX-License-Identifier: BSD-3-Clause
"""The lander articulation built from the preset has the right structure and mass properties (GPU)."""

import pytest
import torch
from isaac_test_support import make_environment

from lunarsim.core.binding import SlotName


@pytest.fixture(scope="module")
def environment():
    """Build S1 with four environments."""
    environment = make_environment("s1_terminal_truth", ["num_envs=4"])
    environment.reset()
    yield environment
    environment.close()


def test_structure_is_one_body_four_legs_four_prismatic_struts(environment):
    """One articulation with 5 links and 4 joints, named as in the builder."""
    lander = environment.bindings[SlotName.PLANT].asset
    assert lander.num_bodies == 5
    assert lander.num_joints == 4
    assert sorted(lander.body_names) == ["body", "leg_1", "leg_2", "leg_3", "leg_4"]
    assert sorted(lander.joint_names) == ["strut_1", "strut_2", "strut_3", "strut_4"]
    assert not lander.is_fixed_base


def test_mass_and_inertia_in_physx_match_the_preset(environment):
    """PhysX holds the preset's total mass and the body inertia.

    Tolerance 1e-4 relative: these are float32 copies of the numbers in the USD file.
    """
    plant = environment.bindings[SlotName.PLANT]
    preset = plant.preset
    masses = plant.asset.data.body_mass.torch
    assert torch.allclose(masses.sum(dim=1), torch.full((4,), preset.mass_at_start, device=masses.device), rtol=1e-4)
    root_inertia = plant.asset.data.body_inertia.torch[:, plant.root_body_index].reshape(4, 3, 3)
    expected = torch.diag(torch.tensor(preset.body_inertia, device=masses.device))
    assert torch.allclose(root_inertia, expected.expand(4, 3, 3), rtol=1e-4, atol=1e-2)


def test_strut_gains_and_limits_match_the_preset(environment):
    """The strut spring, damper and stroke limits in PhysX are the preset's values."""
    plant = environment.bindings[SlotName.PLANT]
    preset = plant.preset
    data = plant.asset.data
    assert torch.allclose(data.joint_stiffness.torch, torch.full((4, 4), preset.strut_stiffness, device=environment.device), rtol=1e-5)
    assert torch.allclose(data.joint_damping.torch, torch.full((4, 4), preset.strut_damping, device=environment.device), rtol=1e-5)
    limits = data.joint_pos_limits.torch
    assert torch.allclose(limits[..., 0], torch.zeros(4, 4, device=environment.device), atol=1e-6)
    assert torch.allclose(limits[..., 1], torch.full((4, 4), preset.strut_stroke, device=environment.device), rtol=1e-5)


def test_solver_and_contact_settings_from_the_scenario_reach_physx(environment):
    """The chosen solver settings are really applied to the stage, not silently dropped."""
    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    parameters = environment.scenario.slots[SlotName.PLANT].parameters
    footpad = stage.GetPrimAtPath("/World/envs/env_0/Lander/leg_1/footpad")
    assert footpad.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(parameters["contact_offset_m"])
    body = stage.GetPrimAtPath("/World/envs/env_0/Lander/body")
    assert body.GetAttribute("physxArticulation:solverPositionIterationCount").Get() == parameters["solver_position_iterations"]
    assert body.GetAttribute("physxRigidBody:linearDamping").Get() == 0.0
    scene = stage.GetPrimAtPath("/physicsScene")
    assert scene.GetAttribute("physxScene:solverType").Get() == "PGS"
