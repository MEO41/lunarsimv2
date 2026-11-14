# SPDX-License-Identifier: BSD-3-Clause
"""Quaternion helpers against hand-computed answers."""

import math

import torch

from lunarsim.core import rotation_math as rotation


def test_axis_angle_rotation_of_a_vector():
    """A 90 degree turn about z takes body x to parent y."""
    quaternion = rotation.quaternion_from_axis_angle(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([math.pi / 2]))
    rotated = rotation.rotate_vector_body_to_parent(quaternion, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)
    back = rotation.rotate_vector_parent_to_body(quaternion, rotated)
    assert torch.allclose(back, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)


def test_multiply_composes_rotations_and_conjugate_inverts():
    """q * conj(q) is the identity; two 45 degree turns make one 90 degree turn."""
    axis = torch.tensor([[1.0, 0.0, 0.0]])
    half = rotation.quaternion_from_axis_angle(axis, torch.tensor([math.pi / 4]))
    full = rotation.quaternion_from_axis_angle(axis, torch.tensor([math.pi / 2]))
    assert torch.allclose(rotation.quaternion_multiply(half, half), full, atol=1e-6)
    identity = rotation.quaternion_multiply(full, rotation.quaternion_conjugate(full))
    assert torch.allclose(identity, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1e-6)


def test_tilt_angle_and_rotation_vector():
    """A 10 degree turn about x tilts the up axis by 10 degrees; the rotation vector recovers it."""
    angle = math.radians(10.0)
    quaternion = rotation.quaternion_from_axis_angle(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([angle]))
    assert torch.allclose(rotation.tilt_angle_from_vertical(quaternion), torch.tensor([angle]), atol=1e-6)
    assert torch.allclose(rotation.quaternion_to_rotation_vector(quaternion), torch.tensor([[angle, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(rotation.quaternion_to_rotation_vector(-quaternion), torch.tensor([[angle, 0.0, 0.0]]), atol=1e-6)


def test_component_order_conversion_round_trip():
    """wxyz -> xyzw -> wxyz is the identity."""
    quaternion = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    assert torch.equal(rotation.quaternion_wxyz_to_xyzw(quaternion), torch.tensor([[0.2, 0.3, 0.4, 0.1]]))
    assert torch.equal(rotation.quaternion_xyzw_to_wxyz(rotation.quaternion_wxyz_to_xyzw(quaternion)), quaternion)
