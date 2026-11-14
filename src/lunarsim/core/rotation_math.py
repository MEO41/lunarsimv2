# SPDX-License-Identifier: BSD-3-Clause
"""Batched quaternion math in pure torch.

Convention used everywhere in LunarSim: quaternions are (w, x, y, z), unit length, and rotate
vectors from the body frame to the parent frame. Isaac Lab 3.0 stores (x, y, z, w); the only
place that converts between the two is the plant binding.
"""

from __future__ import annotations

import torch
from torch import Tensor


def quaternion_wxyz_to_xyzw(quaternion: Tensor) -> Tensor:
    """Reorder (w, x, y, z) to (x, y, z, w)."""
    return torch.cat((quaternion[..., 1:4], quaternion[..., 0:1]), dim=-1)


def quaternion_xyzw_to_wxyz(quaternion: Tensor) -> Tensor:
    """Reorder (x, y, z, w) to (w, x, y, z)."""
    return torch.cat((quaternion[..., 3:4], quaternion[..., 0:3]), dim=-1)


def quaternion_multiply(first: Tensor, second: Tensor) -> Tensor:
    """Return the Hamilton product ``first * second`` of two (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = first.unbind(-1)
    w2, x2, y2, z2 = second.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quaternion_conjugate(quaternion: Tensor) -> Tensor:
    """Return the conjugate (the inverse rotation for a unit quaternion)."""
    return torch.cat((quaternion[..., 0:1], -quaternion[..., 1:4]), dim=-1)


def rotate_vector_body_to_parent(quaternion: Tensor, vector: Tensor) -> Tensor:
    """Rotate ``vector`` from the body frame to the parent frame."""
    scalar = quaternion[..., 0:1]
    axis = quaternion[..., 1:4]
    cross_once = torch.cross(axis, vector, dim=-1)
    return vector + 2.0 * (scalar * cross_once + torch.cross(axis, cross_once, dim=-1))


def rotate_vector_parent_to_body(quaternion: Tensor, vector: Tensor) -> Tensor:
    """Rotate ``vector`` from the parent frame to the body frame."""
    return rotate_vector_body_to_parent(quaternion_conjugate(quaternion), vector)


def quaternion_from_axis_angle(axis: Tensor, angle: Tensor) -> Tensor:
    """Return the quaternion for a rotation of ``angle`` (rad, shape (N,)) about ``axis`` ((N, 3))."""
    unit_axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    half_angle = 0.5 * angle.unsqueeze(-1)
    return torch.cat((torch.cos(half_angle), unit_axis * torch.sin(half_angle)), dim=-1)


def quaternion_to_rotation_vector(quaternion: Tensor) -> Tensor:
    """Return the rotation vector (axis times angle, rad) of a quaternion, shortest way round."""
    quaternion = torch.where(quaternion[..., 0:1] < 0.0, -quaternion, quaternion)
    vector_part = quaternion[..., 1:4]
    vector_norm = vector_part.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, quaternion[..., 0:1])
    # For small angles the rotation vector tends to 2 * vector part.
    scale = torch.where(vector_norm > 1e-8, angle / vector_norm.clamp_min(1e-12), torch.full_like(angle, 2.0))
    return vector_part * scale


def body_up_axis_in_parent(quaternion: Tensor) -> Tensor:
    """Return the body +z axis (the lander's thrust and 'up' axis) expressed in the parent frame."""
    up_axis = torch.zeros(quaternion.shape[:-1] + (3,), dtype=quaternion.dtype, device=quaternion.device)
    up_axis[..., 2] = 1.0
    return rotate_vector_body_to_parent(quaternion, up_axis)


def tilt_angle_from_vertical(quaternion: Tensor) -> Tensor:
    """Return the angle (rad, shape (N,)) between the body +z axis and the site +z axis."""
    return torch.acos(body_up_axis_in_parent(quaternion)[..., 2].clamp(-1.0, 1.0))
