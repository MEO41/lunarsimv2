# SPDX-License-Identifier: BSD-3-Clause
"""Zero-effort-miss / zero-effort-velocity (ZEM/ZEV) guidance math (pure torch).

In a constant gravity field ``g``, a vehicle that stops thrusting now reaches, after the
time-to-go ``T``, the position ``r + v T + g T^2 / 2`` and the velocity ``v + g T``.

- ZEM (zero-effort miss) is the target position minus that position.
- ZEV (zero-effort velocity error) is the target velocity minus that velocity.

The thrust acceleration that removes both errors with the least integral of acceleration
squared is ``a = 6 ZEM / T^2 - 2 ZEV / T``. Along the resulting path the thrust acceleration
is a straight line in time, from ``a(0)`` to ``a(T) = -6 ZEM / T^2 + 4 ZEV / T``.

Reference: Ebrahimi, Bahrami, Roshanian, "Optimal sliding-mode guidance with terminal
velocity constraint for fixed-interval propulsive maneuvers", Acta Astronautica 62 (2008);
Guo, Hawkins, Wie, "Applications of generalized zero-effort-miss/zero-effort-velocity
feedback guidance algorithm", J. Guidance, Control, and Dynamics 36 (2013).
"""

from __future__ import annotations

import torch
from torch import Tensor


def zero_effort_errors(
    position: Tensor,
    velocity: Tensor,
    target_position: Tensor,
    target_velocity: Tensor,
    gravity: Tensor,
    time_to_go: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return (ZEM, ZEV). Vectors have shape (..., 3); ``time_to_go`` has shape (...,)."""
    time = time_to_go.unsqueeze(-1)
    zero_effort_miss = target_position - (position + velocity * time + 0.5 * gravity * time * time)
    zero_effort_velocity = target_velocity - (velocity + gravity * time)
    return zero_effort_miss, zero_effort_velocity


def zero_effort_acceleration_now(zero_effort_miss: Tensor, zero_effort_velocity: Tensor, time_to_go: Tensor) -> Tensor:
    """Return the thrust acceleration to apply now: ``6 ZEM / T^2 - 2 ZEV / T``."""
    time = time_to_go.unsqueeze(-1)
    return 6.0 * zero_effort_miss / (time * time) - 2.0 * zero_effort_velocity / time


def zero_effort_acceleration_at_arrival(
    zero_effort_miss: Tensor, zero_effort_velocity: Tensor, time_to_go: Tensor
) -> Tensor:
    """Return the thrust acceleration at arrival if the plan is flown exactly: ``-6 ZEM / T^2 + 4 ZEV / T``."""
    time = time_to_go.unsqueeze(-1)
    return -6.0 * zero_effort_miss / (time * time) + 4.0 * zero_effort_velocity / time


def planned_height_at_fraction(
    height: Tensor,
    vertical_speed: Tensor,
    vertical_gravity: float,
    acceleration_now: Tensor,
    acceleration_at_arrival: Tensor,
    time_to_go: Tensor,
    fraction: float,
) -> Tensor:
    """Return the height of the planned path after ``fraction`` of the time-to-go.

    The thrust acceleration is linear in time, so the height is a cubic polynomial.
    """
    time = fraction * time_to_go
    slope = (acceleration_at_arrival - acceleration_now) / time_to_go
    return (
        height
        + vertical_speed * time
        + 0.5 * (vertical_gravity + acceleration_now) * time * time
        + slope * time * time * time / 6.0
    )


def choose_time_to_go(
    position: Tensor,
    velocity: Tensor,
    target_position: Tensor,
    target_velocity: Tensor,
    gravity: Tensor,
    candidates: Tensor,
    minimum_vertical_thrust_acceleration: Tensor,
    maximum_vertical_thrust_acceleration: Tensor,
    maximum_tilt_tangent: float,
    minimum_height: Tensor,
    stretch_factor: float = 1.0,
    extra_time: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Pick, per environment, a candidate time-to-go whose plan respects the limits.

    First the shortest such candidate ``T_min`` is found. The result is the shortest candidate
    that respects the limits and is at least ``stretch_factor * T_min + extra_time``. The
    stretch leaves room for what the plan ignores (the time the lander needs to tilt).

    For each candidate (shape (K,)) the planned thrust acceleration is checked at the start
    and at arrival (it is linear in between, so these are its extremes): the vertical part
    must stay inside the engine's range, and the horizontal part must be reachable with a
    tilt below the limit. The planned height must not dip below ``minimum_height``.

    Returns (time_to_go (N,), feasible (N,) bool). If no candidate respects all limits, the
    candidate with the smallest violation is returned and ``feasible`` is False.
    """
    count, number_of_candidates = position.shape[0], candidates.shape[0]
    time = candidates.unsqueeze(0).expand(count, number_of_candidates)  # (N, K)

    def expand(vector: Tensor) -> Tensor:
        """Repeat a per-environment vector for every candidate: (N, 3) -> (N, K, 3)."""
        return vector.unsqueeze(1).expand(count, number_of_candidates, 3)

    miss, velocity_error = zero_effort_errors(
        expand(position), expand(velocity), expand(target_position), expand(target_velocity), gravity, time
    )
    at_start = zero_effort_acceleration_now(miss, velocity_error, time)
    at_arrival = zero_effort_acceleration_at_arrival(miss, velocity_error, time)

    lower = minimum_vertical_thrust_acceleration.unsqueeze(1)
    upper = maximum_vertical_thrust_acceleration.unsqueeze(1)
    violation = torch.zeros(count, number_of_candidates, device=position.device)
    for acceleration in (at_start, at_arrival):
        vertical = acceleration[..., 2]
        horizontal = acceleration[..., :2].norm(dim=-1)
        violation += (lower - vertical).clamp_min(0.0) + (vertical - upper).clamp_min(0.0)
        violation += (horizontal - maximum_tilt_tangent * vertical.clamp_min(0.0)).clamp_min(0.0)
    for fraction in (0.25, 0.5, 0.75):
        height = planned_height_at_fraction(
            position[:, 2:3], velocity[:, 2:3], float(gravity[2]), at_start[..., 2], at_arrival[..., 2], time, fraction
        )
        violation += (minimum_height.unsqueeze(1) - height).clamp_min(0.0)

    feasible = violation <= 1e-6
    # Shortest feasible candidate; if none is feasible, the least-violating one.
    score = torch.where(feasible, time, 1.0e6 + violation)
    shortest = candidates[score.argmin(dim=1)]
    any_feasible = feasible.any(dim=1)
    # Shortest feasible candidate that is at least the stretched time; fall back to the shortest.
    long_enough = feasible & (time >= (stretch_factor * shortest + extra_time).unsqueeze(1))
    stretched = candidates[torch.where(long_enough, time, torch.full_like(time, 1.0e6)).argmin(dim=1)]
    chosen = torch.where(long_enough.any(dim=1) & any_feasible, stretched, shortest)
    return chosen, any_feasible
