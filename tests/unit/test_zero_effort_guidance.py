# SPDX-License-Identifier: BSD-3-Clause
"""ZEM/ZEV guidance law against its closed-form answers (CPU, float64)."""

import pytest
import torch

from lunarsim.agents.classical.zero_effort_guidance import (
    choose_time_to_go,
    planned_height_at_fraction,
    zero_effort_acceleration_at_arrival,
    zero_effort_acceleration_now,
    zero_effort_errors,
)

GRAVITY = torch.tensor([0.0, 0.0, -1.6242], dtype=torch.float64)
START_POSITION = torch.tensor([[12.0, -7.0, 53.0]], dtype=torch.float64)
START_VELOCITY = torch.tensor([[1.5, -2.0, -6.0]], dtype=torch.float64)
TARGET_POSITION = torch.tensor([[0.0, 0.0, 6.6]], dtype=torch.float64)
TARGET_VELOCITY = torch.tensor([[0.0, 0.0, -1.9]], dtype=torch.float64)


def test_closed_loop_guidance_reaches_target_position_and_velocity():
    """A point mass that applies the ZEM/ZEV acceleration arrives at the target state at T.

    Integration: 20,000 semi-implicit Euler steps of 1 ms; the loop stops 1 ms before arrival
    because the gains grow without bound at T. Tolerance 2 cm and 2 cm/s, set by that last
    unguided millisecond and the first-order integrator.
    """
    duration, step = 20.0, 0.001
    position, velocity = START_POSITION.clone(), START_VELOCITY.clone()
    for index in range(int(duration / step) - 1):
        time_to_go = torch.tensor([duration - index * step], dtype=torch.float64)
        miss, velocity_error = zero_effort_errors(position, velocity, TARGET_POSITION, TARGET_VELOCITY, GRAVITY, time_to_go)
        acceleration = zero_effort_acceleration_now(miss, velocity_error, time_to_go)
        velocity = velocity + (acceleration + GRAVITY) * step
        position = position + velocity * step
    assert torch.allclose(position, TARGET_POSITION, atol=0.02)
    assert torch.allclose(velocity, TARGET_VELOCITY, atol=0.02)


def test_planned_acceleration_is_linear_in_time():
    """Flying the plan, the acceleration at mid-time is the mean of the start and arrival values."""
    duration = torch.tensor([20.0], dtype=torch.float64)
    miss, velocity_error = zero_effort_errors(START_POSITION, START_VELOCITY, TARGET_POSITION, TARGET_VELOCITY, GRAVITY, duration)
    at_start = zero_effort_acceleration_now(miss, velocity_error, duration)
    at_arrival = zero_effort_acceleration_at_arrival(miss, velocity_error, duration)
    # Integrate the open-loop linear profile to mid-time, then ask the feedback law.
    step, position, velocity = 0.001, START_POSITION.clone(), START_VELOCITY.clone()
    for index in range(10_000):
        acceleration = at_start + (at_arrival - at_start) * (index * step / 20.0)
        velocity = velocity + (acceleration + GRAVITY) * step
        position = position + velocity * step
    half = torch.tensor([10.0], dtype=torch.float64)
    miss, velocity_error = zero_effort_errors(position, velocity, TARGET_POSITION, TARGET_VELOCITY, GRAVITY, half)
    at_mid = zero_effort_acceleration_now(miss, velocity_error, half)
    assert torch.allclose(at_mid, 0.5 * (at_start + at_arrival), atol=5e-3)
    planned_height = planned_height_at_fraction(
        START_POSITION[:, 2], START_VELOCITY[:, 2], float(GRAVITY[2]), at_start[:, 2], at_arrival[:, 2], duration, 0.5
    )
    assert float(planned_height) == pytest.approx(float(position[0, 2]), abs=0.02)


def test_hover_needs_exactly_minus_gravity():
    """At the target with the target velocity zero and any T, the law asks for a = -g (hover)."""
    position = torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float64)
    zero = torch.zeros(1, 3, dtype=torch.float64)
    time_to_go = torch.tensor([7.0], dtype=torch.float64)
    miss, velocity_error = zero_effort_errors(position, zero, position, zero, GRAVITY, time_to_go)
    assert torch.allclose(zero_effort_acceleration_now(miss, velocity_error, time_to_go), -GRAVITY.unsqueeze(0))


def test_time_to_go_choice_respects_thrust_and_tilt_limits():
    """The chosen T is the shortest whose plan stays inside the limits; impossible cases are flagged."""
    candidates = torch.arange(2.0, 80.5, 0.5, dtype=torch.float64)
    position = torch.cat((START_POSITION, torch.tensor([[0.0, 0.0, 8.6]], dtype=torch.float64)))
    velocity = torch.cat((START_VELOCITY, torch.tensor([[0.0, 0.0, -9.0]], dtype=torch.float64)))
    lower = torch.full((2,), 0.64, dtype=torch.float64)
    upper = torch.full((2,), 3.7, dtype=torch.float64)
    tilt_tangent = 0.176
    duration, feasible = choose_time_to_go(
        position, velocity, TARGET_POSITION.expand(2, 3), TARGET_VELOCITY.expand(2, 3), GRAVITY, candidates,
        lower, upper, tilt_tangent, minimum_height=torch.full((2,), 6.6, dtype=torch.float64),
    )
    assert feasible.tolist() == [True, False]  # 9 m/s down at 2 m above the gate cannot be stopped

    def within_limits(time_value: float) -> bool:
        """Check the plan of the first case for one time-to-go."""
        time = torch.tensor([time_value], dtype=torch.float64)
        miss, velocity_error = zero_effort_errors(START_POSITION, START_VELOCITY, TARGET_POSITION, TARGET_VELOCITY, GRAVITY, time)
        ok = True
        for acceleration in (
            zero_effort_acceleration_now(miss, velocity_error, time),
            zero_effort_acceleration_at_arrival(miss, velocity_error, time),
        ):
            vertical, horizontal = float(acceleration[0, 2]), float(acceleration[0, :2].norm())
            ok &= 0.64 <= vertical <= 3.7 and horizontal <= tilt_tangent * vertical + 1e-9
        return ok

    chosen = float(duration[0])
    assert within_limits(chosen)
    assert not within_limits(chosen - 0.5)
