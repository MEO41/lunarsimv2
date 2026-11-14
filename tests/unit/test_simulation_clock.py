# SPDX-License-Identifier: BSD-3-Clause
"""SimulationClock: one clock, per-environment reset, no drift."""

import pytest
import torch

from lunarsim.core.simulation_clock import SimulationClock


def test_clock_advances_all_environments_and_resets_some():
    """Time advances by one physics step for all envs; reset touches only the chosen envs."""
    clock = SimulationClock(3, physics_step_duration=0.005, physics_steps_per_control_step=4)
    assert clock.control_step_duration == pytest.approx(0.02)
    for _ in range(8):
        clock.advance_one_physics_step()
    assert torch.allclose(clock.time, torch.full((3,), 0.04))
    clock.reset(torch.tensor([1]))
    assert clock.time.tolist() == pytest.approx([0.04, 0.0, 0.04])


def test_clock_does_not_drift_over_many_steps():
    """Time is an integer step count times the step, so 200,000 steps give exactly 1000 s."""
    clock = SimulationClock(1, physics_step_duration=0.005, physics_steps_per_control_step=4)
    clock.physics_step_count.add_(200_000)
    assert float(clock.time[0]) == pytest.approx(1000.0, abs=1e-3)


def test_clock_rejects_bad_configuration():
    """Zero or negative step sizes are configuration errors."""
    with pytest.raises(ValueError):
        SimulationClock(1, physics_step_duration=0.0, physics_steps_per_control_step=4)
    with pytest.raises(ValueError):
        SimulationClock(1, physics_step_duration=0.005, physics_steps_per_control_step=0)
