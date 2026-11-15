# SPDX-License-Identifier: BSD-3-Clause
"""EKF consistency on recorded truth (GPU, milestone 2 gate): S2 with ZEM/ZEV flying on the estimate.

256 landers fly scenario S2 (terminal_suite sensors, ``nav: ekf``) from random terminal-box starts
with the S2 first-estimate errors (1 m, 0.2 m/s, 0.1 deg per axis). Every control step the true
state and the filter's estimate and covariance are recorded, from 1 s after the start until the
first footpad contact (the impact is outside what the IMU model is meant for).

Pass criteria and why:
- Mean NEES of position, velocity and attitude (9 components) over all recorded steps lies
  between 6 and 12: a consistent filter gives 9; one whose variances are wrong by a factor of
  about 1.5 or more falls outside. The band is wider than in the synthetic CPU test because the
  flight has engine thrust steps and attitude manoeuvres, and fewer steps per lander.
- Vertical position error at contact below 0.05 m and velocity error below 0.02 m/s for 95 % of
  the landers (the touchdown values the agent relies on).
- Truth reads: the filter is not among the readers of TruthState.
"""

import pytest
import torch
from isaac_test_support import build_agent, make_environment, read_truth

from lunarsim.core.message_types import Observation
from lunarsim.core.rotation_math import quaternion_conjugate, quaternion_multiply, quaternion_to_rotation_vector

COUNT = 256
MAXIMUM_STEPS = 2500


@pytest.fixture(scope="module")
def recording():
    """Fly S2 and return per-step NEES values and the errors at the first contact."""
    environment = make_environment("s2_terminal_ekf", [f"num_envs={COUNT}", "agent=zem_zev"])
    agent = build_agent(environment)
    filter_state = environment.bindings["nav"].filter
    observations, _ = environment.reset()
    touched = torch.zeros(COUNT, dtype=torch.bool, device=environment.device)
    finished = torch.zeros_like(touched)
    nees_values, contact_errors = [], []
    with torch.inference_mode():
        for _ in range(MAXIMUM_STEPS):
            command = agent.compute_command(Observation(policy=observations["policy"]), environment.clock.control_step_duration)
            observations, _, terminated, truncated, _ = environment.step(command.to_action_tensor())
            finished |= terminated | truncated
            truth = read_truth(environment)
            attitude_error = quaternion_to_rotation_vector(
                quaternion_multiply(quaternion_conjugate(filter_state.orientation), truth.orientation_quaternion)
            )
            errors = torch.cat(
                (truth.position - filter_state.position, truth.velocity - filter_state.velocity, attitude_error), dim=1
            ).to(torch.float64)
            contact_now = (truth.contact.max(dim=1).values > 100.0) & ~touched & ~finished
            contact_errors.append(errors[contact_now])
            touched |= contact_now
            recorded = ~touched & ~finished & (truth.time > 1.0)
            if bool(recorded.any()):
                covariance = filter_state.covariance[recorded][:, :9, :9]
                chosen = errors[recorded]
                nees = (chosen.unsqueeze(1) @ torch.linalg.solve(covariance, chosen.unsqueeze(-1))).flatten()
                nees_values.append(nees)
            if bool((touched | finished).all()):
                break
    result = torch.cat(nees_values), torch.cat(contact_errors), environment.bus.truth_read_log
    yield result
    environment.close()


def test_filter_is_consistent_in_flight(recording):
    """Mean NEES (9 components) between 6 and 12 over all recorded steps."""
    nees, _, _ = recording
    print("mean NEES", float(nees.mean()), "median", float(nees.median()), "samples", nees.numel())
    assert 6.0 < float(nees.mean()) < 12.0


def test_touchdown_estimate_is_accurate(recording):
    """At the first contact: vertical error below 5 cm and velocity error below 2 cm/s for 95 %."""
    _, errors, _ = recording
    print("contacts", errors.shape[0], "vertical p95", float(errors[:, 2].abs().quantile(0.95)),
          "velocity p95", float(errors[:, 3:6].norm(dim=1).quantile(0.95)))
    assert errors.shape[0] > 0.8 * COUNT
    assert float(errors[:, 2].abs().quantile(0.95)) < 0.05
    assert float(errors[:, 3:6].norm(dim=1).quantile(0.95)) < 0.02


def test_filter_does_not_read_truth(recording):
    """Only the sensors, plant-side and judging bindings read TruthState."""
    _, _, readers = recording
    print("truth readers", readers)
    assert "ExtendedKalmanFilterNavigation" not in readers
