# SPDX-License-Identifier: BSD-3-Clause
"""Milestone 0 gate: the empty slot loop steps 1,024 environments without error (GPU)."""

import pytest
import torch
from isaac_test_support import build_agent, make_environment, read_truth, step_with_agent

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import NavigationStateLayout

NUMBER_OF_ENVIRONMENTS = 1024


@pytest.fixture(scope="module")
def environment():
    """Build the null-loop environment once for all tests in this file."""
    environment = make_environment("m0_null_loop", [f"num_envs={NUMBER_OF_ENVIRONMENTS}"])
    yield environment
    environment.close()


def test_reset_puts_every_box_at_the_start_state(environment):
    """After reset, the observation shows the configured start position for all environments."""
    observations, _ = environment.reset()
    policy = observations["policy"]
    assert policy.shape == (NUMBER_OF_ENVIRONMENTS, NavigationStateLayout.DIMENSION)
    expected_position = torch.tensor([0.0, 0.0, 2.0], device=policy.device)
    # Tolerance 1 mm: positions are float32 and environment origins are up to ~130 m away.
    assert torch.allclose(policy[:, 0:3], expected_position.expand_as(policy[:, 0:3]), atol=1e-3)
    assert torch.allclose(policy[:, 6:10], torch.tensor([1.0, 0.0, 0.0, 0.0], device=policy.device).expand(NUMBER_OF_ENVIRONMENTS, 4), atol=1e-5)


def test_hooks_call_slots_in_design_order(environment):
    """One control step calls the slots in the order of the 'One control step' diagram."""
    calls = []

    def record(slot, method_name):
        """Wrap one binding method so that each call is appended to ``calls``."""
        binding = environment.bindings[slot]
        original = getattr(binding, method_name)

        def wrapper(*arguments, **keyword_arguments):
            calls.append(f"{slot}.{method_name}")
            return original(*arguments, **keyword_arguments)

        setattr(binding, method_name, wrapper)
        return lambda: setattr(binding, method_name, original)

    restore = [record(slot, "step") for slot in SlotName.INSIDE_ENVIRONMENT]
    restore += [record(slot, "reset") for slot in SlotName.INSIDE_ENVIRONMENT]
    restore.append(record(SlotName.ACTUATORS, "begin_control_step"))
    restore.append(record(SlotName.PLANT, "apply_wrench"))
    try:
        agent = build_agent(environment)
        observations, _ = environment.reset()
        calls.clear()
        step_with_agent(environment, agent, observations, number_of_steps=1)
    finally:
        for undo in restore:
            undo()

    substep = ["actuators.step", "surface.step", "plant.apply_wrench"]
    expected = (
        ["actuators.begin_control_step"]  # hook 1
        + substep * environment.cfg.decimation  # hook 2, every PhysX substep
        + ["plant.step", "metrics.step"]  # hook 3 (plant read added so metrics see fresh state)
        + ["plant.step", "sensors.step", "faults.step", "nav.step"]  # hook 6 (no reset this step)
    )
    assert calls == expected


def test_1024_environments_step_100_times_and_boxes_land(environment):
    """100 control steps run without error; boxes fall and rest on the ground."""
    agent = build_agent(environment)
    observations, _ = environment.reset()
    observations = step_with_agent(environment, agent, observations, number_of_steps=100)
    truth = read_truth(environment)
    assert torch.isfinite(observations["policy"]).all()
    # A 1 m box rests with its centre 0.5 m above the ground. 2 cm covers the contact offset.
    assert torch.allclose(truth.position[:, 2], torch.full_like(truth.position[:, 2], 0.5), atol=0.02)
    assert float(truth.velocity.abs().max()) < 0.05


def test_time_limit_resets_all_environments(environment):
    """After the 5 s episode length, every environment resets to the start state and time zero."""
    agent = build_agent(environment)
    observations, _ = environment.reset()
    step_with_agent(environment, agent, observations, number_of_steps=int(5.0 / 0.02) + 1)
    truth = read_truth(environment)
    assert float(truth.time.max()) < 0.1
    assert torch.allclose(truth.position[:, 2], torch.full_like(truth.position[:, 2], 2.0), atol=0.01)
