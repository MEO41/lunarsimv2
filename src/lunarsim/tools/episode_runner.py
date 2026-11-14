# SPDX-License-Identifier: BSD-3-Clause
"""Build an agent for a scenario and run it against the environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import Observation
from lunarsim.core.registry import get_binding_class


def build_agent(environment: Any):
    """Create the agent that the environment's scenario names in its ``agent`` slot."""
    import lunarsim.agents  # noqa: F401  (registers the agents)

    scenario = environment.scenario
    selection = scenario.slots[SlotName.AGENT]
    agent_class = get_binding_class(SlotName.AGENT, selection.binding)
    return agent_class(
        parameters=selection.parameters,
        number_of_environments=environment.num_envs,
        number_of_jets=scenario.number_of_jets,
        device=environment.device,
        shared_data=environment.binding_context.shared_data,
    )


def run_agent_in_environment(
    environment: Any,
    agent: Any,
    number_of_steps: int,
    after_step: Callable[[int], None] | None = None,
) -> None:
    """Run the closed loop Observation -> agent -> Command -> environment for some control steps.

    The agent talks to the environment only through the observation dictionary and the flat
    action tensor: the same interface a learned agent uses.
    """
    agent_generator = environment.random_generators.generator_for(SlotName.AGENT)
    all_environment_ids = torch.arange(environment.num_envs, device=environment.device)
    observations, _ = environment.reset()
    agent.reset(all_environment_ids, agent_generator)
    control_step_duration = environment.clock.control_step_duration
    for step_index in range(number_of_steps):
        with torch.inference_mode():
            command = agent.compute_command(Observation(policy=observations["policy"]), control_step_duration)
            observations, _, terminated, truncated, _ = environment.step(command.to_action_tensor())
            finished = terminated | truncated
            if bool(finished.any()):
                agent.reset(all_environment_ids[finished], agent_generator)
        if after_step is not None:
            after_step(step_index)
