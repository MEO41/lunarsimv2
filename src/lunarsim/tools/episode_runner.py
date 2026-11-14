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
    # The reset runs in inference mode like the steps: tensors made in inference mode cannot
    # be changed in place outside of it, and a second call of this function resets again.
    with torch.inference_mode():
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


def collect_episodes(
    environment: Any,
    agent: Any,
    episodes_per_environment: int = 1,
    maximum_steps: int = 100_000,
    after_step: Callable[[int, Any], None] | None = None,
):
    """Run the closed loop until every environment has finished its first episodes; return their records.

    Every environment contributes exactly ``episodes_per_environment`` rows. Without this rule,
    environments with short episodes would finish more episodes and bias the statistics.
    The scenario must use a metrics binding with ``drain_episode_records`` (``metrics: touchdown``).
    ``after_step(step_index, command)`` is called after every control step.
    """
    import pandas

    from lunarsim.slots.metrics.touchdown import OUTCOME_NAMES
    from lunarsim.tools.episode_records import records_to_frame

    metrics = environment.bindings[SlotName.METRICS]
    agent_generator = environment.random_generators.generator_for(SlotName.AGENT)
    all_environment_ids = torch.arange(environment.num_envs, device=environment.device)
    with torch.inference_mode():
        observations, _ = environment.reset()
        agent.reset(all_environment_ids, agent_generator)
    metrics.drain_episode_records()
    control_step_duration = environment.clock.control_step_duration
    wanted_rows = environment.num_envs * episodes_per_environment
    frames, collected_rows = [], 0
    for step_index in range(maximum_steps):
        with torch.inference_mode():
            command = agent.compute_command(Observation(policy=observations["policy"]), control_step_duration)
            observations, _, terminated, truncated, _ = environment.step(command.to_action_tensor())
            finished = terminated | truncated
            if bool(finished.any()):
                agent.reset(all_environment_ids[finished], agent_generator)
        if after_step is not None:
            after_step(step_index, command)
        records = metrics.drain_episode_records()
        if records:
            frame = records_to_frame(records, OUTCOME_NAMES)
            frame = frame[frame["episode_number"] < episodes_per_environment]
            frames.append(frame)
            collected_rows += len(frame)
            if collected_rows >= wanted_rows:
                break
    return pandas.concat(frames, ignore_index=True) if frames else pandas.DataFrame()
