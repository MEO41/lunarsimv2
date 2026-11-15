# SPDX-License-Identifier: BSD-3-Clause
"""rsl_rl vector-environment wrapper around ``LanderEnv`` (milestone 1b task 1).

It changes nothing inside the environment. The policy decides every ``policy_period_steps``
control steps (agent parameter; 5 = 10 Hz guidance over the 50 Hz attitude loop). It maps the
policy's actions to the environment's
action tensor and scales the observation groups (:mod:`policy_action_adapter`). The time limit
is a judged failure, so it is reported as a real end (no ``time_outs``). It logs a few landing
numbers for the training log:

- ``/lunarsim/success_rate``: share of successful episodes among those that ended, averaged over
  the last ``LOG_WINDOW_EPISODES`` finished episodes.
- ``/lunarsim/reward_<term>``: mean per-episode sum of each reward term over the same window.
"""

from __future__ import annotations

import math
from collections import deque

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from lunarsim.core.message_types import MessageKey
from lunarsim.agents.rsl_rl.policy_action_adapter import PolicyActionAdapter
from lunarsim.slots.metrics.touchdown import OUTCOME_CODE

LOG_WINDOW_EPISODES = 4096


class LanderVectorEnvironment(VecEnv):
    """Presents a LanderEnv with a ``training:`` section as an rsl_rl VecEnv."""

    def __init__(self, environment, agent_parameters: dict) -> None:
        """Wrap the environment, check it has a reward, and reset it once."""
        if environment.reward_terms is None:
            raise ValueError("The scenario has no training.reward section: give training=landing_ppo")
        self.environment = environment
        self.num_envs = environment.num_envs
        self.device = environment.device
        self.adapter = PolicyActionAdapter.from_agent_parameters(
            environment.binding_context.shared_data["lander_preset"], self.device, agent_parameters
        )
        self.num_actions = self.adapter.action_dimension
        self.control_step_duration = environment.scenario.control_step_duration
        self.policy_period_steps = int(agent_parameters.get("policy_period_steps", 1))
        # rsl_rl counts decisions; one decision spans policy_period_steps control steps.
        self.max_episode_length = int(math.ceil(environment.max_episode_length / self.policy_period_steps))
        self.recent_success = deque(maxlen=LOG_WINDOW_EPISODES)
        self.recent_terms = {name: deque(maxlen=LOG_WINDOW_EPISODES) for name in environment.reward_terms.term_names}
        observations, _ = environment.reset()
        self._observations = self._to_tensor_dictionary(observations)

    @property
    def cfg(self):
        """Return the environment's config (the runner logs it)."""
        return self.environment.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Return the environment's per-environment step counter."""
        return self.environment.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        """Set the step counter (rsl_rl does this for random episode starts)."""
        self.environment.episode_length_buf = value

    def _to_tensor_dictionary(self, observations: dict) -> TensorDict:
        """Scale every observation group and pack the groups into a TensorDict."""
        self._raw_policy_observation = observations["policy"]
        groups = {name: self.adapter.scale_observation(values) for name, values in observations.items()}
        return TensorDict(groups, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        """Return the latest observation groups."""
        return self._observations

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Hold the policy action for ``policy_period_steps`` control steps and return the summed reward.

        The attitude controller turns the held action into jet firings at every control step,
        from the newest observation. An environment whose episode ends inside the period keeps
        only the reward up to its end; it restarts at once (Isaac Lab resets finished
        environments inside ``step``) and flies the rest of the period with the held action,
        whose reward is not counted. Its next decision is the first one of the new episode.
        """
        total_reward = torch.zeros(self.num_envs, device=self.device)
        ended = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for _ in range(self.policy_period_steps):
            environment_actions = self.adapter.environment_actions(actions, self._raw_policy_observation, self.control_step_duration)
            observations, rewards, terminated, truncated, _ = self.environment.step(environment_actions)
            self._raw_policy_observation = observations["policy"]
            total_reward += torch.where(ended, torch.zeros_like(rewards), rewards)
            just_ended = (terminated | truncated) & ~ended
            if bool(just_ended.any()):
                self._record_finished_episodes(just_ended)
            ended |= just_ended
        self._observations = self._to_tensor_dictionary(observations)
        log = {}
        if self.recent_success:
            log["/lunarsim/success_rate"] = sum(self.recent_success) / len(self.recent_success)
        for name, values in self.recent_terms.items():
            if values:
                log[f"/lunarsim/reward_{name}"] = sum(values) / len(values)
        # The time limit is judged as a failure (timeout_no_contact), so it is a real end of the
        # episode for PPO: no time_outs, no bootstrapping of the value past it.
        extras = {"time_outs": torch.zeros_like(ended), "log": log}
        return self._observations, total_reward, ended.to(torch.long), extras

    def _record_finished_episodes(self, finished: torch.Tensor) -> None:
        """Keep the outcome and reward sums of episodes that ended in this control step (for the log).

        Inside ``step()`` the reward is computed before finished environments reset, so the
        metrics' events and the reward's last episode sums still describe the ended episodes.
        """
        events = self.environment.bus.read(MessageKey.EPISODE_EVENTS, self.environment._training_truth_reader)
        if events.outcome_code is not None:
            self.recent_success.extend((events.outcome_code[finished] == OUTCOME_CODE["success"]).float().tolist())
        for name, values in self.environment.reward_terms.last_episode_sums.items():
            self.recent_terms[name].extend(values[finished].tolist())

    def reset(self) -> tuple[TensorDict, dict]:
        """Reset all environments and return the first observations."""
        observations, extras = self.environment.reset()
        self._observations = self._to_tensor_dictionary(observations)
        return self._observations, extras

    def close(self) -> None:
        """Close the environment."""
        self.environment.close()
