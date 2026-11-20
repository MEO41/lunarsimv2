# SPDX-License-Identifier: BSD-3-Clause
"""AnalyticLanderEnvironment: the LanderEnv slot loop in pure torch, without Isaac Sim.

It runs a scenario whose plant is an analytic binding (``plant: analytic_rigid_body``). Every
other slot binding (start, surface, sensors, faults, navigation, actuators, metrics), the reward
and the observation are the same objects that ``LanderEnv`` uses, called in the same order:

1. ``_pre_physics_step``  agent output -> Command; actuators take the command
2. ``_apply_action``      actuators + surface -> Wrench -> plant (repeats every physics substep)
3. ``_get_dones``         plant state is read, then metrics
4. ``_get_rewards``       reward from the scenario's ``training.reward`` section (zero without it)
5. ``_reset_idx``         start, then every slot resets (only finished environments)
6. ``_get_observations``  plant, sensors, faults, navigation -> Observation

``step`` and ``reset`` follow Isaac Lab's ``DirectRLEnv.step`` and ``DirectRLEnv.reset`` (Isaac
Lab 3.0.0, commit e86463d, PhysX backend: one ``_apply_action`` per physics substep), so a policy
sees the same sequence of observations, rewards and resets in both simulators. Only the plant
differs, which is the point of the transfer study.

The object offers the attributes that LunarSim's tools read from a LanderEnv (``scenario``,
``bindings``, ``bus``, ``clock``, ``random_generators``, ``binding_context``, ``num_envs``,
``device``, ``cfg``, ``episode_length_buf``, ``max_episode_length``, ``reward_terms``), so
``LanderVectorEnvironment``, ``collect_episodes`` and the trajectory log work unchanged.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy
import torch

from lunarsim.core.binding import BindingContext, SlotName
from lunarsim.core.message_bus import MessageBus
from lunarsim.core.message_types import (
    Command,
    EpisodeEvents,
    MessageKey,
    NavEstimate,
    NavigationStateLayout,
    Observation,
)
from lunarsim.core.random_generators import SlotRandomGenerators
from lunarsim.core.registry import ObservationBuilderDeclaration, check_slot_loop
from lunarsim.core.simulation_clock import SimulationClock
from lunarsim.env.scenario_loader import ScenarioConfig
from lunarsim.env.slot_loop_parts import (
    SLOT_RESET_ORDER,
    _AgentPort,
    _EpisodeEventsReader,
    _TrainingTruthReader,
    truth_observation_vector,
)

# Surface bindings that the analytic environment can run: they must not need a simulator scene.
SURFACES_WITHOUT_A_SCENE = ("flat",)


@dataclass
class AnalyticEnvironmentConfig:
    """The few settings that tools read from ``environment.cfg`` (the counterpart of LanderEnvCfg)."""

    seed: int
    decimation: int
    episode_length_s: float
    action_space: int
    observation_space: int
    physics_step_s: float
    scenario: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a plain dictionary (rsl_rl's loggers call this)."""
        return asdict(self)


@dataclass
class AnalyticScene:
    """Stand-in for the Isaac Lab scene: the environment origins that a surface binding reads.

    The analytic environments never touch each other, so all of them sit at the world origin.
    """

    num_envs: int
    env_origins: torch.Tensor


class AnalyticLanderEnvironment:
    """Runs the eight environment slots of a scenario with an analytic plant, in batched torch."""

    RESET_ORDER = SLOT_RESET_ORDER

    def __init__(self, scenario: ScenarioConfig, device: str | None = None) -> None:
        """Build the clock, the bus and the bindings of ``scenario``, and allocate the step buffers."""
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.scenario = scenario
        self.num_envs = scenario.number_of_environments
        self.render_mode = None
        self.physics_dt = scenario.physics_step_duration
        self.step_dt = scenario.control_step_duration
        self.max_episode_length_s = scenario.episode_length_seconds
        self.max_episode_length = math.ceil(self.max_episode_length_s / self.step_dt)
        self.cfg = AnalyticEnvironmentConfig(
            seed=scenario.root_seed,
            decimation=scenario.physics_steps_per_control_step,
            episode_length_s=scenario.episode_length_seconds,
            action_space=Command.action_dimension(scenario.number_of_jets),
            observation_space=NavigationStateLayout.DIMENSION,
            physics_step_s=scenario.physics_step_duration,
            scenario=scenario.resolved,
        )
        self.seed(scenario.root_seed)

        binding_classes = scenario.binding_classes()
        plant_class = binding_classes[SlotName.PLANT]
        if getattr(plant_class, "simulator", "isaac") != "analytic":
            raise ValueError(
                f"AnalyticLanderEnvironment runs analytic plants only; plant '{scenario.slots[SlotName.PLANT].binding}' "
                "needs Isaac Sim (use LanderEnv)"
            )
        surface_name = scenario.slots[SlotName.SURFACE].binding
        if surface_name not in SURFACES_WITHOUT_A_SCENE:
            raise ValueError(f"The analytic environment runs on surface: flat only, not '{surface_name}'")
        check_slot_loop(binding_classes)

        self.clock = SimulationClock(
            self.num_envs,
            physics_step_duration=scenario.physics_step_duration,
            physics_steps_per_control_step=scenario.physics_steps_per_control_step,
            device=device,
        )
        self.bus = MessageBus()
        self.random_generators = SlotRandomGenerators(scenario.root_seed, SlotName.ALL, device=device)
        lander_preset = None
        if scenario.lander:
            from lunarsim.assets.lander_preset import lander_preset_from_dictionary

            lander_preset = lander_preset_from_dictionary(scenario.lander)
        self.binding_context = BindingContext(
            number_of_environments=self.num_envs,
            device=device,
            clock=self.clock,
            bus=self.bus,
            shared_data={
                "lander": scenario.lander,
                "lander_preset": lander_preset,
                "episode_length_seconds": scenario.episode_length_seconds,
                "control_step_duration": scenario.control_step_duration,
                "sensor_suite_parameters": dict(scenario.slots[SlotName.SENSORS].parameters),
                # The analytic plant takes its footpad friction from the surface slot.
                "surface_parameters": dict(scenario.slots[SlotName.SURFACE].parameters),
            },
        )
        self.bindings = {
            slot: binding_classes[slot](scenario.slots[slot].parameters, self.binding_context)
            for slot in SlotName.INSIDE_ENVIRONMENT
        }
        self._agent_port = _AgentPort()
        self._observation_builder = ObservationBuilderDeclaration()
        self._episode_events_reader = _EpisodeEventsReader()
        self.scene = AnalyticScene(num_envs=self.num_envs, env_origins=torch.zeros(self.num_envs, 3, device=device))
        # No setup_scene: there is no simulator scene. initialize() publishes what the slots need.
        for slot in SlotName.INSIDE_ENVIRONMENT:
            self.bindings[slot].initialize(self)

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros_like(self.reset_terminated)
        self.common_step_counter = 0
        self.extras: dict[str, Any] = {}

        self._zero_reward = torch.zeros(self.num_envs, device=device)
        self._training_truth_reader = _TrainingTruthReader()
        self.include_navigation_health = bool((scenario.resolved.get("observation") or {}).get("include_navigation_health", False))
        if self.include_navigation_health:
            raise ValueError("observation.include_navigation_health is not supported by the analytic environment")
        training = dict(scenario.resolved.get("training") or {})
        self.critic_observation = str(training.get("critic_observation", "none"))
        if self.critic_observation not in ("none", "truth"):
            raise ValueError(f"training.critic_observation must be 'none' or 'truth', not '{self.critic_observation}'")
        self.reward_terms = None
        if training.get("reward") is not None:
            from lunarsim.env.reward_terms import LandingRewardTerms

            self.reward_terms = LandingRewardTerms(
                dict(training["reward"]), lander_preset, self.num_envs, device, scenario.control_step_duration
            )
        self.last_reward_terms: dict[str, torch.Tensor] = {}

    # ---- the DirectRLEnv-like interface ----

    @property
    def unwrapped(self) -> AnalyticLanderEnvironment:
        """Return the environment itself (gymnasium-style access used by some tools)."""
        return self

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Seed Python, NumPy and torch, as DirectRLEnv.seed does, and return the seed."""
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        return seed

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict, dict]:
        """Reset every environment and return the first observation groups and the extras."""
        if seed is not None:
            self.seed(seed)
        self._reset_idx(torch.arange(self.num_envs, dtype=torch.long, device=self.device))
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.extras

    def step(self, action: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance one control step (``decimation`` physics substeps) in the order of DirectRLEnv.step."""
        action = action.to(self.device)
        self._pre_physics_step(action)
        for _ in range(self.cfg.decimation):
            self._apply_action()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def close(self) -> None:
        """Nothing to release: there is no simulator."""

    # ---- hook 1 ----
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Turn the agent's flat action tensor into a Command and hand it to the actuators."""
        command = Command.from_action_tensor(actions)
        self.bus.write(MessageKey.COMMAND, command, self._agent_port)
        self.bindings[SlotName.ACTUATORS].begin_control_step(self.bus, self.clock.control_step_duration)

    # ---- hook 2 (every physics substep) ----
    def _apply_action(self) -> None:
        """Actuators and surface write their wrenches; the plant integrates one substep."""
        physics_step_duration = self.clock.physics_step_duration
        self.bindings[SlotName.ACTUATORS].step(self.bus, physics_step_duration)
        self.bindings[SlotName.SURFACE].step(self.bus, physics_step_duration)
        self.bindings[SlotName.PLANT].apply_wrench(self.bus, physics_step_duration)
        self.clock.advance_one_physics_step()

    # ---- hook 3 ----
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Publish the plant state, then let the metrics decide which episodes ended."""
        control_step_duration = self.clock.control_step_duration
        self.bindings[SlotName.PLANT].step(self.bus, control_step_duration)
        self.bindings[SlotName.METRICS].step(self.bus, control_step_duration)
        events: EpisodeEvents = self.bus.read(MessageKey.EPISODE_EVENTS, self._episode_events_reader)
        time_limit_reached = self.episode_length_buf >= self.max_episode_length
        return events.terminated, events.truncated | time_limit_reached

    # ---- hook 4 ----
    def _get_rewards(self) -> torch.Tensor:
        """Return the landing reward of this control step, or zero when the scenario has none."""
        if self.reward_terms is None:
            return self._zero_reward
        truth = self.bus.read(MessageKey.TRUTH_STATE, self._training_truth_reader)
        events: EpisodeEvents = self.bus.read(MessageKey.EPISODE_EVENTS, self._training_truth_reader)
        reward, self.last_reward_terms = self.reward_terms.compute(truth, events)
        return reward

    # ---- hook 5 ----
    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        """Reset finished environments: the start slot samples, then every slot resets."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.episode_length_buf[env_ids] = 0
        self.clock.reset(env_ids)
        start_binding = self.bindings[SlotName.START]
        start_binding.reset(env_ids, self.random_generators.generator_for(SlotName.START))
        start_binding.step(self.bus, 0.0)
        for slot in self.RESET_ORDER:
            self.bindings[slot].reset(env_ids, self.random_generators.generator_for(slot))
        if self.reward_terms is not None:
            self.reward_terms.reset(env_ids)

    # ---- hook 6 ----
    def _get_observations(self) -> dict:
        """Run the measurement chain and build the policy observation from the estimate."""
        control_step_duration = self.clock.control_step_duration
        for slot in (SlotName.PLANT, SlotName.SENSORS, SlotName.FAULTS, SlotName.NAVIGATION):
            self.bindings[slot].step(self.bus, control_step_duration)
        estimate: NavEstimate = self.bus.read(MessageKey.NAVIGATION_ESTIMATE, self._observation_builder)
        observation = Observation(policy=estimate.mean, site_map=estimate.site_map)
        self.bus.write(MessageKey.OBSERVATION, observation, self._observation_builder)
        groups = {"policy": observation.policy}
        if estimate.site_map is not None:
            groups["site_map"] = estimate.site_map
        if self.critic_observation == "truth":
            truth = self.bus.read(MessageKey.TRUTH_STATE, self._training_truth_reader)
            groups["critic"] = truth_observation_vector(truth)
        return groups
