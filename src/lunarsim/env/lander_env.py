# SPDX-License-Identifier: BSD-3-Clause
"""LanderEnv: the slot loop inside the Isaac Lab DirectRLEnv hooks.

Hook order of one control step (from the design, "One control step"):

1. ``_pre_physics_step``  agent output -> Command; actuators take the command
2. ``_apply_action``      actuators + surface -> Wrench -> plant (repeats every PhysX substep)
3. ``_get_dones``         plant state is read, then metrics
4. ``_get_rewards``       reward from the scenario's ``training.reward`` section (zero without it)
5. ``_reset_idx``         start, then every slot resets (only finished environments)
6. ``_get_observations``  plant, sensors, faults, navigation -> Observation

Training (milestone 1b): with a ``training:`` section in the scenario, ``_get_rewards`` returns
the landing reward (``reward_terms.py``), and ``training.critic_observation: truth`` adds a
``critic`` observation group built from ``TruthState`` through a logged truth path (asymmetric
actor-critic, design section "Message bus"). The actor's ``policy`` group never changes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.envs import DirectRLEnv

from lunarsim.core.binding import BindingContext, SlotName
from lunarsim.core.message_bus import MessageBus
from lunarsim.core.message_types import Command, EpisodeEvents, MessageKey, NavEstimate, Observation
from lunarsim.core.random_generators import SlotRandomGenerators
from lunarsim.core.registry import ObservationBuilderDeclaration, check_slot_loop
from lunarsim.core.simulation_clock import SimulationClock
from lunarsim.env.lander_env_cfg import LanderEnvCfg
from lunarsim.env.scenario_loader import ScenarioConfig


class _AgentPort:
    """Lets the environment write the agent's Command to the bus on the agent's behalf."""

    slot = SlotName.AGENT
    requires = frozenset({MessageKey.OBSERVATION})
    provides = frozenset({MessageKey.COMMAND})
    reads_truth = False


class _EpisodeEventsReader:
    """Lets the environment read the metrics output to fill the done flags."""

    requires = frozenset({MessageKey.EPISODE_EVENTS})
    provides = frozenset()
    reads_truth = False


class _TrainingTruthReader:
    """Training-side reader of TruthState (reward and critic group); its truth reads are logged."""

    requires = frozenset({MessageKey.TRUTH_STATE, MessageKey.EPISODE_EVENTS})
    provides = frozenset()
    reads_truth = True


def truth_observation_vector(truth) -> torch.Tensor:
    """Return the critic's view of the truth in the NavigationStateLayout order, (N, 16)."""
    return torch.cat(
        (
            truth.position,
            truth.velocity,
            truth.orientation_quaternion,
            truth.angular_velocity,
            truth.mass.unsqueeze(1),
            truth.footpad_height_above_ground.unsqueeze(1),
            truth.surface_probe_contact.unsqueeze(1),
        ),
        dim=1,
    ).to(torch.float32)


class LanderEnv(DirectRLEnv):
    """Runs the eight environment slots in the order of the design."""

    cfg: LanderEnvCfg

    # Slots that reset after the start slot, in this order.
    # The surface comes first: it may move the site of an environment, and the plant places the
    # lander relative to that site.
    RESET_ORDER = (
        SlotName.SURFACE,
        SlotName.PLANT,
        SlotName.SENSORS,
        SlotName.FAULTS,
        SlotName.NAVIGATION,
        SlotName.ACTUATORS,
        SlotName.METRICS,
    )

    def __init__(self, cfg: LanderEnvCfg, render_mode: str | None = None, **kwargs):
        """Build the simulation, then let every binding allocate its tensors."""
        super().__init__(cfg, render_mode, **kwargs)
        for slot in SlotName.INSIDE_ENVIRONMENT:
            self.bindings[slot].initialize(self)
        self._zero_reward = torch.zeros(self.num_envs, device=self.device)
        self._training_truth_reader = _TrainingTruthReader()
        training = dict(self.scenario.resolved.get("training") or {})
        self.critic_observation = str(training.get("critic_observation", "none"))
        if self.critic_observation not in ("none", "truth"):
            raise ValueError(f"training.critic_observation must be 'none' or 'truth', not '{self.critic_observation}'")
        self.reward_terms = None
        if training.get("reward") is not None:
            from lunarsim.env.reward_terms import LandingRewardTerms

            self.reward_terms = LandingRewardTerms(
                dict(training["reward"]),
                self.binding_context.shared_data["lander_preset"],
                self.num_envs,
                self.device,
                self.scenario.control_step_duration,
            )
        self.last_reward_terms: dict[str, torch.Tensor] = {}

    def _setup_scene(self) -> None:
        """Build the bindings, check the loop, and let the bindings create the scene."""
        self.scenario = ScenarioConfig.from_resolved(self.cfg.scenario)
        number_of_environments = self.scene.num_envs
        self.clock = SimulationClock(
            number_of_environments,
            physics_step_duration=self.scenario.physics_step_duration,
            physics_steps_per_control_step=self.scenario.physics_steps_per_control_step,
            device=self.device,
        )
        self.bus = MessageBus()
        self.random_generators = SlotRandomGenerators(self.scenario.root_seed, SlotName.ALL, device=self.device)
        lander_preset = None
        if self.scenario.lander:
            from lunarsim.assets.lander_preset import lander_preset_from_dictionary

            lander_preset = lander_preset_from_dictionary(self.scenario.lander)
        self.binding_context = BindingContext(
            number_of_environments=number_of_environments,
            device=self.device,
            clock=self.clock,
            bus=self.bus,
            shared_data={
                "lander": self.scenario.lander,
                "lander_preset": lander_preset,
                "episode_length_seconds": self.scenario.episode_length_seconds,
                "control_step_duration": self.scenario.control_step_duration,
                # Navigation knows how its sensors are mounted (beam directions, ranges).
                "sensor_suite_parameters": dict(self.scenario.slots[SlotName.SENSORS].parameters),
            },
        )
        binding_classes = self.scenario.binding_classes()
        check_slot_loop(binding_classes)
        self.bindings = {
            slot: binding_classes[slot](self.scenario.slots[slot].parameters, self.binding_context)
            for slot in SlotName.INSIDE_ENVIRONMENT
        }
        self._agent_port = _AgentPort()
        self._observation_builder = ObservationBuilderDeclaration()
        self._episode_events_reader = _EpisodeEventsReader()

        # Prims that exist once for all environments (the ground). Bindings append to this list.
        self.global_prim_paths: list[str] = []
        self.bindings[SlotName.SURFACE].setup_scene(self)
        self.bindings[SlotName.PLANT].setup_scene(self)
        for slot in SlotName.INSIDE_ENVIRONMENT:
            if slot not in (SlotName.SURFACE, SlotName.PLANT):
                self.bindings[slot].setup_scene(self)

        source_path, destination_path = "/World/envs/env_0", "/World/envs/env_{}"
        positions = cloner.grid_transforms(number_of_environments, self.scene.cfg.env_spacing)[0]
        plan = cloner.clone_plan_from_env_0(
            source_path,
            destination_path,
            number_of_environments,
            positions,
            global_paths=tuple(self.global_prim_paths),
        )
        cloner.replicate(plan)
        if "physx" not in self.scene.physics_backend:
            raise RuntimeError(f"LunarSim needs the PhysX backend, got '{self.scene.physics_backend}'")
        # Environments must not collide with each other; all of them collide with global prims.
        self.scene.filter_collisions(global_prim_paths=list(self.global_prim_paths))

        light_configuration = sim_utils.DistantLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        light_configuration.func("/World/Light", light_configuration)

    # ---- hook 1 ----
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Turn the agent's flat action tensor into a Command and hand it to the actuators."""
        command = Command.from_action_tensor(actions)
        self.bus.write(MessageKey.COMMAND, command, self._agent_port)
        self.bindings[SlotName.ACTUATORS].begin_control_step(self.bus, self.clock.control_step_duration)

    # ---- hook 2 (every PhysX substep) ----
    def _apply_action(self) -> None:
        """Actuators and surface write their wrenches; the plant applies them to PhysX."""
        physics_step_duration = self.clock.physics_step_duration
        self.bindings[SlotName.ACTUATORS].step(self.bus, physics_step_duration)
        self.bindings[SlotName.SURFACE].step(self.bus, physics_step_duration)
        self.bindings[SlotName.PLANT].apply_wrench(self.bus, physics_step_duration)
        self.clock.advance_one_physics_step()

    # ---- hook 3 ----
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Refresh the plant state after physics, then let the metrics decide which episodes ended."""
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
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """Reset finished environments: the start slot samples, then every slot resets."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        self.clock.reset(env_ids)
        start_binding = self.bindings[SlotName.START]
        start_binding.reset(env_ids, self.random_generators.generator_for(SlotName.START))
        start_binding.step(self.bus, 0.0)
        for slot in self.RESET_ORDER:
            self.bindings[slot].reset(env_ids, self.random_generators.generator_for(slot))
        if getattr(self, "reward_terms", None) is not None:
            self.reward_terms.reset(env_ids)

    # ---- hook 6 ----
    def _get_observations(self) -> dict:
        """Run the measurement chain and build the policy observation from the estimate."""
        control_step_duration = self.clock.control_step_duration
        for slot in (SlotName.PLANT, SlotName.SENSORS, SlotName.FAULTS, SlotName.NAVIGATION):
            self.bindings[slot].step(self.bus, control_step_duration)
        estimate: NavEstimate = self.bus.read(MessageKey.NAVIGATION_ESTIMATE, self._observation_builder)
        observation = Observation(policy=estimate.mean)
        self.bus.write(MessageKey.OBSERVATION, observation, self._observation_builder)
        groups = {"policy": observation.policy}
        if getattr(self, "critic_observation", "none") == "truth":
            truth = self.bus.read(MessageKey.TRUTH_STATE, self._training_truth_reader)
            groups["critic"] = truth_observation_vector(truth)
        return groups
