# SPDX-License-Identifier: BSD-3-Clause
"""Isaac Lab config class of the lander environment, built from a ScenarioConfig."""

from __future__ import annotations

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from lunarsim.core.message_types import Command, NavigationStateLayout
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.env.scenario_loader import ScenarioConfig


@configclass
class LanderEnvCfg(DirectRLEnvCfg):
    """Config of :class:`~lunarsim.env.lander_env.LanderEnv`.

    Only simulator settings live here. Everything about the slots lives in ``scenario``, the
    resolved scenario dictionary.
    """

    decimation: int = 4
    episode_length_s: float = 60.0
    action_space: int = 3
    observation_space: int = NavigationStateLayout.DIMENSION
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        gravity=(0.0, 0.0, -MOON_SURFACE_GRAVITY),
        physics=PhysxCfg(),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=20.0, replicate_physics=True)

    # The resolved scenario (plain dictionary), set by build_lander_env_cfg.
    scenario: dict = {}


def build_lander_env_cfg(scenario: ScenarioConfig, device: str = "cuda:0") -> LanderEnvCfg:
    """Return the Isaac Lab config for ``scenario``."""
    configuration = LanderEnvCfg()
    configuration.seed = scenario.root_seed
    configuration.decimation = scenario.physics_steps_per_control_step
    configuration.episode_length_s = scenario.episode_length_seconds
    configuration.action_space = Command.action_dimension(scenario.number_of_jets)
    configuration.observation_space = NavigationStateLayout.DIMENSION
    configuration.sim.device = device
    configuration.sim.dt = scenario.physics_step_duration
    configuration.sim.render_interval = scenario.physics_steps_per_control_step
    # Bodies without their own material (the lander) get this default material. PhysX combines
    # two materials with the mode of higher priority (multiply beats average), so a default of
    # 0.5 would halve the ground's friction. With 1.0 and "multiply", the coefficient that the
    # surface binding sets (from its source) is the one that acts at the footpads.
    configuration.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="min",
    )
    configuration.scene.num_envs = scenario.number_of_environments
    configuration.scene.env_spacing = scenario.environment_spacing
    # Scene-level PhysX settings come from the plant binding; ``env.physx`` can override them.
    physics_overrides = dict(scenario.slots["plant"].parameters.get("physx_scene", {}) or {})
    physics_overrides.update(scenario.resolved.get("env", {}).get("physx", {}) or {})
    for key, value in physics_overrides.items():
        if not hasattr(configuration.sim.physics, key):
            raise ValueError(f"Unknown PhysX setting '{key}' in the scenario file")
        setattr(configuration.sim.physics, key, value)
    configuration.scenario = scenario.resolved
    return configuration
