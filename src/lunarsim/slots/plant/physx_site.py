# SPDX-License-Identifier: BSD-3-Clause
"""``plant: physx_site`` - the lander articulation in PhysX, in the site frame."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import Tensor

from lunarsim.assets.lander_builder import LANDER_PRIM_NAME, leg_link_name, strut_joint_name, write_lander_usd
from lunarsim.assets.lander_preset import LanderPreset
from lunarsim.core.binding import SlotName
from lunarsim.core.registry import register_binding
from lunarsim.slots.plant.physx_root_body_plant import PhysxRootBodyPlant


@register_binding(SlotName.PLANT, "physx_site")
class PhysxSitePlant(PhysxRootBodyPlant):
    """One articulation per environment: body, four legs, a prismatic spring-damper strut per leg.

    Parameters (all optional, defaults are the tuned values reported in M1A_REPORT.md):
    ``solver_position_iterations``, ``solver_velocity_iterations``, ``contact_offset_m``,
    ``rest_offset_m``, ``max_depenetration_velocity_mps``, ``contact_force_threshold_n``.
    """

    def setup_scene(self, environment) -> None:
        """Write the lander USD from the preset, spawn it, and add the leg contact sensor."""
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        preset: LanderPreset = self.context.shared_data["lander_preset"]
        self.preset = preset
        usd_path = write_lander_usd(preset, Path(tempfile.gettempdir()) / "lunarsim_assets")
        parameters = self.parameters
        lander_configuration = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/" + LANDER_PRIM_NAME,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    # PhysX adds a little damping to every body by default. Space has none.
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=float(parameters.get("max_depenetration_velocity_mps", 1.0)),
                    enable_gyroscopic_forces=True,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=int(parameters.get("solver_position_iterations", 8)),
                    solver_velocity_iteration_count=int(parameters.get("solver_velocity_iterations", 1)),
                    sleep_threshold=float(parameters.get("sleep_threshold", 0.0)),
                    stabilization_threshold=float(parameters.get("stabilization_threshold", 0.0)),
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=float(parameters.get("contact_offset_m", 0.02)),
                    rest_offset=float(parameters.get("rest_offset_m", 0.0)),
                    torsional_patch_radius=float(parameters.get("torsional_patch_radius_m", 0.0)),
                    min_torsional_patch_radius=float(parameters.get("torsional_patch_radius_m", 0.0)),
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, preset.center_of_mass_height_above_footpad_plane + 1.0),
                joint_pos={"strut_.*": 0.0},
            ),
            actuators={
                "struts": ImplicitActuatorCfg(
                    joint_names_expr=["strut_.*"],
                    stiffness=preset.strut_stiffness,
                    damping=preset.strut_damping,
                    effort_limit_sim=1.0e7,
                    velocity_limit_sim=1.0e3,
                ),
            },
        )
        self.asset = Articulation(lander_configuration)
        environment.scene.articulations["lander"] = self.asset
        self.contact_sensor = ContactSensor(
            ContactSensorCfg(prim_path="{ENV_REGEX_NS}/" + LANDER_PRIM_NAME + "/leg_.*", history_length=0)
        )
        environment.scene.sensors["leg_contact"] = self.contact_sensor

    def initialize(self, environment) -> None:
        """Resolve body, joint and sensor indices in leg order, then allocate the mass state."""
        self.parameters.setdefault("minimum_mass_kg", 0.5 * self.preset.mass_at_start)
        self.root_body_index = self.asset.find_bodies("body")[0][0]
        leg_names = [leg_link_name(index) for index in range(self.preset.number_of_legs)]
        strut_names = [strut_joint_name(index) for index in range(self.preset.number_of_legs)]
        self.strut_joint_ids = [self.asset.find_joints(name)[0][0] for name in strut_names]
        self.contact_body_ids = [self.contact_sensor.find_bodies(name)[0][0] for name in leg_names]
        super().initialize(environment)

    def _reset_joints(self, env_ids: Tensor) -> None:
        """Extend all struts fully and stop them."""
        zeros = torch.zeros(len(env_ids), len(self.strut_joint_ids), device=self.device)
        self.asset.write_joint_position_to_sim_index(position=zeros, joint_ids=self.strut_joint_ids, env_ids=env_ids)
        self.asset.write_joint_velocity_to_sim_index(velocity=zeros, joint_ids=self.strut_joint_ids, env_ids=env_ids)

    def _read_contact_forces(self) -> Tensor:
        """Return the contact force magnitude on each footpad, (N, n_legs), in leg order."""
        net_forces = self.contact_sensor.data.net_forces_w.torch  # (N, n_sensor_bodies, 3)
        return net_forces[:, self.contact_body_ids, :].norm(dim=-1)

    def strut_compression(self) -> Tensor:
        """Return how far each strut is compressed, (N, n_legs), m. Used by tests and plots."""
        return self.asset.data.joint_pos.torch[:, self.strut_joint_ids]
