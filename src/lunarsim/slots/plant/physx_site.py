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
from lunarsim.core.message_types import MessageKey, SurfaceGeometry
from lunarsim.core.registry import register_binding
from lunarsim.core.rotation_math import rotate_vector_body_to_parent
from lunarsim.slots.plant.physx_root_body_plant import PhysxRootBodyPlant


@register_binding(SlotName.PLANT, "physx_site")
class PhysxSitePlant(PhysxRootBodyPlant):
    """One articulation per environment: body, four legs, a prismatic spring-damper strut per leg.

    Three legs carry a surface probe (preset ``legs_with_surface_probes``). The probes have no
    collision shape; ``TruthState.surface_probe_contact`` is 1.0 while a probe tip is at or below
    the ground lookup of the surface slot.

    Parameters (all optional, defaults are the tuned values reported in M1A_REPORT.md):
    ``solver_position_iterations``, ``solver_velocity_iterations``, ``contact_offset_m``,
    ``rest_offset_m``, ``max_depenetration_velocity_mps``, ``contact_force_threshold_n``,
    ``spawn_height_m`` (height of the first spawn above the ground, before the first reset),
    ``footpad_contact`` (``physx``, the default, or ``soil``).

    With ``footpad_contact: soil`` (milestone 5) the footpads have no PhysX contact. The surface
    slot's soil model (``surface: regolith_plume_l1``) computes a force on every footpad each
    PhysX substep from :meth:`footpad_states`, and the plant applies those forces to the leg links
    (``Wrench.leg_forces``). The footpad contact force in ``TruthState.contact`` is then the soil
    force. A surface binding may ask for soil contact through the shared data
    (``footpad_contact``), so one line of the scenario file switches both.
    """

    def footpad_contact_mode(self) -> str:
        """Return ``physx`` or ``soil`` (from the parameters, else from the shared data, else physx)."""
        mode = str(self.parameters.get("footpad_contact", self.context.shared_data.get("footpad_contact", "physx")))
        if mode not in ("physx", "soil"):
            raise ValueError(f"footpad_contact must be 'physx' or 'soil', not '{mode}'")
        return mode

    def setup_scene(self, environment) -> None:
        """Write the lander USD from the preset, spawn it, and add the leg contact sensor."""
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        preset: LanderPreset = self.context.shared_data["lander_preset"]
        self.preset = preset
        self.reference_height_above_ground_contact = preset.center_of_mass_height_above_footpad_plane
        self.soil_contact = self.footpad_contact_mode() == "soil"
        usd_path = write_lander_usd(preset, Path(tempfile.gettempdir()) / "lunarsim_assets", footpad_collision=not self.soil_contact)
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
                pos=(0.0, 0.0, self.spawn_height()),
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
        # The body link carries the descent- and ascent-stage shapes: any force on it is a strike.
        self.body_contact_sensor = ContactSensor(
            ContactSensorCfg(prim_path="{ENV_REGEX_NS}/" + LANDER_PRIM_NAME + "/body", history_length=0)
        )
        environment.scene.sensors["body_contact"] = self.body_contact_sensor

    def initialize(self, environment) -> None:
        """Resolve body, joint and sensor indices in leg order, then allocate the mass state."""
        self.parameters.setdefault("minimum_mass_kg", 0.5 * self.preset.mass_at_start)
        self.root_body_index = self.asset.find_bodies("body")[0][0]
        leg_names = [leg_link_name(index) for index in range(self.preset.number_of_legs)]
        strut_names = [strut_joint_name(index) for index in range(self.preset.number_of_legs)]
        self.strut_joint_ids = [self.asset.find_joints(name)[0][0] for name in strut_names]
        self.contact_body_ids = [self.contact_sensor.find_bodies(name)[0][0] for name in leg_names]
        # Surface probes: the leg links that carry one, and the tip offset from the leg-link origin.
        self.probe_leg_body_ids = [
            self.asset.find_bodies(leg_link_name(leg - 1))[0][0] for leg in self.preset.legs_with_surface_probes
        ]
        self.probe_tip_offset = torch.tensor(self.preset.surface_probe_tip_offset_from_leg_link, device=self.device)
        self.leg_body_ids = [self.asset.find_bodies(name)[0][0] for name in leg_names]
        self.footpad_bottom_offset = torch.tensor([0.0, 0.0, -self.preset.footpad_depth], device=self.device)
        self.soil_contact_force = torch.zeros(self.number_of_environments, self.preset.number_of_legs, device=self.device)
        super().initialize(environment)

    def body_pose(self) -> tuple[Tensor, Tensor]:
        """Return the current world position (N, 3) and orientation (N, 4, w x y z) of the body (centre of mass)."""
        from lunarsim.core.rotation_math import quaternion_xyzw_to_wxyz

        data = self.asset.data
        return data.root_link_pos_w.torch, quaternion_xyzw_to_wxyz(data.root_link_quat_w.torch)

    def footpad_states(self) -> tuple[Tensor, Tensor]:
        """Return the world position of each dish's lowest point (N, L, 3) and the pad velocity (N, L, 3), now.

        The leg links do not rotate against the body (prismatic struts), so the body orientation
        turns the offset from the link origin (flat top of the dish) to the lowest point.
        """
        data = self.asset.data
        _, orientation = self.body_pose()
        links = data.body_link_pos_w.torch[:, self.leg_body_ids]
        count, legs = links.shape[0], links.shape[1]
        offset = rotate_vector_body_to_parent(orientation, self.footpad_bottom_offset.expand(count, 3))
        return links + offset.unsqueeze(1).expand(count, legs, 3), data.body_link_lin_vel_w.torch[:, self.leg_body_ids]

    def apply_wrench(self, bus, physics_step_duration: float) -> None:
        """Apply the body wrench, then (soil contact) add the soil forces on the leg links."""
        super().apply_wrench(bus, physics_step_duration)
        surface_wrench = bus.read(MessageKey.SURFACE_WRENCH, self)
        if surface_wrench.leg_forces is None:
            return
        if not self.soil_contact:
            raise RuntimeError("the surface sends footpad soil forces, but the plant has footpad_contact: physx")
        self.asset.permanent_wrench_composer.add_forces_and_torques_index(
            forces=surface_wrench.leg_forces,
            torques=torch.zeros_like(surface_wrench.leg_forces),
            positions=surface_wrench.leg_force_points,
            body_ids=self.leg_body_ids,
            is_global=True,
        )
        self.soil_contact_force = surface_wrench.leg_forces.norm(dim=-1)

    def _reset_joints(self, env_ids: Tensor) -> None:
        """Extend all struts fully and stop them."""
        zeros = torch.zeros(len(env_ids), len(self.strut_joint_ids), device=self.device)
        self.asset.write_joint_position_to_sim_index(position=zeros, joint_ids=self.strut_joint_ids, env_ids=env_ids)
        self.asset.write_joint_velocity_to_sim_index(velocity=zeros, joint_ids=self.strut_joint_ids, env_ids=env_ids)

    def _read_contact_forces(self) -> Tensor:
        """Return the contact force magnitude on each footpad, (N, n_legs), in leg order (the soil force with soil contact)."""
        if self.soil_contact:
            return self.soil_contact_force.clone()
        net_forces = self.contact_sensor.data.net_forces_w.torch  # (N, n_sensor_bodies, 3)
        return net_forces[:, self.contact_body_ids, :].norm(dim=-1)

    def _read_body_contact_force(self) -> Tensor:
        """Return the contact force magnitude on the body link (descent and ascent stages), (N,)."""
        return self.body_contact_sensor.data.net_forces_w.torch[:, 0, :].norm(dim=-1)

    def surface_probe_tip_positions(self, orientation: Tensor) -> Tensor:
        """Return the world position of every probe tip, (N, probes, 3), m.

        A probe hangs from its footpad along the body -z axis; the leg links do not rotate
        against the body (the struts are prismatic), so the body orientation turns the offset.
        """
        leg_positions = self.asset.data.body_link_pos_w.torch[:, self.probe_leg_body_ids]  # (N, probes, 3)
        count, probes = leg_positions.shape[0], leg_positions.shape[1]
        offset = rotate_vector_body_to_parent(orientation, self.probe_tip_offset.expand(count, 3))  # (N, 3)
        return leg_positions + offset.unsqueeze(1).expand(count, probes, 3)

    def _read_surface_probe_contact(self, geometry: SurfaceGeometry, orientation: Tensor) -> Tensor:
        """Return 1.0 where at least one probe tip is at or below the ground, else 0.0, (N,)."""
        if not self.probe_leg_body_ids:
            return torch.zeros(self.number_of_environments, device=self.device)
        tips = self.surface_probe_tip_positions(orientation)
        ground = geometry.ground_height_at(tips[..., 0], tips[..., 1])
        return (tips[..., 2] <= ground).any(dim=1).to(torch.float32)

    def strut_compression(self) -> Tensor:
        """Return how far each strut is compressed, (N, n_legs), m. Used by tests and plots."""
        return self.asset.data.joint_pos.torch[:, self.strut_joint_ids]
