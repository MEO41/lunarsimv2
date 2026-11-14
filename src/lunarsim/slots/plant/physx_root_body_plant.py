# SPDX-License-Identifier: BSD-3-Clause
"""Shared logic of plant bindings that simulate one free body (or articulation) in PhysX.

The plant has three jobs:

1. ``step``: read the body state from PhysX and write it to the bus as ``TruthState`` in the
   SITE frame (position relative to the environment origin, quaternion as w, x, y, z).
2. ``apply_wrench``: every PhysX substep, add up the actuator and surface wrenches and hand
   them to PhysX; integrate the vehicle mass from the propellant flow.
3. ``reset``: put the body at the state that the start slot sampled.

This is the only place where Isaac Lab conventions (x, y, z, w quaternions, world-frame
positions) are converted to LunarSim conventions.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.binding import BindingBase
from lunarsim.core.message_types import Frame, MessageKey, TruthState, Wrench
from lunarsim.core.rotation_math import (
    quaternion_wxyz_to_xyzw,
    quaternion_xyzw_to_wxyz,
    rotate_vector_body_to_parent,
    rotate_vector_parent_to_body,
)


class PhysxRootBodyPlant(BindingBase):
    """Base class. Subclasses create the asset in ``setup_scene`` and store it in ``self.asset``."""

    requires = frozenset({MessageKey.START_STATE, MessageKey.ACTUATOR_WRENCH, MessageKey.SURFACE_WRENCH})
    provides = frozenset({MessageKey.TRUTH_STATE})

    asset = None  # isaaclab RigidObject or Articulation, set by the subclass in setup_scene
    root_body_index: int = 0

    def initialize(self, environment) -> None:
        """Allocate the mass state once the simulation is running."""
        self.environment_origins = environment.scene.env_origins
        self.default_body_masses = self.asset.data.body_mass.torch.clone()  # (N, n_bodies)
        self.default_root_inertia = self.asset.data.body_inertia.torch[:, self.root_body_index].clone()  # (N, 9)
        self.default_total_mass = self.default_body_masses.sum(dim=1)
        self.non_root_mass = self.default_total_mass - self.default_body_masses[:, self.root_body_index]
        self.total_mass = self.default_total_mass.clone()
        self.minimum_total_mass = self.default_total_mass * 0.0 + float(self.parameters.get("minimum_mass_kg", 0.0))
        self.mass_changed_since_last_write = torch.zeros(
            self.number_of_environments, dtype=torch.bool, device=self.device
        )
        self._root_body_ids = [self.root_body_index]
        self._force_application_offset = torch.zeros(self.number_of_environments, 1, 3, device=self.device)
        self._all_environment_ids = torch.arange(self.number_of_environments, device=self.device)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Move the given environments to the state sampled by the start slot."""
        start: TruthState = self.context.bus.read(MessageKey.START_STATE, self)
        if start.frame is not Frame.SITE:
            raise ValueError("The PhysX plant needs a start state in the SITE frame")
        orientation = start.orientation_quaternion[env_ids].to(torch.float32)
        root_pose = torch.cat(
            (
                start.position[env_ids].to(torch.float32) + self.environment_origins[env_ids],
                quaternion_wxyz_to_xyzw(orientation),
            ),
            dim=1,
        )
        angular_velocity_world = rotate_vector_body_to_parent(
            orientation, start.angular_velocity[env_ids].to(torch.float32)
        )
        root_velocity = torch.cat((start.velocity[env_ids].to(torch.float32), angular_velocity_world), dim=1)
        self.asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.asset.write_root_link_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        self._reset_joints(env_ids)

        start_mass = start.mass[env_ids].to(torch.float32)
        self.total_mass[env_ids] = torch.where(torch.isnan(start_mass), self.default_total_mass[env_ids], start_mass)
        self.mass_changed_since_last_write[env_ids] = True
        self._write_mass_to_simulation()

    def _reset_joints(self, env_ids: Tensor) -> None:
        """Reset joint states. A single rigid body has no joints, so the base does nothing."""

    def apply_wrench(self, bus, physics_step_duration: float) -> None:
        """Hand the summed wrench to PhysX and integrate the mass. Called every PhysX substep."""
        actuator_wrench: Wrench = bus.read(MessageKey.ACTUATOR_WRENCH, self)
        surface_wrench: Wrench = bus.read(MessageKey.SURFACE_WRENCH, self)
        force = actuator_wrench.force + surface_wrench.force
        torque = actuator_wrench.torque + surface_wrench.torque
        # Body-frame force acting at the origin of the body frame (the vehicle centre of mass),
        # plus a body-frame torque about that point.
        self.asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force.unsqueeze(1),
            torques=torque.unsqueeze(1),
            positions=self._force_application_offset,
            body_ids=self._root_body_ids,
            is_global=False,
        )
        mass_flow_rate = actuator_wrench.mass_flow_rate + surface_wrench.mass_flow_rate
        if bool((mass_flow_rate != 0.0).any()):
            self.total_mass = torch.maximum(
                self.total_mass - mass_flow_rate * physics_step_duration, self.minimum_total_mass
            )
            self.mass_changed_since_last_write |= mass_flow_rate != 0.0

    def _write_mass_to_simulation(self) -> None:
        """Push the current vehicle mass (and the scaled inertia) of changed environments to PhysX.

        All mass change is booked on the root body. Its inertia is scaled by the same factor as
        its mass, which keeps the radii of gyration constant while propellant burns.
        """
        if not bool(self.mass_changed_since_last_write.any()):
            return
        env_ids = self._all_environment_ids[self.mass_changed_since_last_write]
        root_mass = (self.total_mass - self.non_root_mass)[env_ids]
        scale = root_mass / self.default_body_masses[env_ids, self.root_body_index]
        self.asset.set_masses_index(masses=root_mass.unsqueeze(1), body_ids=self._root_body_ids, env_ids=env_ids)
        self.asset.set_inertias_index(
            inertias=(self.default_root_inertia[env_ids] * scale.unsqueeze(1)).unsqueeze(1),
            body_ids=self._root_body_ids,
            env_ids=env_ids,
        )
        self.mass_changed_since_last_write[:] = False

    def _read_contact_forces(self) -> Tensor:
        """Return leg contact force magnitudes (N, n_legs). A body without legs returns (N, 0)."""
        return torch.zeros(self.number_of_environments, 0, device=self.device)

    def step(self, bus, dt: float) -> None:
        """Read the state from PhysX and write TruthState in the SITE frame."""
        self._write_mass_to_simulation()
        data = self.asset.data
        orientation = quaternion_xyzw_to_wxyz(data.root_link_quat_w.torch)
        state = TruthState(
            frame=Frame.SITE,
            time=self.context.clock.time,
            position=data.root_link_pos_w.torch - self.environment_origins,
            velocity=data.root_link_lin_vel_w.torch.clone(),
            orientation_quaternion=orientation,
            angular_velocity=rotate_vector_parent_to_body(orientation, data.root_link_ang_vel_w.torch),
            mass=self.total_mass.clone(),
            contact=self._read_contact_forces(),
        )
        bus.write(MessageKey.TRUTH_STATE, state, self)
