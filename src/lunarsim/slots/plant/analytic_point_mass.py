# SPDX-License-Identifier: BSD-3-Clause
"""``plant: analytic_point_mass`` - the lander as a point mass whose thrust points where it is commanded.

The last and simplest step of the model ablation in the analytic-to-PhysX transfer study (test
T8, docs/knowledge/analytic-to-physx-transfer-study.md). Compared with ``analytic_rigid_body`` it
drops the rotation and the legs; everything else (start states, actuators, mass flow, gravity,
observation, reward, metrics) is unchanged.

Model (every physics substep, batched, semi-implicit Euler like the rigid-body plant):

- **Attitude at once** (user decision 2026-11-21, option A). The body takes the attitude that
  points its thrust axis along the policy's commanded thrust direction, at the heading it has,
  in the same substep (``quaternion_for_thrust_direction``). There is no rotation dynamics: the
  body rate is zero and the actuator torques are ignored. The training wrapper hands the
  direction over (``receive_commanded_thrust_direction``) because the Command carries jet
  duties, not the direction.
- **Translation.** The actuator and surface forces, turned into the world frame with that
  attitude, plus lunar gravity; the mass falls by the propellant flow.
- **Ground.** No legs: when the footpad plane reaches the flat ground the vehicle stops there
  (no bounce, no slide) and the legs report a contact force of the start weight, so the touchdown
  metric sees its first contact. The episode must end at that contact
  (``metrics.end_at_first_contact=true``; the environment refuses otherwise), and the approach
  values of the control step before judge the landing. The tilt at contact is the commanded
  tilt; the tilt rate is zero.

Only training (``scripts/train_ppo.py`` with ``ppo_mlp`` in ``attitude_commands`` mode) hands the
direction over; any other use stops with an error instead of flying a wrong attitude.
Parameters: ``minimum_mass_kg`` (as the rigid-body plant).
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.agents.classical.attitude_controller import quaternion_for_thrust_direction
from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey, Wrench
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.core.registry import register_binding
from lunarsim.core.rotation_math import rotate_vector_body_to_parent
from lunarsim.slots.plant.analytic_rigid_body import AnalyticRigidBodyPlant


@register_binding(SlotName.PLANT, "analytic_point_mass")
class AnalyticPointMassPlant(AnalyticRigidBodyPlant):
    """Point mass with the thrust along the commanded direction; stops on the flat ground."""

    # The environment and the training wrapper look for this flag (see AnalyticLanderEnvironment).
    needs_commanded_thrust_direction = True

    def __init__(self, parameters, context) -> None:
        """Set up the rigid-body plant's state and geometry, plus the commanded direction and the ground flag."""
        super().__init__(parameters, context)
        count = self.number_of_environments
        self.commanded_thrust_direction: Tensor | None = None
        self.on_ground = torch.zeros(count, dtype=torch.bool, device=self.device)
        # Reported per leg once the vehicle rests on the ground (above the metric's contact threshold).
        self.ground_contact_force_per_leg = self.preset.mass_at_start * MOON_SURFACE_GRAVITY / self.number_of_legs

    def receive_commanded_thrust_direction(self, direction: Tensor | None) -> None:
        """Store the commanded thrust direction (site frame, (N, 3)) for the coming control step."""
        if direction is None:
            raise ValueError("plant: analytic_point_mass needs an attitude command (ppo_mlp in attitude_commands mode)")
        self.commanded_thrust_direction = direction.to(device=self.device, dtype=torch.float32)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Start as the rigid-body plant does, then clear the body rate and the ground flag."""
        super().reset(env_ids, gen)
        env_ids = env_ids.to(torch.long)
        count = len(env_ids)
        self.angular_velocity_body = self.angular_velocity_body.index_put(
            (env_ids,), torch.zeros(count, 3, device=self.device)
        )
        self.on_ground = self.on_ground.index_put((env_ids,), torch.zeros(count, dtype=torch.bool, device=self.device))

    def apply_wrench(self, bus, physics_step_duration: float) -> None:
        """Advance one physics substep: attitude at once, translation under thrust and gravity, stop on the ground."""
        if self.commanded_thrust_direction is None:
            raise RuntimeError(
                "plant: analytic_point_mass got no commanded thrust direction; only PPO training "
                "(scripts/train_ppo.py, ppo_mlp in attitude_commands mode) hands it over"
            )
        actuator_wrench: Wrench = bus.read(MessageKey.ACTUATOR_WRENCH, self)
        surface_wrench: Wrench = bus.read(MessageKey.SURFACE_WRENCH, self)
        if surface_wrench.leg_forces is not None:
            raise RuntimeError("The point-mass plant has no legs; a soil surface is not supported")
        dt = physics_step_duration
        orientation = quaternion_for_thrust_direction(self.commanded_thrust_direction, self.orientation)
        force_world = rotate_vector_body_to_parent(orientation, actuator_wrench.force + surface_wrench.force)
        gravity = torch.tensor([0.0, 0.0, -MOON_SURFACE_GRAVITY], device=self.device)
        velocity = self.velocity + (force_world / self.mass.unsqueeze(1) + gravity) * dt
        position = self.position_world + velocity * dt

        # The footpad plane stops on the flat ground (world z = 0).
        resting_height = self.reference_height_above_ground_contact
        on_ground = self.on_ground | (position[:, 2] - resting_height <= 0.0)
        position = torch.cat(
            (
                position[:, :2],
                torch.where(on_ground, torch.full_like(position[:, 2], resting_height), position[:, 2]).unsqueeze(1),
            ),
            dim=1,
        )
        velocity = torch.where(on_ground.unsqueeze(1), torch.zeros_like(velocity), velocity)

        mass_flow_rate = actuator_wrench.mass_flow_rate + surface_wrench.mass_flow_rate
        self.mass = torch.clamp_min(self.mass - mass_flow_rate * dt, self.minimum_mass)
        self.velocity = velocity
        self.position_world = position
        self.orientation = orientation
        self.angular_velocity_body = torch.zeros_like(self.angular_velocity_body)
        self.on_ground = on_ground
        self.leg_contact_force = torch.where(
            on_ground.unsqueeze(1),
            torch.full_like(self.leg_contact_force, self.ground_contact_force_per_leg),
            torch.zeros_like(self.leg_contact_force),
        )
