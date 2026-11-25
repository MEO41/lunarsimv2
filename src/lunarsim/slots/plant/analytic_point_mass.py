# SPDX-License-Identifier: BSD-3-Clause
"""``plant: analytic_point_mass`` - the lander as a point mass whose thrust points where it is commanded.

The last and simplest step of the model ablation in the analytic-to-PhysX transfer study (test
T8, docs/knowledge/analytic-to-physx-transfer-study.md). Compared with ``analytic_rigid_body`` it
drops the rotation and the legs; everything else (start states, actuators, mass flow, gravity,
observation, reward, metrics) is unchanged.

Model (every physics substep, batched, semi-implicit Euler like the rigid-body plant):

- **Attitude at once** (``attitude_time_constant_s: 0``, the default; user decision 2026-11-21,
  option A). The body takes the attitude that points its thrust axis along the policy's
  commanded thrust direction, at the heading it has, in the same substep
  (``quaternion_for_thrust_direction``). There is no rotation dynamics: the body rate is zero.
- **Attitude with a first-order lag** (``attitude_time_constant_s`` > 0, option B, user approval
  2026-11-23). Every substep the thrust axis turns towards the commanded direction by the
  fraction ``1 - exp(-dt / tau)`` of the angle between them, about the shortest axis (no heading
  change). The body rate is that turn divided by the substep, so the policy sees a rate and the
  touchdown metric a tilt rate. The value used in T8 (1.3 s) is measured on the rigid-body lander
  with the classical attitude controller: ``scripts/measure_attitude_time_constant.py``.

In both modes the actuator torques are ignored. The training wrapper hands the direction over
(``receive_commanded_thrust_direction``) because the Command carries jet duties, not the
direction.
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

**Reported body rate** (``body_rate_report``; observation-channel control asked for by the
simulated review of the paper, item R20, 2026-11-25). The body rate is what the policy observes
and what the touchdown metric judges as the tilt rate, so it can change the transfer by itself:

- ``physical`` (default, the T8 runs): zero at once, the turn over the substep with a lag.
- ``zero``: always zero, also with a lag (a lag without a rate).
- ``held_turn_rate``: the attitude change of the latest turn divided by
  ``held_turn_rate_period_s`` (default 0.1 s, one policy decision: 5 control steps of 4 substeps
  of 5 ms), held until the attitude turns again (an attitude at once with a rate).

Parameters: ``attitude_time_constant_s`` (default 0 = at once), ``body_rate_report`` (default
``physical``), ``held_turn_rate_period_s`` (default 0.1), ``minimum_mass_kg`` (as the rigid-body
plant).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.agents.classical.attitude_controller import quaternion_for_thrust_direction, shortest_rotation_between
from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey, Wrench
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.core.registry import register_binding
from lunarsim.core.rotation_math import (
    body_up_axis_in_parent,
    quaternion_from_axis_angle,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_vector,
    rotate_vector_body_to_parent,
    rotate_vector_parent_to_body,
)
from lunarsim.slots.plant.analytic_rigid_body import AnalyticRigidBodyPlant


@register_binding(SlotName.PLANT, "analytic_point_mass")
class AnalyticPointMassPlant(AnalyticRigidBodyPlant):
    """Point mass with the thrust along the commanded direction; stops on the flat ground."""

    # The environment and the training wrapper look for this flag (see AnalyticLanderEnvironment).
    needs_commanded_thrust_direction = True

    def __init__(self, parameters, context) -> None:
        """Set up the rigid-body plant's state and geometry, plus the attitude lag, the commanded direction and the ground flag."""
        super().__init__(parameters, context)
        count = self.number_of_environments
        self.attitude_time_constant = float(self.parameters.get("attitude_time_constant_s", 0.0))
        if self.attitude_time_constant < 0.0:
            raise ValueError(f"attitude_time_constant_s must be zero or positive, not {self.attitude_time_constant}")
        self.body_rate_report = str(self.parameters.get("body_rate_report", "physical"))
        if self.body_rate_report not in ("physical", "zero", "held_turn_rate"):
            raise ValueError(f"body_rate_report must be physical, zero or held_turn_rate, not {self.body_rate_report}")
        self.held_turn_rate_period = float(self.parameters.get("held_turn_rate_period_s", 0.1))
        if self.held_turn_rate_period <= 0.0:
            raise ValueError(f"held_turn_rate_period_s must be positive, not {self.held_turn_rate_period}")
        # Parent-frame rate of the latest turn, used by body_rate_report=held_turn_rate.
        self.held_turn_rate_parent = torch.zeros(count, 3, device=self.device)
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
        self.held_turn_rate_parent = self.held_turn_rate_parent.index_put(
            (env_ids,), torch.zeros(count, 3, device=self.device)
        )

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
        orientation, angular_velocity_body = self._turn_towards_commanded_direction(dt)
        angular_velocity_body = self._reported_body_rate(orientation, angular_velocity_body)
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
        self.angular_velocity_body = torch.where(
            on_ground.unsqueeze(1), torch.zeros_like(angular_velocity_body), angular_velocity_body
        )
        self.on_ground = on_ground
        self.leg_contact_force = torch.where(
            on_ground.unsqueeze(1),
            torch.full_like(self.leg_contact_force, self.ground_contact_force_per_leg),
            torch.zeros_like(self.leg_contact_force),
        )

    def _turn_towards_commanded_direction(self, dt: float) -> tuple[Tensor, Tensor]:
        """Return the attitude after this substep and the body rate of the turn, (N, 4) and (N, 3) rad/s.

        At once (time constant 0): the attitude along the commanded direction, zero rate. With a
        lag: the shortest turn from the current thrust axis to the commanded direction, scaled by
        ``1 - exp(-dt / tau)``; the rate is that turn over ``dt``, in the body frame.
        """
        if self.attitude_time_constant == 0.0:
            orientation = quaternion_for_thrust_direction(self.commanded_thrust_direction, self.orientation)
            return orientation, torch.zeros_like(self.angular_velocity_body)
        full_turn = shortest_rotation_between(body_up_axis_in_parent(self.orientation), self.commanded_thrust_direction)
        fraction = 1.0 - math.exp(-dt / self.attitude_time_constant)
        turn_vector = quaternion_to_rotation_vector(full_turn) * fraction  # parent frame, rad
        angle = turn_vector.norm(dim=1)
        partial_turn = quaternion_from_axis_angle(turn_vector, angle)
        orientation = quaternion_multiply(partial_turn, self.orientation)
        orientation = orientation / orientation.norm(dim=1, keepdim=True).clamp_min(1e-9)
        angular_velocity_body = rotate_vector_parent_to_body(self.orientation, turn_vector / dt)
        return orientation, angular_velocity_body

    def _reported_body_rate(self, orientation: Tensor, physical_body_rate: Tensor) -> Tensor:
        """Return the body rate the plant reports, (N, 3) rad/s, as chosen by ``body_rate_report``.

        ``physical`` passes the turn rate through, ``zero`` reports no rate, and
        ``held_turn_rate`` reports the latest attitude change (from the attitude before this
        substep to ``orientation``) divided by ``held_turn_rate_period_s``, kept until the attitude
        turns again, in the body frame of ``orientation``.
        """
        if self.body_rate_report == "physical":
            return physical_body_rate
        if self.body_rate_report == "zero":
            return torch.zeros_like(physical_body_rate)
        change = quaternion_multiply(orientation, quaternion_conjugate(self.orientation))
        turn_vector = quaternion_to_rotation_vector(change)  # parent frame, rad
        # Smaller changes are float32 noise of re-pointing the same command (about 1e-7 rad).
        turned = turn_vector.norm(dim=1) > 1e-5
        self.held_turn_rate_parent = torch.where(
            turned.unsqueeze(1), turn_vector / self.held_turn_rate_period, self.held_turn_rate_parent
        )
        return rotate_vector_parent_to_body(orientation, self.held_turn_rate_parent)
