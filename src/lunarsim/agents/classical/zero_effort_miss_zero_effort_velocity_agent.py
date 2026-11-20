# SPDX-License-Identifier: BSD-3-Clause
"""``agent: zem_zev`` - controller build-up step 4: full ZEM/ZEV landing guidance.

Two phases per episode:

1. **Guided phase.** ZEM/ZEV guidance flies to a gate point above the target: a few metres
   above the ground, with zero horizontal speed and the descent speed that the final descent
   profile asks for at that height. The lateral acceleration is made by tilting the lander,
   because the engine can only push along the body axis.
2. **Final descent.** The vertical controller of step 3 follows the descent-rate profile to
   the engine cut-off, while small tilts remove what is left of the horizontal speed. Close
   to the ground the lander is commanded upright.

The agent uses only the observation (the navigation estimate) and the lander's design data.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.agents.classical.attitude_controller import quaternion_for_thrust_direction
from lunarsim.agents.classical.landing_agents import ClassicalLandingAgentBase
from lunarsim.agents.classical.zero_effort_guidance import (
    choose_time_to_go,
    zero_effort_acceleration_now,
    zero_effort_errors,
)
from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import Command, Observation
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.core.registry import register_binding


@register_binding(SlotName.AGENT, "zem_zev")
class ZeroEffortMissZeroEffortVelocityAgent(ClassicalLandingAgentBase):
    """Classical baseline agent of scenario S1.

    Parameters (all optional):

    - ``target_position_m``: horizontal landing target in the site frame (default ``[0, 0]``).
    - ``gate_altitude_m``: footpad height of the gate where the guided phase ends (default 3).
    - ``guided_tilt_limit_deg``: largest tilt the guided phase may command (default 15).
    - ``planning_tilt_limit_deg``: tilt assumed when choosing the time-to-go (default 10).
    - ``planning_thrust_margin``: share of the maximum thrust assumed when planning (default 0.9).
    - ``time_to_go_stretch_factor`` and ``time_to_go_extra_s``: the plan lasts at least
      ``factor * shortest possible time + extra`` (defaults 1.25 and 2 s), which leaves room
      for the time the lander needs to tilt.
    - ``descend_below_horizontal_speed_mps`` and ``hold_above_horizontal_speed_mps``: in the
      final descent the wanted descent rate is scaled from full (horizontal speed below the
      first value, default 0.3) to zero (above the second, default 1.0), so the lander waits
      above the ground until its horizontal speed is small.
    - ``final_tilt_limit_deg``: largest tilt in the final descent (default 8).
    - ``upright_below_altitude_m``: below this footpad height the lander is held upright (default 2).
    - ``final_horizontal_velocity_gain_per_s``: gain that removes horizontal speed in the final descent (default 0.8).
    - ``minimum_time_to_go_s``: the guided phase hands over when less time than this is left (default 1).
    - ``longest_time_to_go_s``: the longest plan the chooser may pick (default 80 s; a gate far from the site needs more).
    - ``attitude`` and ``vertical``: parameters of the two inner controllers.
    """

    def __init__(self, parameters, number_of_environments, number_of_jets, device, shared_data=None) -> None:
        """Read the guidance parameters and allocate the per-episode plan."""
        super().__init__(parameters, number_of_environments, number_of_jets, device, shared_data)
        get = self.parameters.get
        target = get("target_position_m", [0.0, 0.0])
        self.gate_altitude = float(get("gate_altitude_m", 3.0))
        center_of_mass_height = self.preset.center_of_mass_height_above_footpad_plane
        self.gate_position = torch.tensor(
            [float(target[0]), float(target[1]), self.gate_altitude + center_of_mass_height], device=device
        )
        gate_speed, _ = self.vertical_controller.descent_rate_reference(torch.tensor([self.gate_altitude]))
        self.gate_velocity = torch.tensor([0.0, 0.0, float(gate_speed)], device=device)
        self.guided_tilt_tangent = math.tan(math.radians(float(get("guided_tilt_limit_deg", 15.0))))
        self.planning_tilt_tangent = math.tan(math.radians(float(get("planning_tilt_limit_deg", 10.0))))
        self.planning_thrust_margin = float(get("planning_thrust_margin", 0.9))
        self.final_tilt_tangent = math.tan(math.radians(float(get("final_tilt_limit_deg", 8.0))))
        self.upright_below_altitude = float(get("upright_below_altitude_m", 2.0))
        self.time_to_go_stretch_factor = float(get("time_to_go_stretch_factor", 1.25))
        self.time_to_go_extra = float(get("time_to_go_extra_s", 2.0))
        self.descend_below_horizontal_speed = float(get("descend_below_horizontal_speed_mps", 0.3))
        self.hold_above_horizontal_speed = float(get("hold_above_horizontal_speed_mps", 1.0))
        self.final_horizontal_velocity_gain = float(get("final_horizontal_velocity_gain_per_s", 0.8))
        self.minimum_time_to_go = float(get("minimum_time_to_go_s", 1.0))
        self.gravity = torch.tensor([0.0, 0.0, -MOON_SURFACE_GRAVITY], device=device)
        self.time_to_go_candidates = torch.arange(2.0, float(get("longest_time_to_go_s", 80.0)) + 0.5, 0.5, device=device)

        count = number_of_environments
        # Gate point of every environment (site frame). All start above the configured target; a
        # divert agent moves single environments with set_landing_targets.
        self.gate_positions = self.gate_position.repeat(count, 1)
        self.needs_plan = torch.ones(count, dtype=torch.bool, device=device)
        self.in_final_descent = torch.zeros(count, dtype=torch.bool, device=device)
        self.planned_duration = torch.zeros(count, device=device)
        self.elapsed = torch.zeros(count, device=device)
        self.plan_was_feasible = torch.ones(count, dtype=torch.bool, device=device)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Forget the plan of the given environments; a new one is made at their next step."""
        super().reset(env_ids, gen)
        self.gate_positions[env_ids] = self.gate_position
        self.needs_plan[env_ids] = True
        self.in_final_descent[env_ids] = False
        self.elapsed[env_ids] = 0.0

    def set_landing_targets(self, env_ids: Tensor, target_xy: Tensor) -> None:
        """Move the landing target of ``env_ids`` to site-frame (x, y) ``target_xy`` (K, 2) and plan again from the current state."""
        self.gate_positions[env_ids, 0:2] = target_xy.to(self.gate_positions.dtype)
        self.needs_plan[env_ids] = True

    def _plan(self, position: Tensor, velocity: Tensor, mass: Tensor) -> None:
        """Choose the time-to-go of every environment that starts a new episode."""
        preset = self.preset
        minimum_vertical = preset.engine_minimum_thrust / mass
        maximum_vertical = self.planning_thrust_margin * preset.engine_maximum_thrust / mass
        count = position.shape[0]
        duration, feasible = choose_time_to_go(
            position,
            velocity,
            self.gate_positions,
            self.gate_velocity.expand(count, 3),
            self.gravity,
            self.time_to_go_candidates,
            minimum_vertical,
            maximum_vertical,
            self.planning_tilt_tangent,
            minimum_height=self.gate_positions[:, 2],
            stretch_factor=self.time_to_go_stretch_factor,
            extra_time=self.time_to_go_extra,
        )
        new_plan = self.needs_plan
        self.planned_duration = torch.where(new_plan, duration, self.planned_duration)
        self.plan_was_feasible = torch.where(new_plan, feasible, self.plan_was_feasible)
        self.elapsed = torch.where(new_plan, torch.zeros_like(self.elapsed), self.elapsed)
        # A start at or below the gate goes straight to the final descent.
        below_gate = position[:, 2] <= self.gate_positions[:, 2] + 0.5
        self.in_final_descent = torch.where(new_plan, below_gate, self.in_final_descent)
        self.needs_plan[:] = False

    @staticmethod
    def _limit_tilt(thrust_acceleration: Tensor, tilt_tangent: Tensor | float) -> Tensor:
        """Shrink the horizontal part of a thrust acceleration so that the tilt stays below the limit."""
        vertical = thrust_acceleration[:, 2].clamp_min(1e-3)
        horizontal = thrust_acceleration[:, :2]
        horizontal_size = horizontal.norm(dim=1).clamp_min(1e-9)
        scale = (tilt_tangent * vertical / horizontal_size).clamp_max(1.0)
        return torch.cat((horizontal * scale.unsqueeze(1), vertical.unsqueeze(1)), dim=1)

    def compute_command(self, observation: Observation, control_step_duration: float) -> Command:
        """Return the command of the current phase for every environment."""
        position, velocity, orientation, angular_velocity, mass = self.split_state(observation)
        controller = self.vertical_controller
        altitude = self.height_above_ground(observation)
        count = position.shape[0]
        if bool(self.needs_plan.any()):
            self._plan(position, velocity, mass)

        # ---- guided phase: ZEM/ZEV to the gate ----
        time_to_go = (self.planned_duration - self.elapsed).clamp_min(self.minimum_time_to_go)
        miss, velocity_error = zero_effort_errors(
            position, velocity, self.gate_positions, self.gate_velocity.expand(count, 3), self.gravity, time_to_go
        )
        guided_thrust_acceleration = zero_effort_acceleration_now(miss, velocity_error, time_to_go)
        minimum_vertical = self.preset.engine_minimum_thrust / mass
        guided_thrust_acceleration[:, 2] = guided_thrust_acceleration[:, 2].clamp_min(minimum_vertical)
        guided_thrust_acceleration = self._limit_tilt(guided_thrust_acceleration, self.guided_tilt_tangent)

        # ---- final descent: descent-rate profile, horizontal speed removed by small tilts ----
        # The wanted descent rate is scaled down while the horizontal speed is still large, so
        # the lander waits above the ground instead of touching down while sliding sideways.
        horizontal_speed = velocity[:, :2].norm(dim=1)
        readiness = (
            (self.hold_above_horizontal_speed - horizontal_speed)
            / (self.hold_above_horizontal_speed - self.descend_below_horizontal_speed)
        ).clamp(0.0, 1.0)
        readiness = torch.where(altitude < self.upright_below_altitude, torch.ones_like(readiness), readiness)
        wanted_speed, feed_forward = controller.descent_rate_reference(altitude)
        final_vertical = (
            MOON_SURFACE_GRAVITY
            + feed_forward * readiness
            + controller.descent_rate_gain * (wanted_speed * readiness - velocity[:, 2])
        )
        final_horizontal = -self.final_horizontal_velocity_gain * velocity[:, :2]
        final_thrust_acceleration = torch.cat((final_horizontal, final_vertical.clamp_min(minimum_vertical).unsqueeze(1)), dim=1)
        tilt_tangent = torch.where(altitude < self.upright_below_altitude, 0.0, self.final_tilt_tangent)
        final_thrust_acceleration = self._limit_tilt(final_thrust_acceleration, tilt_tangent)

        # ---- phase switch ----
        self.elapsed += control_step_duration
        time_is_up = self.planned_duration - self.elapsed <= self.minimum_time_to_go
        at_gate = altitude <= self.gate_altitude + 0.25
        self.in_final_descent |= time_is_up | at_gate

        thrust_acceleration = torch.where(
            self.in_final_descent.unsqueeze(1), final_thrust_acceleration, guided_thrust_acceleration
        )
        wanted_orientation = quaternion_for_thrust_direction(thrust_acceleration, orientation)
        throttle = controller.throttle_for_vertical_acceleration(
            thrust_acceleration[:, 2] - MOON_SURFACE_GRAVITY, mass, orientation
        )
        return self.assemble_command(throttle, orientation, angular_velocity, wanted_orientation, observation, control_step_duration)
