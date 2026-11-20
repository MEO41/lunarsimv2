# SPDX-License-Identifier: BSD-3-Clause
"""Classical guidance of the orbit phases of S7 (milestone 6): ``scripted_burns`` and ``quadratic_guidance``.

**scripted_burns** - the descent orbit insertion (DOI): "a short retrograde maneuver of
approximately 75 fps" that lowers the perilune to 50,000 ft ([D-6846] Bennett, NASA TN D-6846,
p. 3). The agent points the thrust axis retrograde, burns at ``throttle`` from ``start_s`` until
its speed has dropped by the delta-v that the vis-viva equation gives for the wanted perilune
(apolune = the current radius), then holds attitude. For a 60 nmi circular orbit and a 50,000 ft
perilune that is 21.8 m/s (71.5 ft/s).

**quadratic_guidance** - Apollo's descent guidance law ([D-6846] p. 6, Fig. 5: "The basic LM
descent guidance logic is defined by an acceleration command which is a quadratic function of
time [...] quadratic guidance"), here with the time-to-go counted down from the phase's planned
duration: with time-to-go ``T``, target position ``r_D``, velocity ``v_D`` and acceleration ``a_D``,

    a_command = a_D - 6 (v + v_D) / T - 12 (r - r_D) / T^2,

the acceleration of the constant-jerk path that meets all three targets at ``T = 0``. Thrust
acceleration = command - effective gravity (point mass, plus Coriolis and centrifugal terms in the
Moon-fixed frame). Throttle as on the LM ([D-6846] p. 4 and p. 7): commands inside the throttle
region (10 to 60 % of design thrust) are flown; above it the engine runs at the fixed throttle
position (FTP). Targets are the Apollo 11 premission gates: high gate (braking phase, P63) and low
gate (approach phase, P64), [D-6846] Table I and p. 10. The phase ends at the gate (``metrics:
phase_gate``), before ``T`` reaches zero.

Both read only the observation (the navigation estimate, float64 in the orbit frames).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.agents.classical.attitude_controller import quaternion_for_thrust_direction
from lunarsim.agents.classical.landing_agents import ClassicalLandingAgentBase
from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import Command, Observation
from lunarsim.core.physical_constants import MOON_GRAVITATIONAL_PARAMETER, MOON_ROTATION_RATE
from lunarsim.core.registry import register_binding
from lunarsim.slots.plant.orbit_float64 import site_from_parameters


class OrbitGuidanceBase(ClassicalLandingAgentBase):
    """Shared parts: float64 state, effective gravity, thrust-vector to command with the LM throttle logic."""

    def __init__(self, parameters, number_of_environments, number_of_jets, device, shared_data=None) -> None:
        """Read the frame (MCI or MCMF) and the throttle logic."""
        super().__init__(parameters, number_of_environments, number_of_jets, device, shared_data)
        get = self.parameters.get
        self.frame = str(get("frame", "MCMF"))
        if self.frame not in ("MCI", "MCMF", "SITE"):
            raise ValueError("guidance frame must be MCI, MCMF or SITE")
        # In the SITE frame (local_f64, S5) the Moon's centre lies one site radius below the origin.
        self.moon_center = torch.zeros(3, dtype=torch.float64, device=device)
        if self.frame == "SITE":
            self.moon_center[2] = -site_from_parameters(get("site")).radius_m
        throttle = dict(get("throttle") or {})
        self.lower_throttle = float(throttle.get("lower_fraction", 0.10))  # [D-6846] p. 4: throttleable between 10 and 60 %
        self.upper_throttle = float(throttle.get("upper_fraction", 0.60))
        self.fixed_throttle_point = float(throttle.get("fixed_throttle_point", 0.925))  # [D-6846] p. 4: 92.5 to 95.5 % of design thrust
        # [D-6846] p. 8: at the fixed throttle point "when the command decreases to 57 percent, a
        # 3-percent low bias, the DPS is throttled as commanded" (throttle recovery).
        self.recovery_fraction = float(throttle.get("recovery_fraction", 0.57))
        # Braking starts at the fixed throttle point; the approach was flown throttled ([D-6846] p. 4).
        self.start_at_fixed_point = bool(throttle.get("start_at_fixed_point", True))
        self.at_fixed_point = torch.full((number_of_environments,), self.start_at_fixed_point, dtype=torch.bool, device=device)
        self.rated_thrust = float(self.preset.engine_rated_thrust)  # the throttle is a fraction of rated thrust
        # The engine ignites once the thrust axis is within this angle of the wanted direction; after
        # that it keeps burning while the attitude follows (the DPS made one burn from PDI to
        # touchdown, [D-6846] p. 7).
        self.alignment_limit = math.radians(float(get("ignite_within_deg", 10.0)))  # model choice
        self.elapsed = torch.zeros(number_of_environments, dtype=torch.float64, device=device)
        self.ignited = torch.zeros(number_of_environments, dtype=torch.bool, device=device)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Restart the phase clock of ``env_ids``; their engine waits for alignment again."""
        super().reset(env_ids, gen)
        self.elapsed[env_ids] = 0.0
        self.ignited[env_ids] = False
        self.at_fixed_point[env_ids] = self.start_at_fixed_point

    def effective_gravity(self, position: Tensor, velocity: Tensor) -> Tensor:
        """Return gravity plus the frame terms (N, 3), float64."""
        relative = position - self.moon_center
        gravity = -MOON_GRAVITATIONAL_PARAMETER * relative / relative.norm(dim=1, keepdim=True) ** 3
        if self.frame == "MCMF":
            spin = torch.tensor([0.0, 0.0, MOON_ROTATION_RATE], dtype=torch.float64, device=position.device).expand_as(position)
            gravity = gravity - 2.0 * torch.cross(spin, velocity, dim=1) - torch.cross(spin, torch.cross(spin, position, dim=1), dim=1)
        return gravity

    def apollo_throttle(self, fraction: Tensor) -> Tensor:
        """Map a wanted thrust fraction to the LM's: 0 when off, the 10-60 % region, or the fixed throttle point.

        The fixed throttle point is latched: a lander stays there until the command falls below
        ``recovery_fraction`` (57 %), and goes back only when the command reaches the fixed point
        itself. Without the latch, commands between 60 and 92.5 % flip the thrust between the
        two each step (and with it the direction the guidance asks for).
        """
        self.at_fixed_point = torch.where(
            self.at_fixed_point, fraction >= self.recovery_fraction, fraction >= self.fixed_throttle_point
        )
        flown = fraction.clamp(self.lower_throttle, self.upper_throttle)
        flown = torch.where(self.at_fixed_point, torch.full_like(fraction, self.fixed_throttle_point), flown)
        return torch.where(fraction > 1e-6, flown, torch.zeros_like(fraction))

    def command_for(self, thrust_acceleration: Tensor, throttle: Tensor, observation: Observation, control_step_duration: float) -> Command:
        """Point the thrust axis along ``thrust_acceleration`` and fly ``throttle`` once the axis is aligned."""
        _, _, orientation, angular_velocity, _ = self.split_state(observation)
        orientation32 = orientation.to(torch.float32)
        wanted = quaternion_for_thrust_direction(thrust_acceleration.to(torch.float32), orientation32)
        body_axis = torch.stack(
            (
                2.0 * (orientation32[:, 1] * orientation32[:, 3] + orientation32[:, 0] * orientation32[:, 2]),
                2.0 * (orientation32[:, 2] * orientation32[:, 3] - orientation32[:, 0] * orientation32[:, 1]),
                1.0 - 2.0 * (orientation32[:, 1] ** 2 + orientation32[:, 2] ** 2),
            ),
            dim=1,
        )
        direction = torch.nn.functional.normalize(thrust_acceleration.to(torch.float32), dim=1)
        aligned = (body_axis * direction).sum(dim=1) >= math.cos(self.alignment_limit)
        may_burn = aligned | self.ignited
        throttle = torch.where(may_burn, throttle.to(torch.float32), torch.zeros_like(throttle, dtype=torch.float32))
        self.ignited |= throttle > 0.0
        jets = self.attitude_controller.compute_jet_duties(orientation32, angular_velocity.to(torch.float32), wanted, control_step_duration)
        return Command(throttle=throttle, gimbal=torch.zeros(throttle.shape[0], 2, device=throttle.device), reaction_control_jets=jets)


@register_binding(SlotName.AGENT, "scripted_burns")
class ScriptedBurnsAgent(OrbitGuidanceBase):
    """Retrograde burn that lowers the perilune (DOI), then attitude hold (see the module text).

    Parameters: ``frame`` (MCI), ``site`` (for the perilune radius), ``start_s``,
    ``perilune_altitude_m`` (default 15,240 m = 50,000 ft, [D-6846] p. 3), ``throttle_fraction``
    (default 0.4, model choice).
    """

    def __init__(self, parameters, number_of_environments, number_of_jets, device, shared_data=None) -> None:
        """Read the burn settings."""
        parameters = {"frame": "MCI", **dict(parameters)}
        super().__init__(parameters, number_of_environments, number_of_jets, device, shared_data)
        get = self.parameters.get
        self.site = site_from_parameters(get("site"))
        self.start_time = float(get("start_s", 10.0))
        self.perilune_radius = self.site.radius_m + float(get("perilune_altitude_m", 15240.0))
        self.throttle_fraction = float(get("throttle_fraction", 0.4))
        count = number_of_environments
        self.burn_speed_target = torch.full((count,), float("nan"), dtype=torch.float64, device=device)
        self.burn_done = torch.zeros(count, dtype=torch.bool, device=device)
        self.hold_direction = torch.zeros(count, 3, dtype=torch.float64, device=device)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Arm the burn of ``env_ids`` again."""
        super().reset(env_ids, gen)
        self.burn_speed_target[env_ids] = float("nan")
        self.burn_done[env_ids] = False
        self.hold_direction[env_ids] = 0.0

    def compute_command(self, observation: Observation, control_step_duration: float) -> Command:
        """Point retrograde; burn from ``start_s`` until the speed has dropped by the wanted delta-v."""
        position, velocity, _, _, _ = self.split_state(observation)
        position, velocity = position.to(torch.float64), velocity.to(torch.float64)
        radius, speed = position.norm(dim=1), velocity.norm(dim=1)
        starting = (self.elapsed >= self.start_time) & torch.isnan(self.burn_speed_target) & ~self.burn_done
        needed = torch.sqrt(MOON_GRAVITATIONAL_PARAMETER * (2.0 / radius - 2.0 / (radius + self.perilune_radius)))
        self.burn_speed_target = torch.where(starting, needed, self.burn_speed_target)
        burning = ~self.burn_done & torch.isfinite(self.burn_speed_target)
        finished = burning & (speed <= self.burn_speed_target)
        self.burn_done |= finished
        burning = burning & ~finished
        retrograde = -velocity / speed.unsqueeze(1).clamp_min(1e-9)
        self.hold_direction = torch.where(self.burn_done.unsqueeze(1) & (self.hold_direction.norm(dim=1, keepdim=True) == 0), retrograde, self.hold_direction)
        direction = torch.where(self.burn_done.unsqueeze(1), self.hold_direction, retrograde)
        throttle = torch.where(burning, torch.full_like(speed, self.throttle_fraction), torch.zeros_like(speed))
        self.elapsed += control_step_duration
        return self.command_for(direction, throttle, observation, control_step_duration)


@register_binding(SlotName.AGENT, "quadratic_guidance")
class QuadraticGuidanceAgent(OrbitGuidanceBase):
    """Apollo-style quadratic guidance to a gate fixed to the Moon (see the module text).

    Parameters: ``frame`` (MCMF, or SITE for ``local_f64``), ``site``, ``approach_azimuth_deg`` (ground-track heading at the
    site, from north, clockwise), ``target`` (``uprange_m``, ``crossrange_m``, ``altitude_m``,
    ``forward_speed_mps``, ``vertical_speed_mps``), ``time_to_go_s`` (planned duration of the
    phase), ``minimum_time_to_go_s``, ``throttle`` (``lower_fraction``, ``upper_fraction``,
    ``fixed_throttle_point``).
    """

    def __init__(self, parameters, number_of_environments, number_of_jets, device, shared_data=None) -> None:
        """Build the target state in the Moon-fixed frame."""
        super().__init__(parameters, number_of_environments, number_of_jets, device, shared_data)
        get = self.parameters.get
        site = site_from_parameters(get("site"))
        target = dict(get("target") or {})
        basis = site.east_north_up(torch.float64, device)
        azimuth = math.radians(float(get("approach_azimuth_deg", 270.0)))
        forward = math.sin(azimuth) * basis[:, 0] + math.cos(azimuth) * basis[:, 1]  # flight direction at the site
        left = torch.cross(basis[:, 2], forward, dim=0)
        up = basis[:, 2]
        # The gate lies uprange on the sphere: turn the site's radius back by uprange / radius about the left axis.
        angle = float(target.get("uprange_m", 0.0)) / site.radius_m
        cross_angle = float(target.get("crossrange_m", 0.0)) / site.radius_m
        direction = up * math.cos(angle) - forward * math.sin(angle)
        direction = torch.nn.functional.normalize(direction + left * math.sin(cross_angle), dim=0)
        local_forward = torch.nn.functional.normalize(forward - (forward @ direction) * direction, dim=0)
        # Curved coordinates of the gate (Moon-fixed): its radial direction, the flight direction
        # there and the normal of the descent plane (left of the flight direction).
        self.gate_up = direction.clone()
        self.gate_forward = local_forward.clone()
        self.plane_normal = torch.cross(self.gate_up, self.gate_forward, dim=0)
        self.target_forward_speed = float(target.get("forward_speed_mps", 0.0))
        self.target_vertical_speed = float(target.get("vertical_speed_mps", 0.0))
        self.target_gate_altitude = float(target.get("altitude_m", 0.0))
        self.target_position = (site.radius_m + float(target.get("altitude_m", 0.0))) * direction
        self.target_velocity = float(target.get("forward_speed_mps", 0.0)) * local_forward + float(target.get("vertical_speed_mps", 0.0)) * direction
        if self.frame == "SITE":
            self.target_position = (self.target_position - site.origin_mcmf(torch.float64, device)) @ basis
            self.target_velocity = self.target_velocity @ basis
            self.gate_up, self.gate_forward, self.plane_normal = self.gate_up @ basis, self.gate_forward @ basis, self.plane_normal @ basis
        self.target_acceleration = torch.zeros(3, dtype=torch.float64, device=device)
        self.time_to_go_start = float(get("time_to_go_s", 100.0))
        self.minimum_time_to_go = float(get("minimum_time_to_go_s", 5.0))
        # ``fixed``: the time-to-go counts down from ``time_to_go_s`` (approach, flown throttled).
        # ``planned``: at the phase start each lander plans a constant downrange deceleration to the
        # gate (duration and target acceleration, see ``plan_durations``) and counts down from it.
        # Braking so flies near the fixed-throttle-point thrust, as on Apollo ([D-6846] p. 8:
        # maximum thrust "for most of the phase").
        self.time_to_go_mode = str(get("time_to_go_mode", "fixed"))
        # ``feasible`` (approach): the ZEM/ZEV time-to-go chooser of the terminal phase picks the
        # duration in the curved coordinates treated as flat (8 km), from the engine's vertical
        # range, a tilt limit and no dip below the gate; a_D = 0 (the quadratic law is then ZEM/ZEV).
        if self.time_to_go_mode not in ("fixed", "planned", "feasible"):
            raise ValueError("time_to_go_mode must be 'fixed', 'planned' or 'feasible'")
        self.planning_tilt_tangent = math.tan(math.radians(float(get("planning_tilt_limit_deg", 55.0))))
        self.planning_thrust_margin = float(get("planning_thrust_margin", 0.9))
        self.time_to_go_stretch_factor = float(get("time_to_go_stretch_factor", 1.1))
        self.time_to_go_extra = float(get("time_to_go_extra_s", 2.0))
        # Lowest height the planned path may reach before the gate (default: the gate's own altitude).
        self.planning_minimum_height = float(get("planning_minimum_height_m", float((get("target") or {}).get("altitude_m", 0.0))))
        self.candidate_durations = torch.arange(
            float(get("shortest_duration_s", 300.0)), float(get("longest_duration_s", 1200.0)) + 1e-9, float(get("duration_step_s", 5.0)),
            dtype=torch.float64, device=device,
        )
        # ``planned`` with ``initial_deceleration_fraction``: the Apollo-style front-loaded profile
        # (see ``front_loaded_durations``); without it, a constant deceleration.
        fraction = get("initial_deceleration_fraction")
        self.initial_deceleration_fraction = None if fraction is None else float(fraction)
        self.target_altitude = float(target.get("altitude_m", 0.0))
        self.site_radius = site.radius_m
        self.exhaust_velocity = float(self.preset.engine_exhaust_velocity)
        self.planned_duration = torch.full((number_of_environments,), self.time_to_go_start, dtype=torch.float64, device=device)
        self.planned_target_acceleration = torch.zeros(number_of_environments, 3, dtype=torch.float64, device=device)
        self.needs_plan = torch.ones(number_of_environments, dtype=torch.bool, device=device)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Plan again at the next step of ``env_ids``."""
        super().reset(env_ids, gen)
        self.needs_plan[env_ids] = True

    def plan_durations(self, position: Tensor, velocity: Tensor, mass: Tensor) -> Tensor:
        """Plan each lander's braking: duration and target downrange acceleration of a constant deceleration.

        With constant downrange acceleration ``a`` from rate ``v`` to the gate rate ``v_D`` over the
        distance ``d`` to the gate: ``T = 2 d / (v + v_D)`` and ``a = (v_D - v) / T``. The law's
        target acceleration ``a_D`` is set to ``a`` in the downrange axis, so the path keeps a
        steady deceleration near the fixed-throttle-point thrust instead of the falling profile
        of ``a_D = 0`` (which cannot brake 1.5 km/s in 480 km on the LM engine). Returns T (N,).
        """
        coordinates, rates, _ = self.curved_state(position, velocity)
        target_coordinates, target_rates = self.curved_target()
        distance = (target_coordinates[0] - coordinates[:, 0]).clamp_min(1.0)
        speed_change = target_rates[0] - rates[:, 0]
        if self.initial_deceleration_fraction is None:
            duration = 2.0 * distance / (rates[:, 0] + target_rates[0]).clamp_min(1.0)
        else:
            duration = self.front_loaded_durations(distance, rates[:, 0], speed_change, mass)
        duration = duration.clamp(float(self.candidate_durations[0]), float(self.candidate_durations[-1]))
        if self.initial_deceleration_fraction is None:
            gate_acceleration = speed_change / duration
        else:
            # Linear profile from -A0 to a_D: its mean over T is the mean deceleration.
            gate_acceleration = 2.0 * speed_change / duration + self.initial_deceleration(mass)
        self.planned_target_acceleration[:, 0] = torch.where(self.needs_plan, gate_acceleration, self.planned_target_acceleration[:, 0])
        return duration

    def initial_deceleration(self, mass: Tensor) -> Tensor:
        """Return the planned downrange deceleration at the phase start, A0 (N,): a fraction of the fixed-throttle-point acceleration."""
        return self.initial_deceleration_fraction * self.fixed_throttle_point * self.rated_thrust / mass

    def front_loaded_durations(self, distance: Tensor, rate: Tensor, speed_change: Tensor, mass: Tensor) -> Tensor:
        """Return T (N,) of a downrange deceleration falling linearly in time from A0 (Apollo-style braking).

        [D-6846] p. 8: the braking guidance "is targeted such that the commanded quadratic
        acceleration, and consequently the command thrust, is a decreasing function", flown at the
        fixed throttle point. Here the downrange acceleration is linear in time from ``-A0`` to
        ``a_D`` (constant jerk, which the quadratic law flies exactly). The speed change gives
        ``a_D = 2 dv / T + A0``, and the distance gives ``d = (v + dv / 3) T - A0 T^2 / 6``, whose
        smaller root is T. If even A0 cannot brake within ``d`` the constant-deceleration time is used.
        """
        start = self.initial_deceleration(mass)
        linear = rate + speed_change / 3.0
        discriminant = linear**2 - (2.0 / 3.0) * start * distance
        root = (linear - discriminant.clamp_min(0.0).sqrt()) / (start / 3.0)
        steady = 2.0 * distance / (rate + rate + speed_change).clamp_min(1.0)
        return torch.where(discriminant >= 0.0, root, steady)

    def feasible_durations(self, position: Tensor, velocity: Tensor, mass: Tensor) -> Tensor:
        """Return each lander's duration (N,) from the ZEM/ZEV chooser in the curved coordinates (``feasible`` mode)."""
        from lunarsim.agents.classical.zero_effort_guidance import choose_time_to_go
        from lunarsim.core.physical_constants import MOON_GRAVITATIONAL_PARAMETER

        coordinates, rates, _ = self.curved_state(position, velocity)
        target_coordinates, target_rates = self.curved_target()
        count = coordinates.shape[0]
        radius = (position - self.moon_center).norm(dim=1)
        effective = MOON_GRAVITATIONAL_PARAMETER / radius**2 - (rates[:, 0] ** 2 + rates[:, 1] ** 2) / radius
        gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64, device=position.device) * effective.mean()
        candidates = torch.arange(float(self.candidate_durations[0]), float(self.candidate_durations[-1]) + 0.5, 1.0, dtype=torch.float64, device=position.device)
        # Never shorter than a steady deceleration would take (T = 2 d / (v + v_D)): a shorter plan
        # must speed up first and brake harder later.
        distance = (target_coordinates[0] - coordinates[:, 0]).clamp_min(0.0)
        steady = 2.0 * distance / (rates[:, 0] + target_rates[0]).clamp_min(1.0)
        duration, _ = choose_time_to_go(
            coordinates, rates, target_coordinates.expand(count, 3), target_rates.expand(count, 3), gravity, candidates,
            self.lower_throttle * self.rated_thrust / mass,
            self.planning_thrust_margin * self.fixed_throttle_point * self.rated_thrust / mass,
            self.planning_tilt_tangent,
            minimum_height=torch.full_like(mass, self.planning_minimum_height),
            stretch_factor=self.time_to_go_stretch_factor,
            extra_time=self.time_to_go_extra,
        )
        return torch.maximum(duration.to(torch.float64), steady.clamp(max=float(candidates[-1])))

    def curved_state(self, position: Tensor, velocity: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (coordinates (N, 3), rates (N, 3), local basis (N, 3, 3)) of the curved frame.

        Coordinates: downrange arc to the gate (negative before it), crossrange arc (left
        positive), both at the site radius, and altitude above the site sphere. The basis columns
        are the local downrange, crossrange and up directions at the lander.
        """
        relative = position - self.moon_center
        radius = relative.norm(dim=1, keepdim=True)
        up = relative / radius
        in_plane = up - (up @ self.plane_normal).unsqueeze(1) * self.plane_normal
        downrange_angle = torch.atan2(in_plane @ self.gate_forward, in_plane @ self.gate_up)
        crossrange_angle = torch.asin((up @ self.plane_normal).clamp(-1.0, 1.0))
        downrange_direction = torch.nn.functional.normalize(torch.cross(self.plane_normal.expand_as(up), up, dim=1), dim=1)
        crossrange_direction = torch.cross(up, downrange_direction, dim=1)
        coordinates = torch.stack(
            (self.site_radius * downrange_angle, self.site_radius * crossrange_angle, radius.squeeze(1) - self.site_radius), dim=1
        )
        scale = self.site_radius / radius.squeeze(1)
        rates = torch.stack(
            ((velocity * downrange_direction).sum(dim=1) * scale, (velocity * crossrange_direction).sum(dim=1) * scale, (velocity * up).sum(dim=1)),
            dim=1,
        )
        return coordinates, rates, torch.stack((downrange_direction, crossrange_direction, up), dim=2)

    def curved_target(self) -> tuple[Tensor, Tensor]:
        """Return the gate's curved coordinates and rates (each (3,))."""
        coordinates = torch.tensor([0.0, 0.0, self.target_gate_altitude], dtype=torch.float64, device=self.target_position.device)
        rates = torch.tensor([self.target_forward_speed, 0.0, self.target_vertical_speed], dtype=torch.float64, device=self.target_position.device)
        return coordinates, rates

    def curved_thrust(self, time_to_go: Tensor, position: Tensor, velocity: Tensor, overtime: Tensor | None = None) -> Tensor:
        """Return the thrust acceleration (N, 3) of the quadratic law flown in the curved coordinates.

        ``overtime`` (N,): seconds past the planned arrival. The target point then moves on at its
        target rates, so the lander flies through the gate. (With the time-to-go held at its
        minimum and a fixed target, the law has a steady hover v_D * T / 2 short of the gate.)
        """
        coordinates, rates, basis = self.curved_state(position, velocity)
        target_coordinates, target_rates = self.curved_target()
        if overtime is not None:
            target_coordinates = target_coordinates + target_rates * overtime.unsqueeze(1)
        command = self.planned_target_acceleration - 6.0 * (rates + target_rates) / time_to_go - 12.0 * (coordinates - target_coordinates) / time_to_go**2
        radius = (position - self.moon_center).norm(dim=1)
        horizontal_squared = rates[:, 0] ** 2 + rates[:, 1] ** 2
        # Back to a Cartesian acceleration: curvature terms of motion over a sphere.
        wanted = torch.stack(
            (
                command[:, 0] + 2.0 * rates[:, 2] * rates[:, 0] / radius,
                command[:, 1] + 2.0 * rates[:, 2] * rates[:, 1] / radius,
                command[:, 2] - horizontal_squared / radius,
            ),
            dim=1,
        )
        cartesian = (basis @ wanted.unsqueeze(2)).squeeze(2)
        return cartesian - self.effective_gravity(position, velocity)

    def thrust_for(self, time_to_go: Tensor, position: Tensor, velocity: Tensor, gravity: Tensor) -> Tensor:
        """Return the thrust acceleration (N, 3) of the quadratic law for time-to-go (N, 1)."""
        command = (
            self.target_acceleration
            - 6.0 * (velocity + self.target_velocity) / time_to_go
            - 12.0 * (position - self.target_position) / time_to_go**2
        )
        return command - gravity

    def compute_command(self, observation: Observation, control_step_duration: float) -> Command:
        """Return the quadratic-guidance thrust command of this step."""
        position, velocity, _, _, mass = self.split_state(observation)
        position, velocity, mass = position.to(torch.float64), velocity.to(torch.float64), mass.to(torch.float64)
        gravity = self.effective_gravity(position, velocity)
        if self.time_to_go_mode in ("planned", "feasible") and bool(self.needs_plan.any()):
            planner = self.plan_durations if self.time_to_go_mode == "planned" else self.feasible_durations
            self.planned_duration = torch.where(self.needs_plan, planner(position, velocity, mass), self.planned_duration)
            self.needs_plan[:] = False
        start = self.planned_duration if self.time_to_go_mode in ("planned", "feasible") else torch.full_like(self.elapsed, self.time_to_go_start)
        time_to_go = (start - self.elapsed).clamp_min(self.minimum_time_to_go).unsqueeze(1)
        overtime = (self.elapsed - (start - self.minimum_time_to_go)).clamp_min(0.0)
        thrust_acceleration = self.curved_thrust(time_to_go, position, velocity, overtime)
        throttle = self.apollo_throttle(mass * thrust_acceleration.norm(dim=1) / self.rated_thrust)
        delivered = throttle * self.rated_thrust / mass
        thrust_acceleration = self.share_with_vertical_priority(thrust_acceleration, position, delivered)
        was_ignited = self.ignited.clone()
        command = self.command_for(thrust_acceleration, throttle, observation, control_step_duration)
        # The plan's clock starts at ignition: a lander that ignites this step plans again from
        # its state at the next step (the attitude turn before ignition is not part of the plan).
        just_ignited = self.ignited & ~was_ignited
        if self.time_to_go_mode in ("planned", "feasible"):
            self.needs_plan |= just_ignited
        self.elapsed = torch.where(just_ignited, torch.zeros_like(self.elapsed), self.elapsed + control_step_duration * self.ignited)
        return command

    def share_with_vertical_priority(self, thrust_acceleration: Tensor, position: Tensor, delivered: Tensor) -> Tensor:
        """Point a thrust of size ``delivered`` so its vertical part is the commanded one; the rest goes horizontal.

        The LM engine gives either the commanded 10-60 % or the fixed throttle point, so the thrust
        is often larger or smaller than the command. The vertical part (holding altitude) is kept
        as commanded (clipped to what is delivered); the horizontal part, along the commanded
        horizontal direction, takes the rest: extra braking at the fixed throttle point, less when
        saturated. The guidance then sees the lander ahead or behind and corrects (model choice).
        """
        up = torch.nn.functional.normalize(position - self.moon_center, dim=1)
        vertical = (thrust_acceleration * up).sum(dim=1)
        horizontal = thrust_acceleration - vertical.unsqueeze(1) * up
        vertical = torch.maximum(torch.minimum(vertical, delivered), -delivered)
        room = (delivered**2 - vertical**2).clamp_min(0.0).sqrt()
        direction = torch.nn.functional.normalize(horizontal, dim=1)
        has_horizontal = horizontal.norm(dim=1) > 1e-9
        shared = vertical.unsqueeze(1) * up + room.unsqueeze(1) * direction
        return torch.where(has_horizontal.unsqueeze(1), shared, thrust_acceleration)
