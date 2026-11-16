# SPDX-License-Identifier: BSD-3-Clause
"""``metrics: phase_gate`` - end an orbit phase at its gate and record the state there (milestone 6, S7).

Design S7: "A phase is a list of slot overrides and a gate condition [...] When the gate is met, it
saves the state". Each environment's episode ends (terminated) when its gate condition holds, and
its record holds the full float64 state at the gate (``gate_`` columns, in the plant's frame) for
the next phase's ``start: from_log``, plus the numbers S7 measures per phase (elapsed time,
propellant used, altitude, speeds). An environment that does not reach the gate within
``maximum_elapsed_s`` ends truncated with ``gate_reached`` False.

``until`` (one or more conditions; the first one met ends the phase): ``altitude_below_m`` (footpad height above the ground, from the plant),
``range_below_m`` (ground distance to the site along the sphere; Apollo's high gate is "a range of
approximately 4.5 nautical miles", [D-6846] p. 10; needs ``site``),
``perilune: true`` (the radial velocity turns from negative to non-negative after at least
``minimum_elapsed_s``; PDI is "near the perilune of the descent transfer orbit", [D-6846] p. 8),
``elapsed_at_least_s``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import EpisodeEvents, MessageKey, TruthState
from lunarsim.core.registry import register_binding
from lunarsim.slots.metrics.touchdown import OUTCOME_CODE

FRAME_CODES = ("MCI", "MCMF", "SITE")


@register_binding(SlotName.METRICS, "phase_gate")
class PhaseGateMetrics(BindingBase):
    """Gate detection and gate-state records of one mission phase."""

    requires = frozenset({MessageKey.TRUTH_STATE, MessageKey.START_STATE, MessageKey.NAVIGATION_ESTIMATE})
    provides = frozenset({MessageKey.EPISODE_EVENTS})
    reads_truth = True

    def __init__(self, parameters, context) -> None:
        """Read the gate condition and allocate the per-episode records."""
        super().__init__(parameters, context)
        until = dict(self.parameters.get("until") or {})
        conditions = set(until) - {"minimum_elapsed_s"}
        if not conditions or not conditions <= {"altitude_below_m", "range_below_m", "perilune", "elapsed_at_least_s"}:
            raise ValueError(f"metrics.until needs one or more of altitude_below_m, range_below_m, perilune, elapsed_at_least_s, not {sorted(until)}")
        self.until = until
        self.maximum_elapsed = float(self.parameters.get("maximum_elapsed_s", 1.0e9))
        from lunarsim.slots.plant.orbit_float64 import site_from_parameters

        self.site = site_from_parameters(self.parameters.get("site"))
        count, device = self.number_of_environments, self.device
        self.episode_number = torch.zeros(count, dtype=torch.long, device=device)
        self.start_time = torch.zeros(count, dtype=torch.float64, device=device)
        self.start_mass = torch.zeros(count, dtype=torch.float64, device=device)
        self.previous_radial_speed = torch.full((count,), -1.0, dtype=torch.float64, device=device)
        self.minimum_altitude = torch.full((count,), float("inf"), dtype=torch.float64, device=device)
        self._records: list[dict[str, Tensor]] = []
        self.started = torch.zeros(count, dtype=torch.bool, device=device)
        # Navigation error (estimate minus truth) over the phase and at the gate (S5: TRN error).
        self.navigation_error_squared_sum = torch.zeros(count, dtype=torch.float64, device=device)
        self.navigation_error_steps = torch.zeros(count, dtype=torch.float64, device=device)
        self.health_width = 0

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Start new episode records; the start time and mass come from the start state."""
        start: TruthState = self.context.bus.read(MessageKey.START_STATE, self)
        self.start_time[env_ids] = start.time[env_ids].to(torch.float64)
        self.start_mass[env_ids] = start.mass[env_ids].to(torch.float64)
        self.previous_radial_speed[env_ids] = -1.0
        self.minimum_altitude[env_ids] = float("inf")
        self.started[env_ids] = True
        self.navigation_error_squared_sum[env_ids] = 0.0
        self.navigation_error_steps[env_ids] = 0.0

    def ground_range(self, truth: TruthState) -> Tensor:
        """Return the ground distance (N,) from the lander's radial line to the site, along the site sphere, m."""
        position = truth.position.to(torch.float64)
        if truth.frame.value == "SITE":
            center = torch.tensor([0.0, 0.0, -self.site.radius_m], dtype=torch.float64, device=position.device)
            site_direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=position.device)
            relative = position - center
        elif truth.frame.value == "MCMF":
            site_direction = self.site.east_north_up(torch.float64, position.device)[:, 2]
            relative = position
        else:
            raise ValueError("range_below_m needs a Moon-fixed or site frame")
        cosine = (torch.nn.functional.normalize(relative, dim=1) * site_direction).sum(dim=1).clamp(-1.0, 1.0)
        return self.site.radius_m * torch.acos(cosine)

    def gate_reached(self, truth: TruthState, elapsed: Tensor, radial_speed: Tensor) -> Tensor:
        """Return True where this step meets any of the gate conditions (the first one met ends the phase)."""
        reached = torch.zeros_like(elapsed, dtype=torch.bool)
        if "range_below_m" in self.until:
            reached |= self.ground_range(truth) <= float(self.until["range_below_m"])
        if "altitude_below_m" in self.until:
            reached |= truth.footpad_height_above_ground.to(torch.float64) <= float(self.until["altitude_below_m"])
        if "perilune" in self.until:
            minimum = float(self.until.get("minimum_elapsed_s", 0.0))
            reached |= (self.previous_radial_speed < 0.0) & (radial_speed >= 0.0) & (elapsed >= minimum)
        if "elapsed_at_least_s" in self.until:
            reached |= elapsed >= float(self.until["elapsed_at_least_s"])
        return reached

    def step(self, bus, dt: float) -> None:
        """Check the gate; end the episodes that reached it (or timed out) and store their records."""
        truth: TruthState = bus.read(MessageKey.TRUTH_STATE, self)
        position, velocity = truth.position.to(torch.float64), truth.velocity.to(torch.float64)
        elapsed = truth.time.to(torch.float64) - self.start_time
        center = position.clone()
        if truth.frame.value == "SITE":
            center[:, 2] += 1.7374e6  # radial direction near the site (only its sign change matters)
        radial_speed = (center * velocity).sum(dim=1) / center.norm(dim=1)
        altitude = truth.footpad_height_above_ground.to(torch.float64)
        self.minimum_altitude = torch.minimum(self.minimum_altitude, altitude)
        navigation_error = torch.full_like(altitude, float("nan"))
        health = None
        if bus.has(MessageKey.NAVIGATION_ESTIMATE):
            estimate = bus.read(MessageKey.NAVIGATION_ESTIMATE, self)
            navigation_error = (estimate.mean[:, 0:3].to(torch.float64) - position).norm(dim=1)
            self.navigation_error_squared_sum += navigation_error**2
            self.navigation_error_steps += 1.0
            health = estimate.health
        reached = self.gate_reached(truth, elapsed, radial_speed) & self.started
        timed_out = (elapsed >= self.maximum_elapsed) & ~reached & self.started
        crashed = (altitude < 0.0) & ~reached & self.started
        terminated = reached | crashed
        truncated = timed_out
        finished = terminated | truncated
        self.previous_radial_speed = radial_speed
        outcome = torch.full_like(self.episode_number, -1)
        if bool(finished.any()):
            code = torch.where(reached, OUTCOME_CODE["success"], OUTCOME_CODE["timeout_no_contact"])
            code = torch.where(crashed, OUTCOME_CODE["out_of_bounds"], code)
            outcome = torch.where(finished, code, outcome)
            horizontal = velocity - radial_speed.unsqueeze(1) * torch.nn.functional.normalize(center, dim=1)
            columns = {
                "environment_index": torch.arange(self.number_of_environments, device=self.device),
                "episode_number": self.episode_number.clone(),
                "gate_reached": reached.clone(),
                "outcome_code": code,
                "success": reached.clone(),
                "gate_frame_code": torch.full_like(self.episode_number, FRAME_CODES.index(truth.frame.value)),
                "gate_time_s": truth.time.to(torch.float64).clone(),
                "elapsed_s": elapsed.clone(),
                "gate_mass_kg": truth.mass.to(torch.float64).clone(),
                "propellant_used_kg": self.start_mass - truth.mass.to(torch.float64),
                "gate_altitude_m": altitude.clone(),
                "minimum_altitude_m": self.minimum_altitude.clone(),
                "gate_radial_speed_mps": radial_speed.clone(),
                "gate_horizontal_speed_mps": horizontal.norm(dim=1),
                "gate_navigation_error_m": navigation_error.clone(),
                "navigation_error_rms_m": (self.navigation_error_squared_sum / self.navigation_error_steps.clamp_min(1.0)).sqrt(),
            }
            if health is not None:
                for column in range(health.shape[1]):
                    columns[f"gate_health_{column}"] = health[:, column].to(torch.float64).clone()
            names = ("x", "y", "z")
            for index, name in enumerate(names):
                columns[f"gate_position_{name}_m"] = position[:, index].clone()
                columns[f"gate_velocity_{name}_mps"] = velocity[:, index].clone()
                columns[f"gate_body_rate_{name}_rad_s"] = truth.angular_velocity[:, index].to(torch.float64).clone()
            for index, name in enumerate(("w", "x", "y", "z")):
                columns[f"gate_orientation_{name}"] = truth.orientation_quaternion[:, index].to(torch.float64).clone()
            self._records.append({name: values[finished].detach().cpu() for name, values in columns.items()})
            self.episode_number = self.episode_number + finished.long()
            self.started = self.started & ~finished
        bus.write(MessageKey.EPISODE_EVENTS, EpisodeEvents(terminated=terminated, truncated=truncated, outcome_code=outcome), self)

    def drain_episode_records(self) -> dict[str, Tensor]:
        """Return and forget the finished records (with a ``gate_frame`` name column added by the runner)."""
        if not self._records:
            return {}
        merged = {name: torch.cat([record[name] for record in self._records]) for name in self._records[0]}
        self._records = []
        return merged
