# SPDX-License-Identifier: BSD-3-Clause
"""Reward of the landing task (milestone 1b), with every weight from the scenario file.

The reward is training-side: it reads ``TruthState`` (like the metrics slot) and the judgement
of the metrics slot (``EpisodeEvents``). It never reaches the actor's observation. Terms, added
every control step:

1. **Outcome** (end of an episode): ``success`` for a success, ``failure`` for a failed
   touchdown, and ``failure_without_touchdown`` when the episode ended before any contact (flew
   out of bounds, timed out, tipped over in the air). The last must be worse than the worst
   crash; otherwise a first trial learned to fly away instead of trying to land.
2. **Touchdown quality** (the control step of the first contact): minus
   ``touchdown_weight * sum(min((value / limit)^2, touchdown_term_cap))`` over vertical speed,
   horizontal speed, tilt and tilt rate, with the landing-gear limits of the lander preset. A
   touchdown exactly at all four limits costs ``4 * touchdown_weight``; it rewards a softer
   touchdown also inside the limits. The cap keeps a crash from swamping the other terms.
3. **Velocity tracking** before the first contact, per second of flight: minus
   ``tracking_weight_per_s * |v - v_wanted|``. ``v_wanted`` points to the target: horizontal
   ``-position / horizontal_time_constant_s`` (at most ``maximum_wanted_horizontal_speed_mps``),
   vertical ``-height / vertical_time_constant_s`` kept between ``wanted_touchdown_speed_mps``
   and ``maximum_wanted_descent_speed_mps``. This is the dense velocity-field term of Gaudet,
   Linares and Furfaro (2020, "Deep reinforcement learning for six degree-of-freedom planetary
   landing", Advances in Space Research 65). It costs time when hovering or flying away.
4. **Attitude** before the first contact, per second: minus
   ``tilt_weight_per_s * tilt + rate_weight_per_s * |body rate|`` (radians).
5. **Propellant**: ``propellant_weight_per_kg`` times the mass burned in the step (negative). It
   also stops the policy from keeping the engine running after the contact.

Every one of these numbers is a **model choice** for training; none is a physical constant.
Earlier trials (2026-11-15, 300 iterations at 1,024 environments each) with potential-based
shaping learned to hover or to fly away; see the decisions log.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lunarsim.assets.lander_preset import LanderPreset
from lunarsim.core.message_types import EpisodeEvents, TruthState
from lunarsim.core.rotation_math import tilt_angle_from_vertical

DEFAULT_REWARD_WEIGHTS = {
    "success": 100.0,
    "failure": -100.0,
    "failure_without_touchdown": -300.0,
    "touchdown_weight": 10.0,
    "touchdown_term_cap": 4.0,
    "tracking_weight_per_s": 1.0,
    "tilt_weight_per_s": 1.0,
    "rate_weight_per_s": 1.0,
    "horizontal_time_constant_s": 10.0,
    "maximum_wanted_horizontal_speed_mps": 2.0,
    "vertical_time_constant_s": 4.0,
    "wanted_touchdown_speed_mps": 0.5,
    "maximum_wanted_descent_speed_mps": 6.0,
    "propellant_weight_per_kg": -0.05,
}
SUCCESS_OUTCOME_CODE = 0


class LandingRewardTerms:
    """Computes the per-step reward of every environment from truth and the metrics' judgement."""

    term_names = ("outcome", "touchdown", "tracking", "attitude", "propellant")

    def __init__(
        self, parameters: dict[str, Any], preset: LanderPreset, number_of_environments: int, device, control_step_duration: float
    ) -> None:
        """Read the weights (unknown names are an error) and allocate the per-environment state."""
        unknown = set(parameters) - set(DEFAULT_REWARD_WEIGHTS)
        if unknown:
            raise ValueError(f"Unknown reward parameters {sorted(unknown)}; known: {sorted(DEFAULT_REWARD_WEIGHTS)}")
        self.weights = {**DEFAULT_REWARD_WEIGHTS, **{name: float(value) for name, value in parameters.items()}}
        self.limits = {
            "vertical_speed": preset.maximum_vertical_touchdown_speed,
            "horizontal_speed": preset.maximum_horizontal_touchdown_speed,
            "tilt": preset.maximum_touchdown_tilt,
            "tilt_rate": preset.maximum_touchdown_tilt_rate,
        }
        self.control_step_duration = float(control_step_duration)
        self.previous_mass = torch.full((number_of_environments,), float("nan"), device=device)
        self.has_touched = torch.zeros(number_of_environments, dtype=torch.bool, device=device)
        # Sum of each term over the current episode, and a copy taken before resets (for the log).
        self.episode_sums = {name: torch.zeros(number_of_environments, device=device) for name in self.term_names}
        self.last_episode_sums = {name: values.clone() for name, values in self.episode_sums.items()}

    def reset(self, env_ids: Tensor) -> None:
        """Forget the state of the given environments."""
        self.previous_mass[env_ids] = float("nan")
        self.has_touched[env_ids] = False
        for values in self.episode_sums.values():
            values[env_ids] = 0.0

    def wanted_velocity(self, truth: TruthState) -> Tensor:
        """Return the wanted velocity of every environment, (N, 3), m/s."""
        w = self.weights
        horizontal = -truth.position[:, :2] / w["horizontal_time_constant_s"]
        norm = horizontal.norm(dim=1, keepdim=True).clamp_min(1e-6)
        horizontal = horizontal * (w["maximum_wanted_horizontal_speed_mps"] / norm).clamp(max=1.0)
        height = truth.footpad_height_above_ground.clamp_min(0.0)
        vertical = -(height / w["vertical_time_constant_s"]).clamp(
            w["wanted_touchdown_speed_mps"], w["maximum_wanted_descent_speed_mps"]
        )
        return torch.cat((horizontal, vertical.unsqueeze(1)), dim=1)

    def compute(self, truth: TruthState, events: EpisodeEvents) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the reward of this control step, (N,), and each term, for logging."""
        w = self.weights
        dt = self.control_step_duration
        finished = events.terminated | events.truncated
        zeros = torch.zeros_like(self.previous_mass)
        first_contact = events.first_contact if events.first_contact is not None else torch.zeros_like(finished)

        outcome = zeros
        if events.outcome_code is not None:
            touched = self.has_touched | first_contact
            failure = torch.where(touched, w["failure"], w["failure_without_touchdown"])
            outcome = torch.where(
                finished, torch.where(events.outcome_code == SUCCESS_OUTCOME_CODE, w["success"], failure), zeros
            )

        touchdown = zeros
        if events.touchdown_values is not None and bool(first_contact.any()):
            squared_excess = sum(
                ((events.touchdown_values[name].abs() / limit) ** 2).clamp(max=w["touchdown_term_cap"])
                for name, limit in self.limits.items()
            )
            touchdown = torch.where(first_contact, -w["touchdown_weight"] * squared_excess, zeros)

        in_flight = ~(self.has_touched | first_contact)
        velocity_error = (truth.velocity - self.wanted_velocity(truth)).norm(dim=1)
        tracking = torch.where(in_flight, -w["tracking_weight_per_s"] * velocity_error * dt, zeros)
        tilt = tilt_angle_from_vertical(truth.orientation_quaternion)
        rate = truth.angular_velocity.norm(dim=1)
        attitude = torch.where(in_flight, -(w["tilt_weight_per_s"] * tilt + w["rate_weight_per_s"] * rate) * dt, zeros)
        self.has_touched |= first_contact

        burned = torch.nan_to_num(self.previous_mass - truth.mass, nan=0.0).clamp_min(0.0)
        propellant = w["propellant_weight_per_kg"] * burned
        self.previous_mass = truth.mass.clone()

        terms = {"outcome": outcome, "touchdown": touchdown, "tracking": tracking, "attitude": attitude, "propellant": propellant}
        for name, values in terms.items():
            self.episode_sums[name] += values
        self.last_episode_sums = {name: values.clone() for name, values in self.episode_sums.items()}
        reward = outcome + touchdown + tracking + attitude + propellant
        if not bool(torch.isfinite(reward).all()):
            raise FloatingPointError("The landing reward is not finite")
        return reward, terms
