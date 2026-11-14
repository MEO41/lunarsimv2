# SPDX-License-Identifier: BSD-3-Clause
"""``metrics: touchdown`` - detects touchdown, judges it, ends the episode, keeps one record per episode."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.assets.lander_preset import LanderPreset
from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import EpisodeEvents, MessageKey, TruthState
from lunarsim.core.registry import register_binding
from lunarsim.core.rotation_math import tilt_angle_from_vertical

# Outcome of an episode. The first matching cause, in this order, is the outcome.
OUTCOME_NAMES = (
    "success",
    "out_of_bounds",  # flew away from the site before touching down
    "timeout_no_contact",  # the time limit passed before any footpad touched
    "propellant_depleted",  # the engine ran dry before contact and the touchdown broke a limit
    "hard_vertical",  # vertical speed at contact above the limit
    "hard_horizontal",  # horizontal speed at contact above the limit
    "tilt_at_contact",  # tilt at contact above the limit
    "tilt_rate_at_contact",  # body rate at contact above the limit
    "tip_over",  # inside all limits at contact, but tipped over afterwards
    "not_settled",  # inside all limits at contact, but still moving when the settle time ran out
)
OUTCOME_CODE = {name: code for code, name in enumerate(OUTCOME_NAMES)}


@register_binding(SlotName.METRICS, "touchdown")
class TouchdownMetrics(BindingBase):
    """Judges each landing against the lander's touchdown limits.

    Touchdown values (speeds, tilt, tilt rate) are taken from the last control step *before*
    the first footpad contact, so they describe the approach and not the impact.

    An episode ends (terminated) when the lander has settled, tipped over, failed to settle in
    time, or left the site; it is truncated at the time limit.

    Parameters (all optional): ``contact_force_threshold_n``, ``settle_speed_mps``,
    ``settle_rate_deg_per_s``, ``settle_hold_s``, ``settle_timeout_s``, ``tip_over_tilt_deg``,
    ``maximum_horizontal_distance_m``, ``maximum_altitude_m``.
    """

    requires = frozenset({MessageKey.TRUTH_STATE, MessageKey.START_STATE})
    provides = frozenset({MessageKey.EPISODE_EVENTS})
    reads_truth = True

    def __init__(self, parameters, context) -> None:
        """Read the limits from the lander preset and allocate the per-episode state."""
        super().__init__(parameters, context)
        preset: LanderPreset = context.shared_data["lander_preset"]
        self.preset = preset
        get = self.parameters.get
        self.contact_force_threshold = float(get("contact_force_threshold_n", 100.0))
        self.settle_speed = float(get("settle_speed_mps", 0.02))
        self.settle_rate = math.radians(float(get("settle_rate_deg_per_s", 0.5)))
        self.settle_hold = float(get("settle_hold_s", 1.0))
        self.settle_timeout = float(get("settle_timeout_s", 10.0))
        self.tip_over_tilt = math.radians(float(get("tip_over_tilt_deg", 45.0)))
        self.maximum_horizontal_distance = float(get("maximum_horizontal_distance_m", 300.0))
        self.maximum_altitude = float(get("maximum_altitude_m", 200.0))
        self.episode_length = float(context.shared_data["episode_length_seconds"])

        count, device = self.number_of_environments, self.device

        def zeros(*shape, dtype=torch.float32) -> Tensor:
            """Allocate a zero tensor with the environment count first."""
            return torch.zeros(count, *shape, dtype=dtype, device=device)

        self.has_contact = zeros(dtype=torch.bool)
        self.contact_time = zeros()
        self.settled_duration = zeros()
        self.peak_leg_force = zeros()
        self.episode_number = zeros(dtype=torch.long)
        # State one control step ago (the approach state when contact is first seen).
        self.previous = {name: zeros() for name in ("vertical_speed", "horizontal_speed", "tilt", "tilt_rate")}
        self.previous_position = zeros(3)
        self.touchdown = {name: zeros() for name in ("vertical_speed", "horizontal_speed", "tilt", "tilt_rate")}
        self.touchdown_position = zeros(3)
        self.start = {
            name: zeros()
            for name in ("altitude", "vertical_speed", "horizontal_velocity_x", "horizontal_velocity_y", "tilt", "mass")
        }
        self._finished_records: list[dict[str, Tensor]] = []

    # ---- limits ----

    def horizontal_speed_limit(self, vertical_speed: Tensor) -> Tensor:
        """Return the allowed horizontal touchdown speed for a given vertical touchdown speed.

        Full limit up to the corner speed, then a linear drop to zero at the maximum vertical speed.
        """
        preset = self.preset
        corner = preset.vertical_speed_where_horizontal_limit_starts_to_drop
        fraction = (preset.maximum_vertical_touchdown_speed - vertical_speed.abs()) / (
            preset.maximum_vertical_touchdown_speed - corner
        )
        return preset.maximum_horizontal_touchdown_speed * fraction.clamp(0.0, 1.0)

    # ---- binding interface ----

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Start a new episode record for the given environments."""
        start: TruthState = self.context.bus.read(MessageKey.START_STATE, self)
        tilt = tilt_angle_from_vertical(start.orientation_quaternion[env_ids].to(torch.float32))
        velocity = start.velocity[env_ids].to(torch.float32)
        position = start.position[env_ids].to(torch.float32)
        self.start["altitude"][env_ids] = position[:, 2] - self.preset.center_of_mass_height_above_footpad_plane
        self.start["vertical_speed"][env_ids] = velocity[:, 2]
        self.start["horizontal_velocity_x"][env_ids] = velocity[:, 0]
        self.start["horizontal_velocity_y"][env_ids] = velocity[:, 1]
        self.start["tilt"][env_ids] = tilt
        self.start["mass"][env_ids] = start.mass[env_ids].to(torch.float32)
        self.previous["vertical_speed"][env_ids] = velocity[:, 2]
        self.previous["horizontal_speed"][env_ids] = velocity[:, :2].norm(dim=1)
        self.previous["tilt"][env_ids] = tilt
        self.previous["tilt_rate"][env_ids] = start.angular_velocity[env_ids].to(torch.float32).norm(dim=1)
        self.previous_position[env_ids] = position
        self.has_contact[env_ids] = False
        self.contact_time[env_ids] = 0.0
        self.settled_duration[env_ids] = 0.0
        self.peak_leg_force[env_ids] = 0.0
        for values in self.touchdown.values():
            values[env_ids] = 0.0
        self.touchdown_position[env_ids] = 0.0

    def step(self, bus, dt: float) -> None:
        """Update touchdown detection, decide which episodes end, and store their records."""
        truth: TruthState = bus.read(MessageKey.TRUTH_STATE, self)
        tilt = tilt_angle_from_vertical(truth.orientation_quaternion)
        tilt_rate = truth.angular_velocity.norm(dim=1)
        speed = truth.velocity.norm(dim=1)
        leg_force = truth.contact.max(dim=1).values if truth.contact.shape[1] > 0 else torch.zeros_like(speed)
        self.peak_leg_force = torch.maximum(self.peak_leg_force, leg_force)

        first_contact = (leg_force > self.contact_force_threshold) & ~self.has_contact
        for name in self.touchdown:
            self.touchdown[name] = torch.where(first_contact, self.previous[name], self.touchdown[name])
        self.touchdown_position = torch.where(first_contact.unsqueeze(1), self.previous_position, self.touchdown_position)
        self.contact_time = torch.where(first_contact, truth.time, self.contact_time)
        self.has_contact |= first_contact

        at_rest = self.has_contact & (speed < self.settle_speed) & (tilt_rate < self.settle_rate)
        self.settled_duration = torch.where(at_rest, self.settled_duration + dt, torch.zeros_like(speed))
        settled = self.settled_duration >= self.settle_hold
        tipped_over = tilt > self.tip_over_tilt
        settle_timed_out = self.has_contact & (truth.time - self.contact_time > self.settle_timeout)
        altitude = truth.position[:, 2] - self.preset.center_of_mass_height_above_footpad_plane
        out_of_bounds = (truth.position[:, :2].norm(dim=1) > self.maximum_horizontal_distance) | (
            altitude > self.maximum_altitude
        )
        terminated = settled | tipped_over | settle_timed_out | out_of_bounds
        truncated = (truth.time >= self.episode_length - 0.5 * dt) & ~terminated

        finished = terminated | truncated
        if bool(finished.any()):
            self._store_records(finished, truth, tilt, settled, tipped_over, out_of_bounds)

        self.previous["vertical_speed"] = truth.velocity[:, 2].clone()
        self.previous["horizontal_speed"] = truth.velocity[:, :2].norm(dim=1)
        self.previous["tilt"] = tilt
        self.previous["tilt_rate"] = tilt_rate
        self.previous_position = truth.position.clone()
        bus.write(MessageKey.EPISODE_EVENTS, EpisodeEvents(terminated=terminated, truncated=truncated), self)

    # ---- records ----

    def classify(self, settled: Tensor, tipped_over: Tensor, out_of_bounds: Tensor, propellant_used: Tensor) -> Tensor:
        """Return the outcome code of every environment, as if its episode ended now."""
        preset = self.preset
        touchdown = self.touchdown
        too_fast_down = touchdown["vertical_speed"].abs() > preset.maximum_vertical_touchdown_speed
        too_fast_sideways = touchdown["horizontal_speed"] > self.horizontal_speed_limit(touchdown["vertical_speed"])
        too_tilted = touchdown["tilt"] > preset.maximum_touchdown_tilt
        too_fast_turning = touchdown["tilt_rate"] > preset.maximum_touchdown_tilt_rate
        any_limit_broken = too_fast_down | too_fast_sideways | too_tilted | too_fast_turning
        propellant_depleted = propellant_used >= preset.usable_descent_propellant_at_start - 0.01

        outcome = torch.full_like(self.episode_number, OUTCOME_CODE["not_settled"])
        outcome[settled] = OUTCOME_CODE["success"]
        # Later lines have higher priority.
        outcome[tipped_over] = OUTCOME_CODE["tip_over"]
        outcome[too_fast_turning] = OUTCOME_CODE["tilt_rate_at_contact"]
        outcome[too_tilted] = OUTCOME_CODE["tilt_at_contact"]
        outcome[too_fast_sideways] = OUTCOME_CODE["hard_horizontal"]
        outcome[too_fast_down] = OUTCOME_CODE["hard_vertical"]
        outcome[propellant_depleted & any_limit_broken] = OUTCOME_CODE["propellant_depleted"]
        outcome[~self.has_contact] = OUTCOME_CODE["timeout_no_contact"]
        outcome[~self.has_contact & tipped_over] = OUTCOME_CODE["tip_over"]
        outcome[out_of_bounds] = OUTCOME_CODE["out_of_bounds"]
        return outcome

    def _store_records(self, finished, truth, tilt, settled, tipped_over, out_of_bounds) -> None:
        """Append one record row per finished environment (kept on the CPU until drained)."""
        propellant_used = self.start["mass"] - truth.mass
        outcome = self.classify(settled, tipped_over, out_of_bounds, propellant_used)
        degrees = 180.0 / math.pi
        columns = {
            "environment_index": torch.arange(self.number_of_environments, device=self.device),
            "episode_number": self.episode_number,
            "start_altitude_m": self.start["altitude"],
            "start_vertical_speed_mps": self.start["vertical_speed"],
            "start_horizontal_velocity_x_mps": self.start["horizontal_velocity_x"],
            "start_horizontal_velocity_y_mps": self.start["horizontal_velocity_y"],
            "start_tilt_deg": self.start["tilt"] * degrees,
            "start_mass_kg": self.start["mass"],
            "had_contact": self.has_contact,
            "touchdown_time_s": self.contact_time,
            "touchdown_vertical_speed_mps": self.touchdown["vertical_speed"],
            "touchdown_horizontal_speed_mps": self.touchdown["horizontal_speed"],
            "touchdown_tilt_deg": self.touchdown["tilt"] * degrees,
            "touchdown_tilt_rate_deg_per_s": self.touchdown["tilt_rate"] * degrees,
            "touchdown_position_x_m": self.touchdown_position[:, 0],
            "touchdown_position_y_m": self.touchdown_position[:, 1],
            "final_position_x_m": truth.position[:, 0],
            "final_position_y_m": truth.position[:, 1],
            "final_tilt_deg": tilt * degrees,
            "tipped_over": tipped_over,
            "settled": settled,
            "peak_leg_force_n": self.peak_leg_force,
            "propellant_used_kg": propellant_used,
            "episode_duration_s": truth.time,
            "outcome_code": outcome,
            "success": outcome == OUTCOME_CODE["success"],
        }
        self._finished_records.append({name: values[finished].cpu() for name, values in columns.items()})
        self.episode_number += finished.long()

    def drain_episode_records(self) -> dict[str, Tensor]:
        """Return all records stored since the last call (one row per finished episode) and clear them."""
        if not self._finished_records:
            return {}
        merged = {
            name: torch.cat([record[name] for record in self._finished_records]) for name in self._finished_records[0]
        }
        self._finished_records = []
        return merged
