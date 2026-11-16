# SPDX-License-Identifier: BSD-3-Clause
"""``agent: site_pick_divert`` - pick a safe site from the onboard hazard map and divert to it (S6).

The agent flies ZEM/ZEV guidance toward the nominal target. While it descends it reads the site
map in its observation (``Observation.site_map``, made by ``nav: hazard_map``) and picks a site:

1. at the first step below ``decision_altitude_m`` (footpad height above the ground, from the
   navigation estimate), and again at every map update while above ``lock_altitude_m``
   (redesignation, as the Apollo crew could move the landing point during the approach,
   NASA TN D-6846 p. 6);
2. among the candidates the map judges safe and that lie within ``maximum_divert_distance_m``
   of the nominal target, it takes the one with the best ``margin - distance_weight_per_m *
   distance`` (distance to the nominal target);
3. it moves its ZEM/ZEV target there and plans again from the current state. A redesignation
   happens only if the new site is better by ``redesignation_margin`` or the current site is no
   longer judged safe.

When no candidate is safe it keeps the nominal target (recorded as ``no_safe_site``). Each episode
leaves one decision record (``drain_decision_records``), which the evaluation merges into the
episode table: decision time and altitude, chosen site, divert distance, the map's margin, the
number of redesignations, and the estimated mass at the first decision (for the divert propellant).
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.agents.classical.zero_effort_miss_zero_effort_velocity_agent import ZeroEffortMissZeroEffortVelocityAgent
from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import Command, NavigationStateLayout, Observation, SiteMapLayout
from lunarsim.core.registry import register_binding


@register_binding(SlotName.AGENT, "site_pick_divert")
class SitePickDivertAgent(ZeroEffortMissZeroEffortVelocityAgent):
    """ZEM/ZEV guidance with site selection from the onboard hazard map.

    Parameters (besides those of ``zem_zev``): ``decision_altitude_m``, ``lock_altitude_m``,
    ``maximum_divert_distance_m``, ``distance_weight_per_m``, ``redesignation_margin``.
    """

    def __init__(self, parameters, number_of_environments, number_of_jets, device, shared_data=None) -> None:
        """Read the selection settings and the site-map geometry from the shared data."""
        super().__init__(parameters, number_of_environments, number_of_jets, device, shared_data)
        get = self.parameters.get
        self.decision_altitude = float(get("decision_altitude_m", 100.0))
        self.lock_altitude = float(get("lock_altitude_m", 30.0))
        self.maximum_divert_distance = float(get("maximum_divert_distance_m", 40.0))
        self.distance_weight = float(get("distance_weight_per_m", 0.005))
        self.redesignation_margin = float(get("redesignation_margin", 0.1))
        geometry = (shared_data or {}).get("site_map_geometry")
        if geometry is None:
            raise ValueError("agent: site_pick_divert needs a navigation binding that makes a site map (nav: hazard_map)")
        rows, columns = int(geometry["rows"]), int(geometry["columns"])
        spacing = float(geometry["candidate_spacing_m"])
        origin_x, origin_y = (float(value) for value in geometry["candidate_origin_m"])
        y, x = torch.meshgrid(
            origin_y + spacing * torch.arange(rows, device=device, dtype=torch.float32),
            origin_x + spacing * torch.arange(columns, device=device, dtype=torch.float32),
            indexing="ij",
        )
        self.candidates = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)  # (C, 2) site frame
        self.nominal = self.gate_position[0:2].clone()
        self.candidate_distance = (self.candidates - self.nominal).norm(dim=1)  # (C,)
        self.reachable = self.candidate_distance <= self.maximum_divert_distance
        count = number_of_environments
        self.episode_number = torch.full((count,), -1, dtype=torch.long, device=device)
        self.decided = torch.zeros(count, dtype=torch.bool, device=device)
        self.no_safe_site = torch.zeros(count, dtype=torch.bool, device=device)
        self.chosen_index = torch.full((count,), -1, dtype=torch.long, device=device)
        self.chosen_score = torch.full((count,), -1e9, device=device)
        self.decision_time = torch.full((count,), float("nan"), device=device)
        self.decision_altitude_reached = torch.full((count,), float("nan"), device=device)
        self.decision_mass = torch.full((count,), float("nan"), device=device)
        self.redesignations = torch.zeros(count, dtype=torch.long, device=device)
        self.elapsed_since_start = torch.zeros(count, device=device)
        self._records: list[dict[str, Tensor]] = []

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Store the finished episode's decision record, then start a new episode."""
        finished = env_ids[self.episode_number[env_ids] >= 0]
        if len(finished) > 0:
            self._store_records(finished)
        super().reset(env_ids, gen)
        self.episode_number[env_ids] += 1
        self.decided[env_ids] = False
        self.no_safe_site[env_ids] = False
        self.chosen_index[env_ids] = -1
        self.chosen_score[env_ids] = -1e9
        self.decision_time[env_ids] = float("nan")
        self.decision_altitude_reached[env_ids] = float("nan")
        self.decision_mass[env_ids] = float("nan")
        self.redesignations[env_ids] = 0
        self.elapsed_since_start[env_ids] = 0.0

    def best_sites(self, site_map: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the best candidate index, its score and whether any safe reachable site exists, each (N,)."""
        count = site_map.shape[0]
        safe = site_map[:, SiteMapLayout.SAFE].reshape(count, -1) > 0.5
        margin = site_map[:, SiteMapLayout.MARGIN].reshape(count, -1)
        allowed = safe & self.reachable.unsqueeze(0)
        score = margin - self.distance_weight * self.candidate_distance.unsqueeze(0)
        score = torch.where(allowed, score, torch.full_like(score, -1e9))
        best_score, best_index = score.max(dim=1)
        return best_index, best_score, allowed.any(dim=1)

    def select_sites(self, observation: Observation) -> None:
        """Pick or redesignate the landing site of every environment that is due."""
        site_map = observation.site_map
        if site_map is None:
            return
        altitude = self.height_above_ground(observation)
        below_decision = altitude <= self.decision_altitude
        may_redesignate = altitude > self.lock_altitude
        best_index, best_score, any_safe = self.best_sites(site_map)
        first = below_decision & ~self.decided
        # The current site's score on the newest map (it may have become unsafe).
        current = self.chosen_index.clamp_min(0)
        count = site_map.shape[0]
        safe = site_map[:, SiteMapLayout.SAFE].reshape(count, -1) > 0.5
        current_safe = torch.gather(safe, 1, current.unsqueeze(1)).squeeze(1) & (self.chosen_index >= 0)
        better = best_score > self.chosen_score + self.redesignation_margin
        redesignate = self.decided & may_redesignate & any_safe & (better | ~current_safe)
        move = (first & any_safe) | redesignate
        if bool(first.any()):
            mass = observation.policy[:, NavigationStateLayout.MASS].squeeze(1)
            self.decision_time = torch.where(first, self.elapsed_since_start, self.decision_time)
            self.decision_altitude_reached = torch.where(first, altitude, self.decision_altitude_reached)
            self.decision_mass = torch.where(first, mass, self.decision_mass)
            self.no_safe_site |= first & ~any_safe
            self.decided |= first
        if bool(move.any()):
            env_ids = torch.nonzero(move).squeeze(1)
            self.redesignations += (redesignate & move).long()
            self.chosen_index = torch.where(move, best_index, self.chosen_index)
            self.chosen_score = torch.where(move, best_score, self.chosen_score)
            self.set_landing_targets(env_ids, self.candidates[best_index[env_ids]])

    def compute_command(self, observation: Observation, control_step_duration: float) -> Command:
        """Pick sites when due, then fly ZEM/ZEV to the current target."""
        self.select_sites(observation)
        self.elapsed_since_start += control_step_duration
        return super().compute_command(observation, control_step_duration)

    def _store_records(self, env_ids: Tensor) -> None:
        """Keep one decision record per finished episode of ``env_ids``."""
        chosen = self.chosen_index[env_ids]
        has_site = chosen >= 0
        site = torch.where(has_site.unsqueeze(1), self.candidates[chosen.clamp_min(0)], self.nominal.expand(len(env_ids), 2))
        record = {
            "environment_index": env_ids.clone(),
            "episode_number": self.episode_number[env_ids].clone(),
            "site_decided": self.decided[env_ids].clone(),
            "no_safe_site": self.no_safe_site[env_ids].clone(),
            "decision_time_s": self.decision_time[env_ids].clone(),
            "decision_altitude_m": self.decision_altitude_reached[env_ids].clone(),
            "decision_mass_estimate_kg": self.decision_mass[env_ids].clone(),
            "chosen_site_x_m": site[:, 0].clone(),
            "chosen_site_y_m": site[:, 1].clone(),
            "chosen_divert_distance_m": (site - self.nominal).norm(dim=1),
            "chosen_map_score": torch.where(has_site, self.chosen_score[env_ids], torch.full_like(self.chosen_score[env_ids], float("nan"))),
            "redesignations": self.redesignations[env_ids].clone(),
        }
        self._records.append({name: values.detach().cpu() for name, values in record.items()})

    def drain_decision_records(self) -> dict[str, Tensor]:
        """Return and forget the decision records of the finished episodes."""
        if not self._records:
            return {}
        merged = {name: torch.cat([record[name] for record in self._records]) for name in self._records[0]}
        self._records = []
        return merged
