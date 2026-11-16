# SPDX-License-Identifier: BSD-3-Clause
"""``metrics: site_safety`` - the touchdown metric plus the true safety of the landing site (S6).

Design, scenario S6: "The surface keeps the true hazard map, so the metric can check the site
that the lander touched. Measure: fraction of landings on a safe site, slope and rock height
under the footpads, divert distance, fuel for the divert, and time to pick the site."

On top of every ``touchdown`` column this metric judges, with the same footprint check as the
onboard map (:mod:`lunarsim.terrain.hazard_assessment`, Apollo LM design surface) but on the
**true** ground (the surface's exact height lookup):

- the footprint under the lander at its first contact: ``touchdown_site_safe``,
  ``touchdown_mean_slope_deg``, ``touchdown_effective_slope_deg``, ``touchdown_relief_m``;
- the nominal target at the start of the episode: ``nominal_site_safe`` and the same numbers, so a
  study can separate "the nominal site was safe anyway" from "the divert found a safe site";
- ``divert_distance_m``: horizontal distance of the first contact from the nominal target.

Time to pick and the divert propellant come from the agent's decision records
(``site_pick_divert``), which the evaluation merges into the same episode table.

Parameters: those of ``touchdown``, plus ``hazard_criteria`` and ``assessment_cell_size_m``
(spacing of the true-height samples, default 0.25 m, the detail grid of ``measured_terrain``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey, SurfaceGeometry, TruthState
from lunarsim.core.registry import register_binding
from lunarsim.slots.metrics.touchdown import TouchdownMetrics
from lunarsim.terrain.hazard_assessment import FootprintAssessor, HazardCriteria

ASSESSED = ("mean_slope_deg", "effective_slope_deg", "relief_m")


@register_binding(SlotName.METRICS, "site_safety")
class SiteSafetyMetrics(TouchdownMetrics):
    """Touchdown judgement plus the true footprint safety of the touched and the nominal site."""

    def __init__(self, parameters, context) -> None:
        """Build the true-ground footprint check and the extra per-episode columns."""
        super().__init__(parameters, context)
        criteria = HazardCriteria.from_parameters(self.parameters.get("hazard_criteria"), self.preset)
        self.assessor = FootprintAssessor(criteria, float(self.parameters.get("assessment_cell_size_m", 0.25)), self.device)
        count, device = self.number_of_environments, self.device
        self.touchdown_site = {name: torch.full((count,), float("nan"), device=device) for name in ASSESSED}
        self.touchdown_site_safe = torch.zeros(count, dtype=torch.bool, device=device)
        self.nominal_site = {name: torch.full((count,), float("nan"), device=device) for name in ASSESSED}
        self.nominal_site_safe = torch.zeros(count, dtype=torch.bool, device=device)
        self.first_contact_position = torch.full((count, 2), float("nan"), device=device)

    def judge_footprints(self, geometry: SurfaceGeometry, world_x: Tensor, world_y: Tensor) -> dict[str, Tensor]:
        """Return the footprint check of the true ground centred at world (x, y)."""
        patches = self.assessor.sample_patches(geometry, world_x, world_y)
        return self.assessor.assess(patches)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Start a new record and judge the nominal target (the new site origin) on the true ground."""
        super().reset(env_ids, gen)
        geometry: SurfaceGeometry = self.context.bus.read(MessageKey.SURFACE_GEOMETRY, self)
        origin = geometry.site_origin[env_ids]
        result = self.judge_footprints(geometry, origin[:, 0], origin[:, 1])
        for name in ASSESSED:
            self.nominal_site[name][env_ids] = result[name]
            self.touchdown_site[name][env_ids] = float("nan")
        self.nominal_site_safe[env_ids] = result["safe"]
        self.touchdown_site_safe[env_ids] = False
        self.first_contact_position[env_ids] = float("nan")

    def step(self, bus, dt: float) -> None:
        """Judge the true footprint under every lander at its first contact, then run the touchdown metric."""
        truth: TruthState = bus.read(MessageKey.TRUTH_STATE, self)
        geometry: SurfaceGeometry = bus.read(MessageKey.SURFACE_GEOMETRY, self)
        leg_force = truth.contact.max(dim=1).values if truth.contact.shape[1] > 0 else torch.zeros_like(truth.time)
        first = ((leg_force > self.contact_force_threshold) | (truth.body_contact > self.contact_force_threshold)) & ~self.has_contact
        if bool(first.any()):
            world = truth.position + geometry.site_origin
            result = self.judge_footprints(geometry, world[:, 0], world[:, 1])
            for name in ASSESSED:
                self.touchdown_site[name] = torch.where(first, result[name], self.touchdown_site[name])
            self.touchdown_site_safe = torch.where(first, result["safe"], self.touchdown_site_safe)
            self.first_contact_position = torch.where(first.unsqueeze(1), truth.position[:, :2].to(torch.float32), self.first_contact_position)
        super().step(bus, dt)

    def extra_record_columns(self) -> dict[str, Tensor]:
        """Return the site-safety columns of every environment."""
        columns = dict(super().extra_record_columns())
        columns["touchdown_site_safe"] = self.touchdown_site_safe.clone()
        columns["nominal_site_safe"] = self.nominal_site_safe.clone()
        for name in ASSESSED:
            columns[f"touchdown_{name}"] = self.touchdown_site[name].clone()
            columns[f"nominal_{name}"] = self.nominal_site[name].clone()
        columns["divert_distance_m"] = self.first_contact_position.norm(dim=1)
        return columns
