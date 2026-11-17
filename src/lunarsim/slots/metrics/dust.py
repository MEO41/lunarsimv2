# SPDX-License-Identifier: BSD-3-Clause
"""``metrics: dust`` - the touchdown metric plus plume, dust and soil numbers (scenario S8, milestone 5).

Design S8: "Measure: touchdown metrics, footpad sinkage, time with sensors blinded by dust, ejecta
mass, and the plume torque at contact." Columns added to every ``touchdown`` record:

- ``dust_blinded_time_s``: time with the dust density at or above ``blinded_dust_density``
  (model choice 0.5: half the LiDAR pixels lost with the default dust fault);
- ``erosion_start_nozzle_height_m``: nozzle height when the plume first moved soil (NaN if never);
- ``ejecta_mass_kg``: ejecta mass rate integrated over the episode;
- ``peak_dust_density``;
- ``plume_force_at_contact_n``, ``plume_torque_at_contact_n_m``: the plume wrench at first contact;
- ``footpad_sinkage_max_m`` (deepest pad over the episode) and ``footpad_sinkage_final_mean_m``.

With the L2 grains running (``surface.grains``), the same numbers from the grains, prefixed
``grain_``: ``grain_dust_blinded_time_s``, ``grain_erosion_start_nozzle_height_m``,
``grain_eroded_mass_kg``, ``grain_peak_dust_density``, plus ``grain_peak_optical_depth`` and
``grain_peak_number_density_per_m3`` (densest cell of the cloud, for the 1e8-1e13 /m^3 of the
Apollo videos).
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey, RegolithState, TruthState
from lunarsim.core.registry import register_binding
from lunarsim.slots.metrics.touchdown import TouchdownMetrics


@register_binding(SlotName.METRICS, "dust")
class DustMetrics(TouchdownMetrics):
    """Touchdown judgement plus the plume, dust and soil columns of S8."""

    requires = TouchdownMetrics.requires | frozenset({MessageKey.REGOLITH_STATE})

    def __init__(self, parameters, context) -> None:
        """Allocate the per-episode plume and soil records."""
        super().__init__(parameters, context)
        self.blinded_density = float(self.parameters.get("blinded_dust_density", 0.5))
        count, device = self.number_of_environments, self.device
        legs = context.shared_data["lander_preset"].number_of_legs
        self.blinded_time = torch.zeros(count, device=device)
        self.erosion_start_height = torch.full((count,), float("nan"), device=device)
        self.ejecta_mass = torch.zeros(count, device=device)
        self.peak_dust = torch.zeros(count, device=device)
        self.plume_force_at_contact = torch.full((count,), float("nan"), device=device)
        self.plume_torque_at_contact = torch.full((count,), float("nan"), device=device)
        self.deepest_sinkage = torch.zeros(count, device=device)
        self.final_sinkage = torch.zeros(count, legs, device=device)
        self.grain_records = {name: torch.zeros(count, device=device) for name in ("grain_dust_blinded_time_s", "grain_eroded_mass_kg", "grain_peak_dust_density", "grain_peak_optical_depth", "grain_peak_number_density_per_m3")}
        self.grain_records["grain_erosion_start_nozzle_height_m"] = torch.full((count,), float("nan"), device=device)
        self.has_grains = False

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Start new plume and soil records for ``env_ids``."""
        super().reset(env_ids, gen)
        self.blinded_time[env_ids] = 0.0
        self.erosion_start_height[env_ids] = float("nan")
        self.ejecta_mass[env_ids] = 0.0
        self.peak_dust[env_ids] = 0.0
        self.plume_force_at_contact[env_ids] = float("nan")
        self.plume_torque_at_contact[env_ids] = float("nan")
        self.deepest_sinkage[env_ids] = 0.0
        self.final_sinkage[env_ids] = 0.0
        for name, values in self.grain_records.items():
            values[env_ids] = float("nan") if name == "grain_erosion_start_nozzle_height_m" else 0.0

    def step(self, bus, dt: float) -> None:
        """Accumulate the plume and soil numbers, then run the touchdown metric."""
        regolith: RegolithState = bus.read(MessageKey.REGOLITH_STATE, self)
        truth: TruthState = bus.read(MessageKey.TRUTH_STATE, self)
        density = regolith.dust_density
        self.blinded_time += (density >= self.blinded_density).float() * dt
        first_erosion = regolith.eroding & torch.isnan(self.erosion_start_height)
        self.erosion_start_height = torch.where(first_erosion, regolith.nozzle_height, self.erosion_start_height)
        self.ejecta_mass += regolith.ejecta_mass_rate * dt
        self.peak_dust = torch.maximum(self.peak_dust, density)
        self.deepest_sinkage = torch.maximum(self.deepest_sinkage, regolith.footpad_sinkage.amax(dim=1))
        self.final_sinkage = regolith.footpad_sinkage.clone()
        leg_force = truth.contact.max(dim=1).values if truth.contact.shape[1] > 0 else torch.zeros_like(truth.time)
        first = ((leg_force > self.contact_force_threshold) | (truth.body_contact > self.contact_force_threshold)) & ~self.has_contact
        self.plume_force_at_contact = torch.where(first, regolith.plume_force.norm(dim=1), self.plume_force_at_contact)
        self.plume_torque_at_contact = torch.where(first, regolith.plume_torque.norm(dim=1), self.plume_torque_at_contact)
        grains = regolith.grains
        if grains is not None:
            self.has_grains = True
            records = self.grain_records
            records["grain_dust_blinded_time_s"] += (grains.dust_density >= self.blinded_density).float() * dt
            start = records["grain_erosion_start_nozzle_height_m"]
            records["grain_erosion_start_nozzle_height_m"] = torch.where(grains.eroding & torch.isnan(start), regolith.nozzle_height, start)
            records["grain_eroded_mass_kg"] = grains.eroded_mass.clone()
            records["grain_peak_dust_density"] = torch.maximum(records["grain_peak_dust_density"], grains.dust_density)
            records["grain_peak_optical_depth"] = torch.maximum(records["grain_peak_optical_depth"], grains.optical_depth)
            records["grain_peak_number_density_per_m3"] = torch.maximum(records["grain_peak_number_density_per_m3"], grains.peak_number_density)
        super().step(bus, dt)

    def extra_record_columns(self) -> dict[str, Tensor]:
        """Return the plume, dust and soil columns of every environment."""
        columns = dict(super().extra_record_columns())
        columns.update(
            {
                "dust_blinded_time_s": self.blinded_time.clone(),
                "erosion_start_nozzle_height_m": self.erosion_start_height.clone(),
                "ejecta_mass_kg": self.ejecta_mass.clone(),
                "peak_dust_density": self.peak_dust.clone(),
                "plume_force_at_contact_n": self.plume_force_at_contact.clone(),
                "plume_torque_at_contact_n_m": self.plume_torque_at_contact.clone(),
                "footpad_sinkage_max_m": self.deepest_sinkage.clone(),
                "footpad_sinkage_final_mean_m": self.final_sinkage.mean(dim=1),
            }
        )
        if self.has_grains:
            columns.update({name: values.clone() for name, values in self.grain_records.items()})
        return columns
