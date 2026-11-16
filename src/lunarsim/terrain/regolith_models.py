# SPDX-License-Identifier: BSD-3-Clause
"""Soil under the footpads and the engine plume over the ground (milestone 5, scenario S8). Pure torch.

**Footpad soil contact (Bekker pressure-sinkage).** The pressure under a plate pressed ``z`` into
the soil is ``p = k z^n`` (Bekker's law with the cohesive and frictional moduli lumped into
``k``). NASA TN D-6850 (Rogers 1972) p. 3 gives the soil of the LM design and of Apollo 11:

- design soil: "a static load of 1.0 lb/in^2 would result in a penetration of 4 inches or less,
  and a dynamic load of 12 lb/in^2 would result in a penetration of 24 inches" (two points of the
  law: n = ln 12 / ln 6 = 1.387);
- Apollo 11: "the Apollo 11 landing indicated a 2 to 3 psi/in. lower boundary of bearing strength"
  (n = 1, k = 2.5 psi/in = 679 kPa/m, the default).

The LM footpad is a dish (spherical cap of radius ``R`` and depth ``h``), so the pressed area grows
with sinkage. The normal force is the pressure integrated over rings of the dish:
``F(z) = sum_rings k (z - s(r))^n dA``, with ``s(r) = R - sqrt(R^2 - r^2)`` the height of the dish
above its lowest point. Lunar soil deforms plastically: on unloading the force falls to zero over
``rebound_depth_m`` below the deepest sinkage reached, so a pad does not spring back. A damper
resists penetration only. Friction is Coulomb with ``friction_coefficient`` (0.4, [D-6850] p. 3),
smoothed below ``friction_smoothing_mps``. Under the LM's own weight (four pads, 7,327 kg) the
Apollo 11 soil sinks a pad about 4.4 cm at rest (4.0 cm at 3 psi/in). For comparison,
[D-6850] p. 18: "A postflight
simulation of the landing dynamics indicated a maximum footpad penetration of 0.5 to 1.5 inches
and a footpad slide distance of 18 to 22 inches" (1.3 to 3.8 cm).

**Plume over the ground (level L1 of design S8).** A cheap, all-environment model of what the
plume does near the ground, as a function of the nozzle height and the engine thrust:

- *reflected-plume force*: ``T * c * exp(-(h / D) / lambda)`` upward along the ground normal,
  acting under the descent stage where the thrust axis meets the ground (so a tilted lander also
  gets a torque). ``c`` and ``lambda`` are model choices to be fitted to the L3 data (design S8);
- *dust density* (0..1): zero above ``erosion_onset_height_m``, rising linearly to one at
  ``full_dust_height_m``, scaled by the thrust over ``reference_thrust_n``. The onset is from
  Apollo 11: "I first noticed that we were, in fact, disturbing the dust on the surface when we
  were something less than 100 feet" (Armstrong, technical debrief, Apollo Lunar Surface Journal,
  "The First Lunar Landing", after 102:45:17);
- *erosion flag* and *ejecta mass rate* ``= dust density * maximum_ejecta_mass_rate_kg_per_s``
  (model choice until L3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

PSI_PER_INCH = 6894.757 / 0.0254  # Pa/m
RINGS = 64  # 0.3 % quadrature error at 1 cm sinkage
SAMPLE_RADIUS_FRACTION = 0.7  # ground samples under a pad: centre and 6 points at 70 % of the pad radius


@dataclass
class BekkerSoil:
    """Bekker pressure-sinkage soil (defaults: Apollo 11, [D-6850] p. 3)."""

    modulus: float = 2.5 * PSI_PER_INCH  # k in Pa / m^n
    exponent: float = 1.0
    rebound_depth_m: float = 0.002  # model choice: elastic rebound of compacted regolith
    damping_n_s_per_m: float = 2000.0  # model choice: near half of critical damping of one pad on the soil
    friction_coefficient: float = 0.4  # [D-6850] p. 3: "0.4 was a realistic value"
    friction_smoothing_mps: float = 0.01  # model choice

    @classmethod
    def from_parameters(cls, parameters: dict | None) -> BekkerSoil:
        """Build the soil from a ``soil`` block: ``preset: apollo11 | lm_design`` and/or explicit values."""
        parameters = dict(parameters or {})
        preset = str(parameters.get("preset", "apollo11"))
        if preset == "lm_design":
            exponent = math.log(12.0) / math.log(6.0)  # 1 psi at 4 in, 12 psi at 24 in
            modulus = 6894.757 / (4.0 * 0.0254) ** exponent
            soil = cls(modulus=modulus, exponent=exponent)
        elif preset == "apollo11":
            soil = cls()
        else:
            raise ValueError(f"unknown soil preset '{preset}' (apollo11 or lm_design)")
        for name in ("modulus", "exponent", "rebound_depth_m", "damping_n_s_per_m", "friction_coefficient", "friction_smoothing_mps"):
            if name in parameters:
                setattr(soil, name, float(parameters[name]))
        return soil


class FootpadSoilContact:
    """Soil forces on dish footpads, batched over environments and legs, with plastic memory."""

    def __init__(self, soil: BekkerSoil, pad_radius: float, sphere_radius: float, pad_depth: float, count: int, legs: int, device) -> None:
        """Precompute the dish rings and the ground sample offsets; allocate the deepest sinkage."""
        self.soil = soil
        self.pad_radius, self.sphere_radius, self.pad_depth = pad_radius, sphere_radius, pad_depth
        edges = torch.linspace(0.0, pad_radius, RINGS + 1, device=device)
        self.ring_radius = 0.5 * (edges[1:] + edges[:-1])
        self.ring_area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        self.ring_height = sphere_radius - torch.sqrt((sphere_radius**2 - self.ring_radius**2).clamp_min(0.0))
        angles = torch.arange(6, device=device, dtype=torch.float32) * (math.pi / 3.0)
        sample_radius = SAMPLE_RADIUS_FRACTION * pad_radius
        self.sample_offsets = torch.cat(
            (torch.zeros(1, 2, device=device), torch.stack((sample_radius * torch.cos(angles), sample_radius * torch.sin(angles)), dim=1))
        )  # (S, 2)
        self.sample_height = torch.cat(
            (torch.zeros(1, device=device), torch.full((6,), sphere_radius - math.sqrt(sphere_radius**2 - sample_radius**2), device=device))
        )  # dish height above its lowest point at the samples
        self.deepest = torch.zeros(count, legs, device=device)

    def reset(self, env_ids: Tensor) -> None:
        """Forget the sinkage history of the given environments."""
        self.deepest[env_ids] = 0.0

    def loading_force(self, sinkage: Tensor) -> Tensor:
        """Return the Bekker normal force for a sinkage of the dish's lowest point, any shape, N."""
        depth = (sinkage.unsqueeze(-1) - self.ring_height).clamp_min(0.0)
        return (self.soil.modulus * depth**self.soil.exponent * self.ring_area).sum(dim=-1)

    def forces(self, bottom: Tensor, velocity: Tensor, height_lookup) -> tuple[Tensor, Tensor]:
        """Return the soil force (N, L, 3) world on each pad and its sinkage (N, L).

        ``bottom`` (N, L, 3) is the world position of each dish's lowest point, ``velocity`` (N, L, 3)
        the pad velocity, ``height_lookup`` anything with ``ground_height_at`` and ``ground_gradient_at``.
        The sinkage is the deepest ground sample under the pad, measured against the dish shape.
        """
        points_x = bottom[..., 0:1] + self.sample_offsets[:, 0]
        points_y = bottom[..., 1:2] + self.sample_offsets[:, 1]
        ground = height_lookup.ground_height_at(points_x, points_y)  # (N, L, S)
        sinkage = (ground - (bottom[..., 2:3] + self.sample_height)).amax(dim=-1).clamp_min(0.0)
        in_contact = sinkage > 0.0
        self.deepest = torch.maximum(self.deepest, sinkage)
        loading = self.loading_force(self.deepest)
        unloading_share = ((sinkage - (self.deepest - self.soil.rebound_depth_m)) / self.soil.rebound_depth_m).clamp(0.0, 1.0)
        normal_force = torch.where(in_contact, loading * unloading_share, torch.zeros_like(loading))
        slope_x, slope_y = height_lookup.ground_gradient_at(bottom[..., 0], bottom[..., 1])
        normal = torch.nn.functional.normalize(torch.stack((-slope_x, -slope_y, torch.ones_like(slope_x)), dim=-1), dim=-1)
        normal_speed = (velocity * normal).sum(dim=-1)
        damping = torch.where(in_contact, self.soil.damping_n_s_per_m * (-normal_speed).clamp_min(0.0), torch.zeros_like(normal_speed))
        normal_force = normal_force + damping
        tangential = velocity - normal_speed.unsqueeze(-1) * normal
        speed = torch.sqrt((tangential**2).sum(dim=-1) + self.soil.friction_smoothing_mps**2)
        friction = -self.soil.friction_coefficient * normal_force.unsqueeze(-1) * tangential / speed.unsqueeze(-1)
        return normal_force.unsqueeze(-1) * normal + friction, sinkage


@dataclass
class PlumeModel:
    """Level-L1 plume-ground interaction (see the module text for every parameter)."""

    ground_effect_coefficient: float = 0.05  # model choice (L3 calibration pending)
    ground_effect_decay_exit_diameters: float = 1.0  # model choice (L3 calibration pending)
    erosion_onset_height_m: float = 100.0 * 0.3048  # Apollo 11, Armstrong: dust "something less than 100 feet"
    full_dust_height_m: float = 3.0  # model choice
    reference_thrust_n: float = 7327.0 * 1.62  # derived: hover at the Apollo 11 touchdown mass ([A11-MR], lunar g)
    maximum_ejecta_mass_rate_kg_per_s: float = 2.0  # model choice (L3 calibration pending)
    enabled: bool = True

    @classmethod
    def from_parameters(cls, parameters: dict | None) -> PlumeModel:
        """Build the model from a ``plume`` block (``enabled`` false turns the plume off)."""
        model = cls()
        for name, value in dict(parameters or {}).items():
            if name in ("model", "erosion", "source"):
                continue
            if not hasattr(model, name):
                raise ValueError(f"unknown plume parameter '{name}'")
            setattr(model, name, bool(value) if name == "enabled" else float(value))
        return model

    def evaluate(
        self, nozzle_exit: Tensor, thrust_direction: Tensor, thrust: Tensor, nozzle_exit_diameter: float,
        center_of_mass: Tensor, height_lookup,
    ) -> dict[str, Tensor]:
        """Return the plume's force and torque on the vehicle (world frame), dust density, erosion and ejecta rate.

        ``nozzle_exit`` (N, 3) world, ``thrust_direction`` (N, 3) unit direction of the exhaust (body
        -z in the world), ``thrust`` (N,) N, ``center_of_mass`` (N, 3) world.
        """
        ground_below = height_lookup.ground_height_at(nozzle_exit[:, 0], nozzle_exit[:, 1])
        height = (nozzle_exit[:, 2] - ground_below).clamp_min(0.0)
        thrust = thrust.clamp_min(0.0)
        count = thrust.shape[0]
        if not self.enabled:
            zeros = torch.zeros(count, 3, device=thrust.device)
            return {"force": zeros, "torque": zeros.clone(), "dust_density": torch.zeros_like(thrust),
                    "eroding": torch.zeros_like(thrust, dtype=torch.bool), "ejecta_mass_rate": torch.zeros_like(thrust), "nozzle_height": height}
        magnitude = thrust * self.ground_effect_coefficient * torch.exp(-(height / nozzle_exit_diameter) / self.ground_effect_decay_exit_diameters)
        force = torch.zeros(count, 3, device=thrust.device)
        force[:, 2] = magnitude
        # The reflected plume pushes on the vehicle base above the point where the thrust axis meets the ground.
        down = (-thrust_direction[:, 2]).clamp_min(0.2)
        reach = height / down
        impingement = nozzle_exit + reach.unsqueeze(1) * thrust_direction
        center_of_pressure = torch.stack((impingement[:, 0], impingement[:, 1], nozzle_exit[:, 2]), dim=1)
        torque = torch.cross(center_of_pressure - center_of_mass, force, dim=1)
        span = max(self.erosion_onset_height_m - self.full_dust_height_m, 1e-6)
        height_share = ((self.erosion_onset_height_m - height) / span).clamp(0.0, 1.0)
        dust = height_share * (thrust / self.reference_thrust_n).clamp(0.0, 1.0)
        return {
            "force": force,
            "torque": torque,
            "dust_density": dust,
            "eroding": dust > 0.0,
            "ejecta_mass_rate": dust * self.maximum_ejecta_mass_rate_kg_per_s,
            "nozzle_height": height,
        }
