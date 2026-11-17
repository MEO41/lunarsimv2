# SPDX-License-Identifier: BSD-3-Clause
"""Plume level L2: soil grains eroded by the rocket plume, as batched torch particles (milestone 5).

Design M5: "L2 check scene ... visual checks of L1". The design named PhysX particles; this
Isaac Sim has no particle tensor view, so the user chose (2026-11-17) a torch grain model that
stays light on the GPU. The method follows Metzger, Lane and Immer (2008) [MLI-08] for Apollo 12:
Roberts' (1963) erosion theory integrated over the lunar size distribution, ejection angles
imposed from the Apollo videos, then lunar gravity and particle ballistics.

Per lander, once per control step:

1. **Gas over the ground** (rings around the plume axis): mass flow ``m = T / v_e``; the
   impinging gas turns outward at ``u_g`` (3,200 m/s, the peak gas speed of the Apollo-like CFD
   of [LMI-08] Fig. 4) in a layer of thickness ``delta = k * sqrt(r^2 + h^2)``; mass conservation
   gives the density ``rho = m / (2 pi r delta u_g)`` (held at its value at ``r = h`` inside the
   impingement zone, where the gas speed rises linearly from zero). ``k`` is fitted so that the
   density 3.8 m from the axis with the nozzle 3 m up at hover thrust is 2e-5 kg/m^3 ([LMI-08] p. 7).
2. **Erosion** (Roberts' excess-shear balance, [MLI-08] Eq. 1): ``(1/2) alpha u dm/dt = tau -
   tau_0`` with the gas shear ``tau = c_f rho u^2 / 2`` and, per size bin, the threshold
   ``tau_0 = A_N^2 (rho_p g D + gamma / D)`` (form of Shao and Lu 2000). The eroded mass is
   split over the size bins by the lunar soil's mass distribution ([MLI-08] Eq. 5) among the
   bins that the shear can move.
3. **Launch**: grains of bin ``k`` leave at ``alpha_k u`` (``alpha_k``: the fraction of the gas
   speed a grain reaches over 10 m of gas, from the drag law of [LMI-08] Eqs. 4-5, computed once)
   at the ejection angle measured in the Apollo videos ([IMH-08] Table 1).
4. **Flight**: ballistic under lunar gravity (no gas outside the thin layer, vacuum); a grain
   ends when it lands on the height map or leaves the domain.

Each simulated particle stands for many grains of one size bin (equal shares of the bin's eroded
mass this step). The particle count per lander is capped; when it is full the oldest particle
is reused (its mass is counted as gone). Outputs per lander: eroded mass and rate, erosion flag,
the optical depth along a sensor line of sight through the grain cloud and the dust density
``1 - exp(-optical depth)``, and the peak lofted number density (to compare with the
1e8-1e13 /m^3 of [IMH-08]).

Sources: [MLI-08] Metzger, Lane, Immer, "Modification of Roberts' theory for rocket exhaust
plumes eroding lunar soil", Earth & Space 2008 (NTRS 20130012034). [LMI-08] Lane, Metzger, Immer,
"Lagrangian trajectory modeling of lunar dust particles", Earth & Space 2008 (NTRS
20130012062). [IMH-08] Immer, Metzger, Hintze et al., "Apollo video photogrammetry estimation of
plume impingement effects", Earth & Space 2008 (arXiv 2104.07669).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY


@dataclass
class GrainState:
    """Per-lander output of the grain model (all (N,))."""

    dust_density: Tensor  # 0..1, 1 - exp(-optical depth) on the sensor line of sight
    optical_depth: Tensor
    eroding: Tensor  # bool
    ejecta_mass_rate: Tensor  # kg/s
    eroded_mass: Tensor  # kg since the episode start
    peak_number_density: Tensor  # grains per m^3, densest cell of the cloud
    particles: Tensor  # simulated particles alive

    @classmethod
    def clear(cls, count: int, device) -> GrainState:
        """Return a state with no grains."""
        zeros = torch.zeros(count, device=device)
        return cls(zeros.clone(), zeros.clone(), torch.zeros(count, dtype=torch.bool, device=device), zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone())


def size_bins(parameters: dict) -> tuple[Tensor, Tensor]:
    """Return the bin diameters (m) and the mass fraction of each bin, from the count distribution P(D) ~ exp(-(lambda D)^a).

    [MLI-08] Eq. 5 with lambda = 36.6 /mm and a = 0.81 (JSC-1A, D above about 4 microns). The bins
    are octaves between ``smallest_diameter_m`` and ``largest_diameter_m``; the bin diameter is the
    geometric middle; the mass fraction integrates P(D) D^3 over the bin.
    """
    smallest, largest = float(parameters["smallest_diameter_m"]), float(parameters["largest_diameter_m"])
    count = int(round(math.log2(largest / smallest)))
    edges = smallest * 2.0 ** torch.arange(count + 1, dtype=torch.float64)
    scale, exponent = float(parameters["size_scale_per_m"]), float(parameters["size_exponent"])
    masses = []
    for index in range(count):
        diameters = torch.linspace(float(edges[index]), float(edges[index + 1]), 400, dtype=torch.float64)
        masses.append(torch.trapezoid(torch.exp(-((scale * diameters) ** exponent)) * diameters**3, diameters))
    masses = torch.stack(masses)
    return torch.sqrt(edges[:-1] * edges[1:]).float(), (masses / masses.sum()).float()


def drag_acceleration_factor(diameter: Tensor, slip: Tensor, gas_density: float, viscosity: float, particle_density: float) -> Tensor:
    """Return the drag acceleration divided by the slip speed, (3 C_D rho / (4 D rho_p)) |slip|, of [LMI-08] Eqs. 1, 4, 5.

    C_D = 24/Re (Re < 2), 18.5 Re^-0.6 (2 <= Re < 500), 0.44 (Re >= 500), Re = D rho |slip| / mu.
    """
    reynolds = (diameter * gas_density * slip / viscosity).clamp_min(1e-12)
    drag = torch.where(reynolds < 2.0, 24.0 / reynolds, torch.where(reynolds < 500.0, 18.5 * reynolds**-0.6, torch.full_like(reynolds, 0.44)))
    return 3.0 * drag * gas_density * slip / (4.0 * diameter * particle_density)


def gas_speed_fractions(diameters: Tensor, parameters: dict) -> Tensor:
    """Return alpha per bin: the fraction of the gas speed a grain reaches after ``acceleration_length_m`` of gas, from rest.

    Integrated once at start with the drag law of [LMI-08]; the gas has the reference speed,
    density and temperature. The step is exact for a fixed drag factor (exponential approach).
    """
    gas_speed = float(parameters["gas_speed_mps"])
    gas_density = float(parameters["reference_gas_density_kg_per_m3"])
    viscosity = float(parameters["viscosity_coefficient"]) * float(parameters["gas_temperature_k"]) ** float(parameters["viscosity_exponent"])
    particle_density = float(parameters["particle_density_kg_per_m3"])
    length = float(parameters["acceleration_length_m"])
    speed = torch.zeros_like(diameters, dtype=torch.float64)
    distance = torch.zeros_like(speed)
    step = 1e-6
    diameters64 = diameters.double()
    for _ in range(200_000):
        factor = drag_acceleration_factor(diameters64, (gas_speed - speed).clamp_min(1e-9), gas_density, viscosity, particle_density)
        new_speed = gas_speed - (gas_speed - speed) * torch.exp(-factor * step)
        moving = distance < length
        distance = torch.where(moving, distance + 0.5 * (speed + new_speed) * step, distance)
        speed = torch.where(moving, new_speed, speed)
        if not bool(moving.any()):
            break
        step = min(step * 1.01, 1e-3)
    return (speed / gas_speed).float()


class PlumeGrainModel:
    """Batched grain particles of the L2 plume (see the module text)."""

    def __init__(self, parameters: dict, number_of_environments: int, device, seed: int = 0) -> None:
        """Read the parameters, build the size bins and alpha, allocate the particle buffers."""
        self.parameters = dict(parameters)
        get = self.parameters.get
        self.count, self.device = number_of_environments, device
        diameters, fractions = size_bins(self.parameters)
        self.diameters = diameters.to(device)
        self.mass_fractions = fractions.to(device)
        self.bins = self.diameters.shape[0]
        self.alpha = gas_speed_fractions(diameters, self.parameters).to(device)
        self.gas_speed = float(get("gas_speed_mps"))
        self.layer_coefficient = float(get("layer_thickness_coefficient"))
        self.friction_coefficient = float(get("skin_friction_coefficient"))
        self.particle_density = float(get("particle_density_kg_per_m3"))
        threshold = float(get("threshold_coefficient")) ** 2 * (self.particle_density * MOON_SURFACE_GRAVITY * self.diameters + float(get("cohesion_n_per_m")) / self.diameters)
        self.threshold_shear = threshold  # (K,) Pa
        self.ejection_tangent = math.tan(math.radians(float(get("ejection_angle_deg"))))
        self.rings = int(get("rings", 40))
        self.ring_reach_heights = float(get("ring_reach_nozzle_heights", 6.0))
        self.minimum_ring_reach = float(get("minimum_ring_reach_m", 20.0))
        self.particles_per_bin_per_step = int(get("particles_per_bin_per_step", 4))
        self.capacity = int(get("particles_per_lander", 2048))
        self.domain_radius = float(get("domain_radius_m", 700.0))
        self.extinction_efficiency = float(get("extinction_efficiency", 2.0))
        self.line_of_sight_depression = math.radians(float(get("line_of_sight_depression_deg", 45.0)))
        self.grid_radius, self.grid_radial_cells = float(get("grid_radius_m", 60.0)), int(get("grid_radial_cells", 40))
        self.grid_height, self.grid_height_cells = float(get("grid_height_m", 6.0)), int(get("grid_height_cells", 24))
        count, capacity = number_of_environments, self.capacity
        self.position = torch.zeros(count, capacity, 3, device=device)
        self.velocity = torch.zeros(count, capacity, 3, device=device)
        self.mass = torch.zeros(count, capacity, device=device)
        self.bin = torch.zeros(count, capacity, dtype=torch.long, device=device)
        self.age = torch.zeros(count, capacity, device=device)
        self.alive = torch.zeros(count, capacity, dtype=torch.bool, device=device)
        self.eroded_mass = torch.zeros(count, device=device)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))
        self.state = GrainState.clear(count, device)

    def reset(self, env_ids: Tensor) -> None:
        """Remove every particle of ``env_ids`` and clear their counters."""
        self.alive[env_ids] = False
        self.mass[env_ids] = 0.0
        self.eroded_mass[env_ids] = 0.0
        for name in GrainState.__dataclass_fields__:
            getattr(self.state, name)[env_ids] = 0

    def ring_erosion(self, nozzle_height: Tensor, mass_flow: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the ring radii (N, R), ring areas (N, R) and the eroded mass rate per area and bin (N, R, K), kg/(m^2 s)."""
        height = nozzle_height.clamp_min(0.5).unsqueeze(1)
        reach = torch.maximum(self.ring_reach_heights * height, torch.full_like(height, self.minimum_ring_reach))
        width = reach / self.rings
        radius = (torch.arange(self.rings, device=self.device) + 0.5).unsqueeze(0) * width  # (N, R)
        area = 2.0 * math.pi * radius * width
        outer = torch.maximum(radius, height)
        thickness = self.layer_coefficient * torch.sqrt(outer**2 + height**2)
        density = mass_flow.unsqueeze(1) / (2.0 * math.pi * outer * thickness * self.gas_speed)
        speed = self.gas_speed * (radius / height).clamp(max=1.0)
        shear = 0.5 * self.friction_coefficient * density * speed**2  # (N, R)
        movable = shear.unsqueeze(2) > self.threshold_shear  # (N, R, K)
        fractions = self.mass_fractions * movable
        share = fractions / fractions.sum(dim=2, keepdim=True).clamp_min(1e-12)
        mean_alpha = (share * self.alpha).sum(dim=2).clamp_min(1e-6)
        mean_threshold = (share * self.threshold_shear).sum(dim=2)
        excess = (shear - mean_threshold).clamp_min(0.0)
        rate = 2.0 * excess / (mean_alpha * speed.clamp_min(1.0))  # [MLI-08] Eq. 1
        return radius, area, rate.unsqueeze(2) * share

    def step(self, nozzle: Tensor, ground_below_nozzle: Tensor, thrust: Tensor, exhaust_velocity: float, ground_height_at, dt: float) -> GrainState:
        """Advance the grains by one control step and return the per-lander state.

        ``nozzle`` (N, 3) world position of the nozzle exit; ``ground_below_nozzle`` (N,) ground
        height under it; ``thrust`` (N,) engine thrust; ``ground_height_at(x, y)`` the height map.
        """
        count, device = self.count, self.device
        nozzle_height = nozzle[:, 2] - ground_below_nozzle
        mass_flow = thrust.clamp_min(0.0) / exhaust_velocity
        radius, area, rate = self.ring_erosion(nozzle_height, mass_flow)
        ring_mass = rate * area.unsqueeze(2) * dt  # (N, R, K) kg this step
        bin_mass = ring_mass.sum(dim=1)  # (N, K)
        step_mass = bin_mass.sum(dim=1)
        self.eroded_mass += step_mass

        # Launch: per bin, a few particles, each carrying an equal share of the bin's mass; the ring is drawn by mass.
        per_bin = self.particles_per_bin_per_step
        new = self.bins * per_bin
        weights = ring_mass.permute(0, 2, 1).reshape(count * self.bins, -1) + 1e-30
        rings = torch.multinomial(weights, per_bin, replacement=True, generator=self.generator).reshape(count, self.bins, per_bin)
        new_bin = torch.arange(self.bins, device=device).view(1, -1, 1).expand(count, -1, per_bin).reshape(count, new)
        new_mass = (bin_mass / per_bin).unsqueeze(2).expand(-1, -1, per_bin).reshape(count, new)
        width = radius[:, :1] * 2.0
        new_radius = (torch.gather(radius, 1, rings.reshape(count, new)) + (torch.rand(count, new, generator=self.generator, device=device) - 0.5) * width)
        azimuth = 2.0 * math.pi * torch.rand(count, new, generator=self.generator, device=device)
        direction = torch.stack((torch.cos(azimuth), torch.sin(azimuth)), dim=2)
        start_xy = nozzle[:, None, :2] + new_radius.unsqueeze(2) * direction
        start_z = ground_height_at(start_xy[..., 0], start_xy[..., 1]) + 0.01
        height = nozzle_height.clamp_min(0.5).unsqueeze(1)
        gas = self.gas_speed * (new_radius / height).clamp(max=1.0)
        speed = self.alpha[new_bin] * gas
        new_velocity = torch.cat((speed.unsqueeze(2) * direction, (speed * self.ejection_tangent).unsqueeze(2)), dim=2)
        launching = new_mass > 0.0

        # Put the new particles in free slots (or reuse the oldest ones).
        priority = torch.where(self.alive, self.age, torch.full_like(self.age, float("inf")))
        slots = torch.topk(priority, new, dim=1).indices  # free slots first, then the oldest
        slots_valid = launching
        rows = torch.arange(count, device=device).unsqueeze(1).expand(-1, new)
        chosen_rows, chosen_slots = rows[slots_valid], slots[slots_valid]
        self.position[chosen_rows, chosen_slots] = torch.cat((start_xy, start_z.unsqueeze(2)), dim=2)[slots_valid]
        self.velocity[chosen_rows, chosen_slots] = new_velocity[slots_valid]
        self.mass[chosen_rows, chosen_slots] = new_mass[slots_valid]
        self.bin[chosen_rows, chosen_slots] = new_bin[slots_valid]
        self.age[chosen_rows, chosen_slots] = 0.0
        self.alive[chosen_rows, chosen_slots] = True
        # A grain launched this step has flown a random part of the step (launches are spread
        # over the step); otherwise a fast grain would always be counted a full step (tens of
        # metres) from where it left, and the cloud near the lander would be missed.
        flight = torch.full_like(self.age, dt)
        flight[chosen_rows, chosen_slots] = dt * torch.rand(chosen_rows.shape[0], generator=self.generator, device=device)

        # Ballistic flight in vacuum; end on landing or leaving the domain.
        gravity = torch.tensor([0.0, 0.0, -MOON_SURFACE_GRAVITY], device=device)
        flight3 = flight.unsqueeze(2)
        self.position += self.velocity * flight3 + 0.5 * gravity * flight3 * flight3
        self.velocity += gravity * flight3
        self.age += flight
        ground = ground_height_at(self.position[..., 0], self.position[..., 1])
        above = self.position[..., 2] - ground
        offset = self.position[..., :2] - nozzle[:, None, :2]
        distance = offset.norm(dim=2)
        self.alive &= (above > 0.0) & (distance < self.domain_radius)

        # Cloud around the plume axis: extinction per volume and grains per volume on an (r, height) grid.
        grains = self.mass / (self.particle_density * math.pi / 6.0 * self.diameters[self.bin] ** 3)
        extinction = self.extinction_efficiency * math.pi / 4.0 * self.diameters[self.bin] ** 2 * grains
        radial_cell = (distance / self.grid_radius * self.grid_radial_cells).long()
        height_cell = (above / self.grid_height * self.grid_height_cells).long()
        inside = self.alive & (radial_cell < self.grid_radial_cells) & (height_cell < self.grid_height_cells)
        cell = (radial_cell.clamp(max=self.grid_radial_cells - 1) * self.grid_height_cells + height_cell.clamp(0, self.grid_height_cells - 1))
        cells = self.grid_radial_cells * self.grid_height_cells
        edges = torch.arange(self.grid_radial_cells + 1, device=device) * (self.grid_radius / self.grid_radial_cells)
        volume = (math.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * (self.grid_height / self.grid_height_cells)).repeat_interleave(self.grid_height_cells)
        extinction_grid = torch.zeros(count, cells, device=device).scatter_add_(1, cell, torch.where(inside, extinction, 0.0)) / volume
        number_grid = torch.zeros(count, cells, device=device).scatter_add_(1, cell, torch.where(inside, grains, 0.0)) / volume

        # Optical depth on a line of sight from the nozzle height, at the set depression angle, to the ground.
        samples = 48
        slant = nozzle_height.clamp_min(0.1) / math.sin(self.line_of_sight_depression)
        along = (torch.arange(samples, device=device) + 0.5) / samples
        sample_radius = along.unsqueeze(0) * (slant * math.cos(self.line_of_sight_depression)).unsqueeze(1)
        sample_height = (1.0 - along.unsqueeze(0)) * nozzle_height.clamp_min(0.1).unsqueeze(1)
        sample_cell = ((sample_radius / self.grid_radius * self.grid_radial_cells).long().clamp(max=self.grid_radial_cells - 1) * self.grid_height_cells
                       + (sample_height / self.grid_height * self.grid_height_cells).long().clamp(max=self.grid_height_cells - 1))
        in_grid = (sample_radius < self.grid_radius) & (sample_height < self.grid_height)
        values = torch.where(in_grid, torch.gather(extinction_grid, 1, sample_cell), 0.0)
        optical_depth = values.sum(dim=1) * (slant / samples)

        self.state = GrainState(
            dust_density=1.0 - torch.exp(-optical_depth),
            optical_depth=optical_depth,
            eroding=step_mass > 0.0,
            ejecta_mass_rate=step_mass / dt,
            eroded_mass=self.eroded_mass.clone(),
            peak_number_density=number_grid.max(dim=1).values,
            particles=self.alive.sum(dim=1).float(),
        )
        return self.state
