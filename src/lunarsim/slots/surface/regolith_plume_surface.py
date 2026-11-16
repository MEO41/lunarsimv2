# SPDX-License-Identifier: BSD-3-Clause
"""``surface: regolith_plume_l1`` - measured terrain with Bekker soil under the footpads and the L1 plume (S8).

Design S8: "plume forces, dust and footpad sinkage" in the last metres; "the surface sends dust
density to the faults slot". This binding is the ``measured_terrain`` surface (Apollo 17) plus,
every PhysX substep:

- **soil contact** (``contact: bekker``, default): the footpads have no PhysX contact (the
  surface asks the plant for ``footpad_contact: soil`` through the shared data); the Bekker soil
  of :mod:`lunarsim.terrain.regolith_models` pushes on each footpad from its current position and
  velocity (read from the plant, the neighbouring truth-side slot, as the ground would);
  ``contact: rigid`` keeps PhysX footpad contact;
- **plume L1**: the reflected-plume force and torque on the vehicle from the engine thrust (the
  actuators' ``engine_thrust``) and the nozzle height, and the dust density, erosion flag and
  ejecta rate; ``plume.enabled: false`` turns it off (the design's same-seed comparison).

Outputs: ``surface_wrench`` (plume wrench on the body, soil forces on the legs) and
``regolith_state`` (written at reset too, so the dust fault can read it before the first step).
Parameters: those of ``measured_terrain``, plus ``contact``, ``soil`` and ``plume`` blocks.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey, RegolithState, SurfaceGeometry, Wrench
from lunarsim.core.registry import register_binding
from lunarsim.core.rotation_math import rotate_vector_body_to_parent, rotate_vector_parent_to_body
from lunarsim.slots.surface.measured_terrain import MeasuredTerrainSurface
from lunarsim.terrain.regolith_models import BekkerSoil, FootpadSoilContact, PlumeModel


@register_binding(SlotName.SURFACE, "regolith_plume_l1")
class RegolithPlumeSurface(MeasuredTerrainSurface):
    """Measured terrain, Bekker footpad soil and the level-L1 plume."""

    requires = frozenset({MessageKey.ACTUATOR_WRENCH})
    provides = frozenset({MessageKey.SURFACE_WRENCH, MessageKey.SURFACE_GEOMETRY, MessageKey.REGOLITH_STATE})

    def __init__(self, parameters, context) -> None:
        """Read the contact mode, the soil and the plume, and tell the plant to leave the footpads to the soil."""
        super().__init__(parameters, context)
        self.contact = str(self.parameters.get("contact", "bekker"))
        if self.contact not in ("bekker", "rigid"):
            raise ValueError(f"surface.contact must be 'bekker' or 'rigid', not '{self.contact}'")
        if self.contact == "bekker":
            context.shared_data["footpad_contact"] = "soil"
        self.soil = BekkerSoil.from_parameters(self.parameters.get("soil"))
        self.plume = PlumeModel.from_parameters(self.parameters.get("plume"))
        preset = context.shared_data["lander_preset"]
        self.preset = preset
        self.legs = preset.number_of_legs
        self.nozzle_exit_body = torch.tensor(preset.nozzle_exit_center, dtype=torch.float32, device=self.device)
        self.exhaust_body = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        self.footpads = FootpadSoilContact(
            self.soil, preset.footpad_radius, preset.footpad_sphere_radius, preset.footpad_depth,
            self.number_of_environments, self.legs, self.device,
        )
        self.state = RegolithState.clear(self.number_of_environments, self.legs, self.device)

    def initialize(self, environment) -> None:
        """Set up the terrain, find the plant, and write a clear regolith state."""
        super().initialize(environment)
        self.plant = environment.bindings[SlotName.PLANT]
        if self.contact == "bekker" and not getattr(self.plant, "soil_contact", False):
            raise RuntimeError("surface contact 'bekker' needs plant.footpad_contact 'soil' (set automatically for physx_site)")
        self.context.bus.write(MessageKey.REGOLITH_STATE, self.state, self)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Draw new targets, forget the soil history, and clear the regolith state of ``env_ids``."""
        super().reset(env_ids, gen)
        self.footpads.reset(env_ids)
        clear = RegolithState.clear(len(env_ids), self.legs, self.device)
        for name in ("dust_density", "eroding", "ejecta_mass_rate", "nozzle_height", "plume_force", "plume_torque", "footpad_sinkage"):
            getattr(self.state, name)[env_ids] = getattr(clear, name)
        self.context.bus.write(MessageKey.REGOLITH_STATE, self.state, self)

    def step(self, bus, dt: float) -> None:
        """Compute the soil forces and the plume this substep; write the surface wrench and the regolith state."""
        geometry: SurfaceGeometry = self.geometry
        actuators: Wrench = bus.read(MessageKey.ACTUATOR_WRENCH, self)
        position, orientation = self.plant.body_pose()
        count = position.shape[0]
        thrust = actuators.engine_thrust if actuators.engine_thrust is not None else torch.zeros(count, device=self.device)
        nozzle = position + rotate_vector_body_to_parent(orientation, self.nozzle_exit_body.expand(count, 3))
        exhaust = rotate_vector_body_to_parent(orientation, self.exhaust_body.expand(count, 3))
        plume = self.plume.evaluate(nozzle, exhaust, thrust, float(self.preset.nozzle_exit_diameter), position, geometry)
        force_body = rotate_vector_parent_to_body(orientation, plume["force"])
        torque_body = rotate_vector_parent_to_body(orientation, plume["torque"])
        leg_forces = leg_points = None
        sinkage = torch.zeros(count, self.legs, device=self.device)
        if self.contact == "bekker":
            bottom, velocity = self.plant.footpad_states()
            leg_forces, sinkage = self.footpads.forces(bottom, velocity, geometry)
            leg_points = bottom
        bus.write(
            MessageKey.SURFACE_WRENCH,
            Wrench(force=force_body, torque=torque_body, leg_forces=leg_forces, leg_force_points=leg_points),
            self,
        )
        self.state = RegolithState(
            dust_density=plume["dust_density"], eroding=plume["eroding"], ejecta_mass_rate=plume["ejecta_mass_rate"],
            nozzle_height=plume["nozzle_height"], plume_force=force_body, plume_torque=torque_body, footpad_sinkage=sinkage,
        )
        bus.write(MessageKey.REGOLITH_STATE, self.state, self)
