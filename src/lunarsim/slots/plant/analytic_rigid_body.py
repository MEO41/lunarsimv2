# SPDX-License-Identifier: BSD-3-Clause
"""``plant: analytic_rigid_body`` - the lander as one rigid body with analytic spring legs, in pure torch.

This plant runs without Isaac Sim and without PhysX. It is the fast simulator of the
analytic-to-PhysX transfer study: it uses the same lander preset, start states, actuators,
observation, action and reward as ``physx_site``, and replaces only the physics.

Model (every PhysX substep, batched over all environments, semi-implicit Euler like PhysX's PGS
solver: first the speeds, then the positions with the new speeds):

- **Body.** One rigid body at the vehicle centre of mass. Forces and torques of the actuators and
  of the surface act in the body frame at the centre of mass. Lunar gravity acts down. The
  rotation follows Euler's equations with the gyroscopic term and a diagonal inertia.
- **Mass and inertia.** The mass falls by the propellant flow of the actuators. Like the PhysX
  plant, all of the mass change is booked on the vehicle body without the legs, and the body's
  share of the inertia is scaled by the same factor; the legs' share stays.
- **Legs** (``leg_contact: spring_strut``). Each leg is a massless strut along the body z axis
  with the preset's stiffness, damping and stroke, ending in a dish footpad (spherical cap).
  The contact point is the lowest point of the dish's sphere, which lies on the dish for tilts
  below the cap's half angle (41 degrees for the Apollo LM). When it is below the flat ground,
  the strut compresses so that the pad stays on the ground; the strut force is
  ``k c + d dc/dt`` (no pull), and past the stroke a stiff end stop adds its own spring and
  damper. The ground carries the strut force divided by the cosine of the tilt, plus the leg's
  own weight (in PhysX the leg is a body of its own whose weight goes straight to the ground).
  Friction is velocity-level Coulomb friction (it may stop the sliding within one substep, but
  never exceeds ``mu N``), as in a PGS contact solver.
- **Body strike.** The bottom rim of the descent stage (a cylinder) is checked against the
  ground; below it, a stiff spring and damper push back with the same friction. Its force is
  ``TruthState.body_contact``. The ascent stage is not checked: on flat ground the lander tips
  over (45 degrees) before it can touch.

Not modelled (compared with ``physx_site``): leg-link mass dynamics, the solver's contact
iterations, the shift of the centre of mass while propellant burns, contact on terrain (flat
ground only). These are the fidelity gaps that the study measures.

Parameters (all optional; every value here is a model choice, not a physical constant):
``leg_contact`` (``spring_strut``), ``footpad_friction_coefficient`` (default: the surface
slot's ``dynamic_friction`` from the shared data, else 0.4), ``stroke_end_stiffness_n_per_m``,
``body_ground_stiffness_n_per_m``, ``minimum_mass_kg``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.assets.lander_preset import LanderPreset
from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import Frame, MessageKey, SurfaceGeometry, TruthState, Wrench
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.core.registry import register_binding
from lunarsim.core.rotation_math import (
    body_up_axis_in_parent,
    quaternion_multiply,
    rotate_vector_body_to_parent,
    rotate_vector_parent_to_body,
)

LEG_CONTACT_MODELS = ("spring_strut",)
# Cosine of the tilt is kept above this when the strut compression is worked out from the
# penetration (a lander lying on its side has tipped over long before).
SMALLEST_TILT_COSINE = 0.2


@register_binding(SlotName.PLANT, "analytic_rigid_body")
class AnalyticRigidBodyPlant(BindingBase):
    """Rigid-body lander with spring-strut legs on flat ground, written in batched torch."""

    requires = frozenset(
        {MessageKey.START_STATE, MessageKey.ACTUATOR_WRENCH, MessageKey.SURFACE_WRENCH, MessageKey.SURFACE_GEOMETRY}
    )
    provides = frozenset({MessageKey.TRUTH_STATE})
    # Which environment class runs this plant (see lunarsim/env/environment_factory.py).
    simulator = "analytic"

    def __init__(self, parameters, context) -> None:
        """Read the preset, work out the leg geometry and the inertia split, and allocate the state."""
        super().__init__(parameters, context)
        preset: LanderPreset | None = context.shared_data.get("lander_preset")
        if preset is None:
            raise ValueError("plant: analytic_rigid_body needs a lander preset in the scenario")
        self.preset = preset
        get = self.parameters.get
        self.leg_contact = str(get("leg_contact", "spring_strut"))
        if self.leg_contact not in LEG_CONTACT_MODELS:
            raise ValueError(f"leg_contact must be one of {LEG_CONTACT_MODELS}, not '{self.leg_contact}'")
        surface_parameters = dict(context.shared_data.get("surface_parameters") or {})
        self.friction_coefficient = float(
            get("footpad_friction_coefficient", surface_parameters.get("dynamic_friction", 0.4))
        )
        self.minimum_mass = float(get("minimum_mass_kg", 0.5 * preset.mass_at_start))
        self.reference_height_above_ground_contact = preset.center_of_mass_height_above_footpad_plane
        device = self.device

        # Legs: strut spring and damper from the preset (the same numbers as the PhysX joint drive).
        self.strut_stiffness = preset.strut_stiffness
        self.strut_damping = preset.strut_damping
        self.strut_stroke = preset.strut_stroke
        quarter_mass = 0.25 * preset.mass_at_start
        self.stroke_end_stiffness = float(get("stroke_end_stiffness_n_per_m", 1.0e7))
        self.stroke_end_damping = 2.0 * math.sqrt(self.stroke_end_stiffness * quarter_mass)
        self.leg_weight = preset.leg_mass * MOON_SURFACE_GRAVITY
        self.footpad_sphere_radius = preset.footpad_sphere_radius
        leg_links = torch.tensor(preset.leg_positions, dtype=torch.float32, device=device)  # (L, 3)
        self.leg_link_positions = leg_links
        sphere_centre_drop = torch.tensor(
            [0.0, 0.0, -preset.footpad_depth + preset.footpad_sphere_radius], dtype=torch.float32, device=device
        )
        self.footpad_sphere_centres = leg_links + sphere_centre_drop  # (L, 3), body frame
        self.number_of_legs = preset.number_of_legs
        self.probe_leg_indices = torch.tensor(
            [leg - 1 for leg in preset.legs_with_surface_probes], dtype=torch.long, device=device
        )
        self.probe_tip_offset = torch.tensor(preset.surface_probe_tip_offset_from_leg_link, dtype=torch.float32, device=device)

        # Body strike: the bottom rim of the descent-stage cylinder.
        centre_x, centre_y, centre_z = preset.descent_stage_center
        self.descent_stage_bottom_centre = torch.tensor(
            [centre_x, centre_y, centre_z - 0.5 * preset.descent_stage_height], dtype=torch.float32, device=device
        )
        self.descent_stage_radius = 0.5 * preset.descent_stage_width
        self.body_ground_stiffness = float(get("body_ground_stiffness_n_per_m", 1.0e7))
        self.body_ground_damping = 2.0 * math.sqrt(self.body_ground_stiffness * preset.mass_at_start)

        # Inertia split: the legs as point masses at their link origins (as in
        # LanderPreset.body_inertia), the rest belongs to the body and scales with its mass.
        legs_inertia = torch.zeros(3, dtype=torch.float32, device=device)
        for x, y, z in preset.leg_positions:
            legs_inertia += preset.leg_mass * torch.tensor([y * y + z * z, x * x + z * z, x * x + y * y], device=device)
        total_inertia = torch.tensor(
            [preset.inertia_about_x, preset.inertia_about_y, preset.inertia_about_z], dtype=torch.float32, device=device
        )
        self.legs_inertia = legs_inertia
        self.body_share_of_inertia = total_inertia - legs_inertia
        self.legs_mass = preset.number_of_legs * preset.leg_mass
        self.body_mass_at_start = preset.body_mass

        count = self.number_of_environments
        self.position_world = torch.zeros(count, 3, device=device)
        self.position_world[:, 2] = self.reference_height_above_ground_contact
        self.velocity = torch.zeros(count, 3, device=device)
        self.orientation = torch.zeros(count, 4, device=device)
        self.orientation[:, 0] = 1.0
        self.angular_velocity_body = torch.zeros(count, 3, device=device)
        self.mass = torch.full((count,), preset.mass_at_start, device=device)
        self.leg_contact_force = torch.zeros(count, self.number_of_legs, device=device)
        self.body_contact_force = torch.zeros(count, device=device)
        self.compression = torch.zeros(count, self.number_of_legs, device=device)

    # ---- slot interface ----

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Put the given environments at the state sampled by the start slot, legs unloaded.

        All updates here and in the step are out of place: training runs the environment inside
        ``torch.inference_mode``, and an in-place write to a tensor made there fails outside it.
        """
        start: TruthState = self.context.bus.read(MessageKey.START_STATE, self)
        geometry: SurfaceGeometry = self.context.bus.read(MessageKey.SURFACE_GEOMETRY, self)
        if start.frame is not Frame.SITE:
            raise ValueError("The analytic plant needs a start state in the SITE frame")
        if geometry.height_field is not None:
            raise ValueError("The analytic plant supports flat ground only (surface: flat)")
        env_ids = env_ids.to(torch.long)
        position = start.position[env_ids].to(torch.float32) + geometry.site_origin[env_ids]
        start_mass = start.mass[env_ids].to(torch.float32)
        mass = torch.where(torch.isnan(start_mass), torch.full_like(start_mass, self.preset.mass_at_start), start_mass)
        self.position_world = self.position_world.index_put((env_ids,), position)
        self.velocity = self.velocity.index_put((env_ids,), start.velocity[env_ids].to(torch.float32))
        self.orientation = self.orientation.index_put((env_ids,), start.orientation_quaternion[env_ids].to(torch.float32))
        self.angular_velocity_body = self.angular_velocity_body.index_put(
            (env_ids,), start.angular_velocity[env_ids].to(torch.float32)
        )
        self.mass = self.mass.index_put((env_ids,), mass)
        zeros_legs = torch.zeros(len(env_ids), self.number_of_legs, device=self.device)
        self.leg_contact_force = self.leg_contact_force.index_put((env_ids,), zeros_legs)
        self.compression = self.compression.index_put((env_ids,), zeros_legs)
        self.body_contact_force = self.body_contact_force.index_put((env_ids,), torch.zeros(len(env_ids), device=self.device))

    def apply_wrench(self, bus, physics_step_duration: float) -> None:
        """Advance every environment by one physics substep under the actuator, surface and ground forces."""
        actuator_wrench: Wrench = bus.read(MessageKey.ACTUATOR_WRENCH, self)
        surface_wrench: Wrench = bus.read(MessageKey.SURFACE_WRENCH, self)
        if surface_wrench.leg_forces is not None:
            raise RuntimeError("The analytic plant has its own leg contact; a soil surface is not supported")
        dt = physics_step_duration
        force_body = actuator_wrench.force + surface_wrench.force
        torque_body = actuator_wrench.torque + surface_wrench.torque
        ground_force_world, ground_torque_body = self._ground_contact(dt)

        orientation = self.orientation
        gravity = torch.tensor([0.0, 0.0, -MOON_SURFACE_GRAVITY], device=self.device)
        force_world = rotate_vector_body_to_parent(orientation, force_body) + ground_force_world
        acceleration = force_world / self.mass.unsqueeze(1) + gravity
        velocity = self.velocity + acceleration * dt
        position = self.position_world + velocity * dt

        inertia = self.inertia(self.mass)  # (N, 3) about the body axes
        rate = self.angular_velocity_body
        gyroscopic = torch.cross(rate, inertia * rate, dim=1)
        angular_acceleration = (torque_body + ground_torque_body - gyroscopic) / inertia
        rate = rate + angular_acceleration * dt
        orientation = quaternion_multiply(orientation, rotation_quaternion_from_rate(rate, dt))
        orientation = orientation / orientation.norm(dim=1, keepdim=True)

        mass_flow_rate = actuator_wrench.mass_flow_rate + surface_wrench.mass_flow_rate
        self.mass = torch.clamp_min(self.mass - mass_flow_rate * dt, self.minimum_mass)
        self.velocity = velocity
        self.position_world = position
        self.angular_velocity_body = rate
        self.orientation = orientation

    def step(self, bus, dt: float) -> None:
        """Write the current state as TruthState in the SITE frame."""
        geometry: SurfaceGeometry = bus.read(MessageKey.SURFACE_GEOMETRY, self)
        position = self.position_world
        ground_height = geometry.ground_height_at(position[:, 0], position[:, 1])
        state = TruthState(
            frame=Frame.SITE,
            time=self.context.clock.time,
            position=position - geometry.site_origin,
            velocity=self.velocity.clone(),
            orientation_quaternion=self.orientation.clone(),
            angular_velocity=self.angular_velocity_body.clone(),
            mass=self.mass.clone(),
            contact=self.leg_contact_force.clone(),
            footpad_height_above_ground=position[:, 2] - self.reference_height_above_ground_contact - ground_height,
            body_contact=self.body_contact_force.clone(),
            surface_probe_contact=self._surface_probe_contact(),
        )
        bus.write(MessageKey.TRUTH_STATE, state, self)

    # ---- model ----

    def inertia(self, mass: Tensor) -> Tensor:
        """Return the diagonal inertia about the body axes for the given vehicle masses, (N, 3), kg m^2."""
        body_scale = (mass - self.legs_mass) / self.body_mass_at_start
        return self.legs_inertia + self.body_share_of_inertia * body_scale.unsqueeze(1)

    def strut_compression(self) -> Tensor:
        """Return how far each strut is compressed, (N, legs), m (clamped to the stroke). Used by tests and plots."""
        return self.compression.clamp(0.0, self.strut_stroke)

    def _ground_contact(self, dt: float) -> tuple[Tensor, Tensor]:
        """Return the summed ground force (world frame) and its torque about the centre of mass (body frame).

        Also stores the per-leg contact force magnitudes, the body-strike force and the strut
        compressions of this substep.
        """
        count, legs = self.number_of_environments, self.number_of_legs
        orientation = self.orientation
        position = self.position_world
        velocity = self.velocity
        angular_velocity_world = rotate_vector_body_to_parent(orientation, self.angular_velocity_body)
        up = body_up_axis_in_parent(orientation)
        tilt_cosine = up[:, 2].clamp_min(SMALLEST_TILT_COSINE)  # (N,)

        # Legs: lowest point of each dish sphere, its depth below the ground, and its velocity.
        orientation_per_leg = orientation.unsqueeze(1).expand(count, legs, 4)
        centres = position.unsqueeze(1) + rotate_vector_body_to_parent(
            orientation_per_leg, self.footpad_sphere_centres.expand(count, legs, 3)
        )
        lowest = centres.clone()
        lowest[..., 2] = lowest[..., 2] - self.footpad_sphere_radius
        lever = lowest - position.unsqueeze(1)
        point_velocity = velocity.unsqueeze(1) + torch.cross(angular_velocity_world.unsqueeze(1).expand(count, legs, 3), lever, dim=2)
        penetration = -lowest[..., 2]  # flat ground at world z = 0
        in_contact = penetration > 0.0
        compression = torch.where(in_contact, penetration / tilt_cosine.unsqueeze(1), torch.zeros_like(penetration))
        compression_rate = -point_velocity[..., 2] / tilt_cosine.unsqueeze(1)
        strut_force = self.strut_stiffness * compression + self.strut_damping * compression_rate
        beyond_stroke = compression - self.strut_stroke
        strut_force = strut_force + torch.where(
            beyond_stroke > 0.0,
            self.stroke_end_stiffness * beyond_stroke + self.stroke_end_damping * compression_rate,
            torch.zeros_like(beyond_stroke),
        )
        strut_force = torch.where(in_contact, strut_force.clamp_min(0.0), torch.zeros_like(strut_force))
        normal_force = torch.where(
            in_contact, strut_force / tilt_cosine.unsqueeze(1) + self.leg_weight, torch.zeros_like(strut_force)
        )
        # Body strike: lowest point of the descent-stage bottom rim.
        rim_centre = position + rotate_vector_body_to_parent(orientation, self.descent_stage_bottom_centre.expand(count, 3))
        downward_in_plane = torch.zeros_like(up)
        downward_in_plane[:, 2] = -1.0
        downward_in_plane = downward_in_plane - (downward_in_plane * up).sum(dim=1, keepdim=True) * up
        in_plane_length = downward_in_plane.norm(dim=1, keepdim=True)
        rim_direction = torch.where(in_plane_length > 1e-6, downward_in_plane / in_plane_length.clamp_min(1e-6), torch.zeros_like(up))
        rim_point = rim_centre + self.descent_stage_radius * rim_direction
        rim_lever = rim_point - position
        rim_velocity = velocity + torch.cross(angular_velocity_world, rim_lever, dim=1)
        rim_penetration = -rim_point[:, 2]
        body_normal = self.body_ground_stiffness * rim_penetration - self.body_ground_damping * rim_velocity[:, 2]
        rim_in_contact = rim_penetration > 0.0
        body_normal = torch.where(rim_in_contact, body_normal.clamp_min(0.0), torch.zeros_like(body_normal))

        # Friction: the mass that a sideways push at a contact point feels, counting the turn of
        # the body it causes (1/M + r^2/I), shared among all points in contact. With the smallest
        # inertia and the full lever this is a lower bound, so the friction never overshoots.
        contacts = in_contact.sum(dim=1) + rim_in_contact.long()
        contacts = contacts.clamp_min(1).to(torch.float32)  # (N,)
        smallest_inertia = self.inertia(self.mass).min(dim=1).values  # (N,)
        inverse_mass = 1.0 / self.mass
        leg_effective_mass = 1.0 / (
            contacts.unsqueeze(1) * (inverse_mass.unsqueeze(1) + (lever * lever).sum(dim=2) / smallest_inertia.unsqueeze(1))
        )
        rim_effective_mass = 1.0 / (contacts * (inverse_mass + (rim_lever * rim_lever).sum(dim=1) / smallest_inertia))
        leg_forces = self._with_friction(normal_force, point_velocity, leg_effective_mass, dt)  # (N, L, 3)
        body_force = self._with_friction(body_normal.unsqueeze(1), rim_velocity.unsqueeze(1), rim_effective_mass.unsqueeze(1), dt).squeeze(1)

        total_force = leg_forces.sum(dim=1) + body_force
        total_torque_world = torch.cross(lever, leg_forces, dim=2).sum(dim=1) + torch.cross(rim_lever, body_force, dim=1)
        self.leg_contact_force = leg_forces.norm(dim=2)
        self.body_contact_force = body_force.norm(dim=1)
        self.compression = compression
        return total_force, rotate_vector_parent_to_body(orientation, total_torque_world)

    def _with_friction(self, normal_force: Tensor, point_velocity: Tensor, effective_mass: Tensor, dt: float) -> Tensor:
        """Return contact forces (..., 3): the normal force up plus Coulomb friction against the sliding velocity.

        Velocity-level Coulomb friction, as a PGS contact solver does it: the friction tries to
        stop the sliding of the contact point within this substep (``effective_mass`` times the
        sliding velocity over ``dt``), but never pushes harder than ``mu N``. So a pad can stick
        (static friction) and a sliding pad feels ``mu N``, with no chatter at small speeds.
        """
        sliding_velocity = point_velocity[..., 0:2]
        sliding_speed = sliding_velocity.norm(dim=-1, keepdim=True)
        stopping_force = effective_mass.unsqueeze(-1) * sliding_speed / dt
        friction_magnitude = torch.minimum(stopping_force, self.friction_coefficient * normal_force.unsqueeze(-1))
        friction = -friction_magnitude * sliding_velocity / sliding_speed.clamp_min(1e-9)
        return torch.cat((friction, normal_force.unsqueeze(-1)), dim=-1)

    def _surface_probe_contact(self) -> Tensor:
        """Return 1.0 where at least one probe tip is at or below the ground, else 0.0, (N,).

        A probe hangs from its footpad along the body -z axis; a compressed strut lifts it with the pad.
        """
        if self.probe_leg_indices.numel() == 0:
            return torch.zeros(self.number_of_environments, device=self.device)
        count = self.number_of_environments
        probes = self.probe_leg_indices.numel()
        compression = self.strut_compression()[:, self.probe_leg_indices]  # (N, probes)
        offset = self.leg_link_positions[self.probe_leg_indices] + self.probe_tip_offset  # (probes, 3)
        offset = offset.expand(count, probes, 3).clone()
        offset[..., 2] = offset[..., 2] + compression
        orientation = self.orientation.unsqueeze(1).expand(count, probes, 4)
        tips = self.position_world.unsqueeze(1) + rotate_vector_body_to_parent(orientation, offset)
        return (tips[..., 2] <= 0.0).any(dim=1).to(torch.float32)


def rotation_quaternion_from_rate(body_rate: Tensor, dt: float) -> Tensor:
    """Return the (w, x, y, z) quaternion of turning at ``body_rate`` (rad/s, body frame) for ``dt`` seconds.

    Exact for a constant rate: angle |w| dt about the axis w / |w|. Near zero rate it tends to the identity.
    """
    angle = body_rate.norm(dim=1, keepdim=True) * dt
    half_angle = 0.5 * angle
    # sin(a/2) / |w| written so that it stays finite at |w| = 0 (limit dt / 2).
    scale = torch.where(angle > 1e-8, torch.sin(half_angle) / body_rate.norm(dim=1, keepdim=True).clamp_min(1e-12), torch.full_like(angle, 0.5 * dt))
    return torch.cat((torch.cos(half_angle), body_rate * scale), dim=1)
