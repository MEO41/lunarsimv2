# SPDX-License-Identifier: BSD-3-Clause
"""``plant: orbit_f64`` (and ``local_f64``) - the lander as a float64 rigid body around a point-mass Moon (milestone 6).

Design: "``orbit_f64`` torch propagator; orbit tests" and the ``hybrid`` handoff to PhysX at the
low gate. This plant integrates, in float64 torch for N environments:

- translation: point-mass lunar gravity ``-mu r / |r|^3`` (mu from DE440) plus the actuator and
  surface forces; in the Moon-fixed frame also the Coriolis and centrifugal terms
  ``-2 w x v - w x (w x r)``;
- rotation: Euler's equations with the preset's body inertia, scaled by mass over the preset's
  start mass (the same rule as ``physx_site``), body rates in the body frame;
- mass: minus the actuators' mass flow.

Fourth-order Runge-Kutta every PhysX substep (the force is held over the substep). There is no
ground contact: an orbit phase ends at a gate (``metrics: phase_gate``) well above the ground.

Parameter ``frame``: ``MCI`` (coast and descent orbit), ``MCMF`` (braking and approach), or
``SITE`` (registry name ``local_f64``: east-north-up at a site, gravity toward the Moon's centre
``radius`` below the site origin, for S5 near one landing site). ``site``: ``latitude_deg``,
``longitude_deg``, ``radius_m`` (default the Moon's mean radius), used for the SITE frame and for
the height above the ground in every frame (height above the site radius, plus the surface's
height lookup near the site in the SITE frame). Model choices: no third bodies, no lunar gravity
harmonics, no mascons.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.frames import SiteLocation, convert_truth_state
from lunarsim.core.message_types import Frame, MessageKey, SurfaceGeometry, TruthState, Wrench
from lunarsim.core.physical_constants import MOON_GRAVITATIONAL_PARAMETER, MOON_MEAN_RADIUS, MOON_ROTATION_RATE
from lunarsim.core.registry import register_binding
from lunarsim.core.rotation_math import quaternion_multiply, rotate_vector_body_to_parent

FRAMES = {"MCI": Frame.MOON_CENTERED_INERTIAL, "MCMF": Frame.MOON_CENTERED_MOON_FIXED, "SITE": Frame.SITE}


def site_from_parameters(parameters: dict | None) -> SiteLocation:
    """Return the SiteLocation of a ``site`` block (default: Apollo 17 LM site on the mean sphere)."""
    site = dict(parameters or {})
    return SiteLocation(
        latitude_deg=float(site.get("latitude_deg", 20.19091)),
        longitude_deg=float(site.get("longitude_deg", 30.77168)),
        radius_m=float(site.get("radius_m", MOON_MEAN_RADIUS)),
    )


def quaternion_derivative(orientation: Tensor, body_rates: Tensor) -> Tensor:
    """Return dq/dt = 0.5 q (x) (0, w) for body rates ``w`` (N, 3)."""
    rates = torch.cat((torch.zeros_like(body_rates[:, :1]), body_rates), dim=1)
    return 0.5 * quaternion_multiply(orientation, rates)


@register_binding(SlotName.PLANT, "orbit_f64")
class OrbitFloat64Plant(BindingBase):
    """Float64 6-degree-of-freedom lander around a point-mass Moon (see the module text)."""

    requires = frozenset({MessageKey.START_STATE, MessageKey.ACTUATOR_WRENCH, MessageKey.SURFACE_WRENCH, MessageKey.SURFACE_GEOMETRY})
    provides = frozenset({MessageKey.TRUTH_STATE})
    reads_truth = True
    default_frame = "MCI"

    def __init__(self, parameters, context) -> None:
        """Read the frame and the site, and allocate the float64 state."""
        super().__init__(parameters, context)
        frame_name = str(self.parameters.get("frame", self.default_frame))
        if frame_name not in FRAMES:
            raise ValueError(f"plant.frame must be one of {sorted(FRAMES)}, not '{frame_name}'")
        self.frame = FRAMES[frame_name]
        self.site = site_from_parameters(self.parameters.get("site"))
        preset = context.shared_data["lander_preset"]
        self.preset = preset
        count, device, dtype = self.number_of_environments, self.device, torch.float64
        self.position = torch.zeros(count, 3, dtype=dtype, device=device)
        self.velocity = torch.zeros(count, 3, dtype=dtype, device=device)
        self.orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device).repeat(count, 1)
        self.body_rates = torch.zeros(count, 3, dtype=dtype, device=device)
        self.mass = torch.full((count,), float(preset.mass_at_start), dtype=dtype, device=device)
        self.time = torch.zeros(count, dtype=dtype, device=device)
        inertia = torch.tensor(preset.body_inertia, dtype=dtype, device=device)
        self.reference_inertia = torch.diag(inertia) if inertia.numel() == 3 else inertia.reshape(3, 3)  # principal moments
        self.reference_mass = float(preset.mass_at_start)
        self.minimum_mass = float(preset.mass_without_usable_propellant)
        self.spin = torch.tensor([0.0, 0.0, MOON_ROTATION_RATE if self.frame is Frame.MOON_CENTERED_MOON_FIXED else 0.0], dtype=dtype, device=device)
        # Moon centre in the plant frame (the SITE frame's origin is on the ground).
        self.moon_center = torch.zeros(3, dtype=dtype, device=device)
        if self.frame is Frame.SITE:
            self.moon_center[2] = -self.site.radius_m
        self.center_of_mass_height = float(preset.center_of_mass_height_above_footpad_plane)

    def setup_scene(self, environment) -> None:
        """Nothing to spawn: this plant lives in torch only."""

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Take the start state of the given environments (converted into the plant's frame)."""
        start: TruthState = self.context.bus.read(MessageKey.START_STATE, self)
        state = convert_truth_state(start, self.frame, self.site)
        self.position[env_ids] = state.position[env_ids].to(torch.float64)
        self.velocity[env_ids] = state.velocity[env_ids].to(torch.float64)
        self.orientation[env_ids] = state.orientation_quaternion[env_ids].to(torch.float64)
        self.body_rates[env_ids] = state.angular_velocity[env_ids].to(torch.float64)
        self.mass[env_ids] = state.mass[env_ids].to(torch.float64)
        self.time[env_ids] = state.time[env_ids].to(torch.float64)

    def gravity(self, position: Tensor) -> Tensor:
        """Return the point-mass lunar gravity (N, 3) at plant-frame ``position``."""
        relative = position - self.moon_center
        return -MOON_GRAVITATIONAL_PARAMETER * relative / relative.norm(dim=1, keepdim=True) ** 3

    def _derivatives(self, position, velocity, orientation, rates, force_body, torque_body, mass):
        """Return the time derivatives of (position, velocity, orientation, body rates)."""
        acceleration = self.gravity(position) + rotate_vector_body_to_parent(orientation, force_body) / mass.unsqueeze(1)
        if self.frame is Frame.MOON_CENTERED_MOON_FIXED:
            spin = self.spin.expand_as(position)
            acceleration = acceleration - 2.0 * torch.cross(spin, velocity, dim=1) - torch.cross(spin, torch.cross(spin, position, dim=1), dim=1)
        inertia = self.reference_inertia * (mass / self.reference_mass).view(-1, 1, 1)
        angular_momentum = (inertia @ rates.unsqueeze(2)).squeeze(2)
        rate_derivative = torch.linalg.solve(inertia, (torque_body - torch.cross(rates, angular_momentum, dim=1)).unsqueeze(2)).squeeze(2)
        return velocity, acceleration, quaternion_derivative(orientation, rates), rate_derivative

    def apply_wrench(self, bus, physics_step_duration: float) -> None:
        """Integrate one PhysX substep with RK4 under the actuator and surface wrench."""
        actuators: Wrench = bus.read(MessageKey.ACTUATOR_WRENCH, self)
        surface: Wrench = bus.read(MessageKey.SURFACE_WRENCH, self)
        force = (actuators.force + surface.force).to(torch.float64)
        torque = (actuators.torque + surface.torque).to(torch.float64)
        flow = (actuators.mass_flow_rate + surface.mass_flow_rate).to(torch.float64)
        dt = float(physics_step_duration)
        mass = self.mass
        state = (self.position, self.velocity, self.orientation, self.body_rates)
        k1 = self._derivatives(*state, force, torque, mass)
        k2 = self._derivatives(*(s + 0.5 * dt * k for s, k in zip(state, k1)), force, torque, mass)
        k3 = self._derivatives(*(s + 0.5 * dt * k for s, k in zip(state, k2)), force, torque, mass)
        k4 = self._derivatives(*(s + dt * k for s, k in zip(state, k3)), force, torque, mass)
        new = [s + dt / 6.0 * (a + 2.0 * b + 2.0 * c + d) for s, a, b, c, d in zip(state, k1, k2, k3, k4)]
        self.position, self.velocity = new[0], new[1]
        self.orientation = torch.nn.functional.normalize(new[2], dim=1)
        self.body_rates = new[3]
        self.mass = torch.clamp(mass - flow * dt, min=self.minimum_mass)
        self.time = self.time + dt

    def height_above_ground(self, geometry: SurfaceGeometry | None) -> Tensor:
        """Return the footpad-plane height above the ground (N,): above the site sphere, or the surface lookup near the site."""
        radius = (self.position - self.moon_center).norm(dim=1)
        height = radius - self.site.radius_m - self.center_of_mass_height
        if self.frame is Frame.SITE and geometry is not None and geometry.height_field is not None:
            # The height map holds heights along the radius relative to the site centre, so the
            # radius above (site radius + local terrain) is the height above the ground; the
            # radius already contains the curvature of the sphere.
            world = self.position.to(torch.float32) + geometry.site_origin
            ground = geometry.ground_height_at(world[:, 0], world[:, 1]).to(torch.float64) - geometry.site_origin[:, 2].to(torch.float64)
            height = height - ground
        return height

    def step(self, bus, dt: float) -> None:
        """Write the TruthState in the plant's frame: float64 in MCI and MCMF, float32 in the SITE frame.

        Near the site float32 keeps millimetres, and the sensors and filters of the site frame
        work in float32 (the PhysX plant reports float32 too); the integration stays float64.
        """
        geometry = bus.read(MessageKey.SURFACE_GEOMETRY, self)
        count = self.number_of_environments
        dtype = torch.float32 if self.frame is Frame.SITE else torch.float64
        state = TruthState(
            frame=self.frame,
            time=self.time.to(dtype),
            position=self.position.to(dtype),
            velocity=self.velocity.to(dtype),
            orientation_quaternion=self.orientation.to(dtype),
            angular_velocity=self.body_rates.to(dtype),
            mass=self.mass.to(dtype),
            contact=torch.zeros(count, self.preset.number_of_legs, dtype=dtype, device=self.device),
            footpad_height_above_ground=self.height_above_ground(geometry).to(dtype),
        )
        bus.write(MessageKey.TRUTH_STATE, state, self)


@register_binding(SlotName.PLANT, "local_f64")
class LocalFloat64Plant(OrbitFloat64Plant):
    """``orbit_f64`` in the SITE frame (S5): east-north-up at the site, gravity toward the Moon's centre."""

    default_frame = "SITE"
