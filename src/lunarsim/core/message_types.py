# SPDX-License-Identifier: BSD-3-Clause
"""Typed messages that slots exchange.

Every tensor is batched: the first dimension is the environment count ``N``.
The design document uses short field names; this table maps them to the names used in code:

==========  ===========================
Design      Code
==========  ===========================
``t``       ``time``
``r``       ``position``
``v``       ``velocity``
``q``       ``orientation_quaternion`` (w, x, y, z; body frame to the state's frame)
``w``       ``angular_velocity`` (body rates, in the body frame)
``cov``     ``covariance``
``rcs``     ``reaction_control_jets``
==========  ===========================
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, fields
from typing import Any, Protocol

import torch
from torch import Tensor


class Frame(enum.Enum):
    """Reference frame of a :class:`TruthState`."""

    MOON_CENTERED_INERTIAL = "MCI"
    MOON_CENTERED_MOON_FIXED = "MCMF"
    SITE = "SITE"  # origin on the surface at the landing target, z axis up, fixed to the Moon


class MessageKey:
    """Names of the messages on the :class:`~lunarsim.core.message_bus.MessageBus`."""

    START_STATE = "start_state"  # TruthState sampled by the start slot, used by the plant at reset
    # TruthState-shaped first navigation estimate (start state plus a sampled error), for the filter
    INITIAL_NAVIGATION_STATE = "initial_navigation_state"
    SURFACE_GEOMETRY = "surface_geometry"  # SurfaceGeometry: site origins and ground heights (truth side)
    TRUTH_STATE = "truth_state"
    NAVIGATION_ESTIMATE = "navigation_estimate"
    OBSERVATION = "observation"
    COMMAND = "command"
    ACTUATOR_WRENCH = "actuator_wrench"
    SURFACE_WRENCH = "surface_wrench"
    EPISODE_EVENTS = "episode_events"
    # Truth-side description of the injected faults (for metrics only; navigation never reads it)
    FAULT_STATUS = "fault_status"
    MEASUREMENT_PREFIX = "measurement."  # followed by the sensor name


def _first_dimension_of(message: object) -> int:
    """Return the shared batch size of all tensor fields of a message, or raise if they differ."""
    batch_sizes = {
        getattr(message, field.name).shape[0]
        for field in fields(message)
        if isinstance(getattr(message, field.name), Tensor)
    }
    if len(batch_sizes) != 1:
        raise ValueError(f"{type(message).__name__}: tensor fields have different batch sizes {sorted(batch_sizes)}")
    return batch_sizes.pop()


@dataclass
class TruthState:
    """The true state of the lander. Only truth-side bindings may read it."""

    frame: Frame
    time: Tensor  # (N,) seconds since the episode start
    position: Tensor  # (N, 3) m; float64 in orbit, float32 at the site
    velocity: Tensor  # (N, 3) m/s, expressed in ``frame``
    orientation_quaternion: Tensor  # (N, 4) w, x, y, z
    angular_velocity: Tensor  # (N, 3) rad/s, body rates in the body frame
    mass: Tensor  # (N,) kg
    contact: Tensor  # (N, n_legs) leg contact force magnitudes, N
    # Height of the footpad plane (struts not stroked) above the ground straight below the
    # centre of mass, m. NaN where unknown (a sampled start state).
    footpad_height_above_ground: Tensor | None = None  # (N,)
    body_contact: Tensor | None = None  # (N,) contact force magnitude on the vehicle body (stages), N
    # 1.0 while the tip of at least one surface probe is at or below the ground (the LM "contact
    # light"), else 0.0.
    surface_probe_contact: Tensor | None = None  # (N,)

    def __post_init__(self) -> None:
        """Check the shapes of all fields and fill the optional ones."""
        if not isinstance(self.frame, Frame):
            raise TypeError("TruthState.frame must be a Frame")
        if self.footpad_height_above_ground is None:
            self.footpad_height_above_ground = torch.full_like(self.time, float("nan"), dtype=self.position.dtype)
        if self.body_contact is None:
            self.body_contact = torch.zeros_like(self.time, dtype=self.position.dtype)
        if self.surface_probe_contact is None:
            self.surface_probe_contact = torch.zeros_like(self.time, dtype=self.position.dtype)
        number_of_environments = _first_dimension_of(self)
        expected_shapes = {
            "time": (number_of_environments,),
            "position": (number_of_environments, 3),
            "velocity": (number_of_environments, 3),
            "orientation_quaternion": (number_of_environments, 4),
            "angular_velocity": (number_of_environments, 3),
            "mass": (number_of_environments,),
            "footpad_height_above_ground": (number_of_environments,),
            "body_contact": (number_of_environments,),
            "surface_probe_contact": (number_of_environments,),
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"TruthState.{name} has shape {tuple(getattr(self, name).shape)}, expected {shape}")
        if self.contact.dim() != 2:
            raise ValueError("TruthState.contact must have shape (N, n_legs)")

    @property
    def number_of_environments(self) -> int:
        """Return the batch size ``N``."""
        return self.position.shape[0]


@dataclass
class Measurement:
    """One sensor output."""

    value: Tensor  # (N, d)
    valid: Tensor  # (N,) bool, False when a fault drops it
    stamp: Tensor  # (N,) capture time (not arrival time)
    covariance: Tensor  # (N, d, d) or (N, d)

    def __post_init__(self) -> None:
        """Check the shapes of all fields."""
        number_of_environments = _first_dimension_of(self)
        if self.value.dim() != 2:
            raise ValueError("Measurement.value must have shape (N, d)")
        if self.valid.dtype != torch.bool or tuple(self.valid.shape) != (number_of_environments,):
            raise ValueError("Measurement.valid must be a bool tensor of shape (N,)")
        if tuple(self.stamp.shape) != (number_of_environments,):
            raise ValueError("Measurement.stamp must have shape (N,)")
        dimension = self.value.shape[1]
        if tuple(self.covariance.shape[1:]) not in ((dimension,), (dimension, dimension)):
            raise ValueError("Measurement.covariance must have shape (N, d) or (N, d, d)")


@dataclass
class NavEstimate:
    """Output of the navigation slot."""

    mean: Tensor  # (N, state_dimension), layout in NavigationStateLayout
    covariance: Tensor  # (N, state_dimension) or (N, state_dimension, state_dimension)
    health: Tensor  # (N, n_sensors) fault flags; (N, 0) when there are no sensors

    def __post_init__(self) -> None:
        """Check the shapes of all fields."""
        _first_dimension_of(self)
        if self.mean.dim() != 2 or self.health.dim() != 2:
            raise ValueError("NavEstimate.mean and NavEstimate.health must be 2-dimensional")


class NavigationStateLayout:
    """Column layout of ``NavEstimate.mean`` and of the policy observation built from it."""

    POSITION = slice(0, 3)
    VELOCITY = slice(3, 6)
    ORIENTATION_QUATERNION = slice(6, 10)
    ANGULAR_VELOCITY = slice(10, 13)
    MASS = slice(13, 14)
    # Height of the footpad plane above the ground straight below the lander, m. On flat ground
    # it equals position z minus the centre-of-mass height above the footpads.
    HEIGHT_ABOVE_GROUND = slice(14, 15)
    # 1.0 while a surface probe touches the ground (the LM contact light), else 0.0.
    SURFACE_PROBE_CONTACT = slice(15, 16)
    DIMENSION = 16


@dataclass
class Observation:
    """What an agent sees. Built by the environment from ``NavEstimate``, never from ``TruthState``."""

    policy: Tensor  # (N, observation_dimension)

    def __post_init__(self) -> None:
        """Check the shape."""
        if self.policy.dim() != 2:
            raise ValueError("Observation.policy must have shape (N, observation_dimension)")


@dataclass
class Command:
    """Output of the agent slot."""

    throttle: Tensor  # (N,) 0..1
    gimbal: Tensor  # (N, 2) rad
    reaction_control_jets: Tensor  # (N, n_jets) on/off or duty, 0..1

    def __post_init__(self) -> None:
        """Check the shapes of all fields."""
        number_of_environments = _first_dimension_of(self)
        if tuple(self.throttle.shape) != (number_of_environments,):
            raise ValueError("Command.throttle must have shape (N,)")
        if tuple(self.gimbal.shape) != (number_of_environments, 2):
            raise ValueError("Command.gimbal must have shape (N, 2)")
        if self.reaction_control_jets.dim() != 2:
            raise ValueError("Command.reaction_control_jets must have shape (N, n_jets)")

    @staticmethod
    def action_dimension(number_of_jets: int) -> int:
        """Return the length of the flat action vector: throttle + 2 gimbal angles + jets."""
        return 1 + 2 + number_of_jets

    @classmethod
    def from_action_tensor(cls, actions: Tensor) -> Command:
        """Split a flat action tensor ``(N, 3 + n_jets)`` into a Command, clamping to valid ranges.

        Layout: column 0 throttle, columns 1-2 gimbal, the rest are the jets.
        """
        return cls(
            throttle=actions[:, 0].clamp(0.0, 1.0),
            gimbal=actions[:, 1:3],
            reaction_control_jets=actions[:, 3:].clamp(0.0, 1.0),
        )

    def to_action_tensor(self) -> Tensor:
        """Pack the command into the flat action tensor that the environment accepts."""
        return torch.cat((self.throttle.unsqueeze(1), self.gimbal, self.reaction_control_jets), dim=1)


@dataclass
class Wrench:
    """Force and torque on the lander body, in the body frame, about the centre of mass.

    ``mass_flow_rate`` is an extension of the design listing: it carries the propellant flow
    that belongs to the force, so the plant can lower the vehicle mass. It is zero for wrenches
    that burn no propellant (for example surface forces).
    """

    force: Tensor  # (N, 3) N
    torque: Tensor  # (N, 3) N m
    mass_flow_rate: Tensor | None = None  # (N,) kg/s, positive when mass leaves the vehicle

    def __post_init__(self) -> None:
        """Check the shapes and fill a zero mass flow if none is given."""
        if self.force.dim() != 2 or self.force.shape[1] != 3 or self.torque.shape != self.force.shape:
            raise ValueError("Wrench.force and Wrench.torque must both have shape (N, 3)")
        if self.mass_flow_rate is None:
            self.mass_flow_rate = torch.zeros(self.force.shape[0], dtype=self.force.dtype, device=self.force.device)
        elif tuple(self.mass_flow_rate.shape) != (self.force.shape[0],):
            raise ValueError("Wrench.mass_flow_rate must have shape (N,)")

    @classmethod
    def zeros(cls, number_of_environments: int, device: torch.device | str = "cpu") -> Wrench:
        """Return a wrench with no force, no torque and no mass flow."""
        return cls(
            force=torch.zeros(number_of_environments, 3, device=device),
            torque=torch.zeros(number_of_environments, 3, device=device),
        )


class GroundHeightLookup(Protocol):
    """Anything that returns ground heights for world positions (for example a TerrainHeightField)."""

    def height_at(self, x: Tensor, y: Tensor) -> Tensor:
        """Return the ground height at world (x, y)."""
        ...

    def gradient_at(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Return the ground slope (dz/dx, dz/dy) at world (x, y)."""
        ...


@dataclass
class SurfaceGeometry:
    """Output of the surface slot that truth-side slots use: where each site is, how high the ground is.

    ``site_origin`` is the world position of the origin of every environment's SITE frame (the
    landing target on the ground). ``height_field`` gives the ground height at any world point;
    None means flat ground at world z = 0.
    """

    site_origin: Tensor  # (N, 3) m, world frame
    height_field: Any | None = None  # GroundHeightLookup

    def __post_init__(self) -> None:
        """Check the shape of the site origins."""
        if self.site_origin.dim() != 2 or self.site_origin.shape[1] != 3:
            raise ValueError("SurfaceGeometry.site_origin must have shape (N, 3)")

    def ground_height_at(self, x: Tensor, y: Tensor) -> Tensor:
        """Return the ground height at world (x, y)."""
        if self.height_field is None:
            return torch.zeros_like(x)
        return self.height_field.height_at(x, y)

    def ground_gradient_at(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Return the ground slope (dz/dx, dz/dy) at world (x, y)."""
        if self.height_field is None:
            return torch.zeros_like(x), torch.zeros_like(x)
        return self.height_field.gradient_at(x, y)


@dataclass
class EpisodeEvents:
    """Output of the metrics slot: which environments ended, and why.

    The optional fields let a reward read the judgement of the metrics slot (milestone 1b)
    without judging the landing a second time. A metrics binding that does not judge leaves
    them empty.
    """

    terminated: Tensor  # (N,) bool, the episode ended by an event (touchdown, tip-over, ...)
    truncated: Tensor  # (N,) bool, the episode ended by the time limit
    # Outcome code of environments whose episode ended in this step, -1 for the others.
    outcome_code: Tensor | None = None  # (N,) long
    # True in the control step in which the first contact was seen.
    first_contact: Tensor | None = None  # (N,) bool
    # Approach values of the first contact (from the step before it), in SI units and radians:
    # vertical speed (negative down), horizontal speed, tilt, tilt rate.
    touchdown_values: dict[str, Tensor] | None = None
