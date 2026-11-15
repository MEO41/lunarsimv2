# SPDX-License-Identifier: BSD-3-Clause
"""Lander preset: the numbers that define one lander, loaded from ``configs/landers``.

Pure Python (no Isaac). A new lander is a new YAML file, not new code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lunarsim.core.physical_constants import STANDARD_GRAVITY


class LanderPresetError(ValueError):
    """The preset file misses a number or a source."""


@dataclass(frozen=True)
class ReactionControlJet:
    """One reaction-control jet: where it sits and which way it pushes (body frame)."""

    name: str
    position: tuple[float, float, float]  # m, from the centre of mass
    force_direction: tuple[float, float, float]  # unit vector of the force on the vehicle


@dataclass(frozen=True)
class LanderPreset:
    """All lander numbers in SI units. See the YAML file for the source of each one."""

    name: str
    # mass
    mass_at_start: float
    descent_propellant_at_start: float
    unusable_descent_propellant: float
    # inertia about the centre of mass (body axes)
    inertia_about_x: float
    inertia_about_y: float
    inertia_about_z: float
    # geometry (heights above the footpad plane are measured with the struts not stroked)
    center_of_mass_height_above_footpad_plane: float
    center_of_mass_offset_x: float  # c.g. position relative to the vehicle axis, body x
    center_of_mass_offset_y: float  # c.g. position relative to the vehicle axis, body y
    descent_stage_width: float
    descent_stage_height: float
    descent_stage_bottom_above_footpad_plane: float
    ascent_stage_width: float  # along body y
    ascent_stage_depth: float  # along body x
    ascent_stage_height: float
    nozzle_exit_diameter: float
    nozzle_exit_height_above_footpad_plane: float
    # descent engine
    engine_rated_thrust: float
    engine_minimum_throttle_fraction: float
    engine_maximum_throttle_fraction: float
    engine_specific_impulse: float
    engine_thrust_lag_time_constant: float
    engine_maximum_throttle_rate: float  # fraction of rated thrust per second
    engine_off_below_fraction: float
    engine_shutdown_thrust_decay_time: float  # s, straight-line fall to zero thrust after shutdown
    # reaction control system
    jet_thrust: float
    jet_specific_impulse: float
    jet_minimum_pulse_duration: float
    jet_valve_delay: float
    jet_cluster_offset_along_body_axes: float
    horizontal_jet_offset_across_firing_axis: float
    horizontal_jet_offset_along_firing_axis: float
    jet_cluster_height_above_center_of_mass: float
    # landing gear
    number_of_legs: int
    footpad_radius_from_thrust_axis: float
    footpad_diameter: float
    footpad_depth: float
    strut_stroke: float
    strut_stiffness: float
    strut_damping_ratio: float
    leg_mass: float
    surface_probe_length: float  # m, below the lowest point of the dish, along body -z
    # touchdown limits
    maximum_vertical_touchdown_speed: float
    maximum_horizontal_touchdown_speed: float
    vertical_speed_where_horizontal_limit_starts_to_drop: float
    maximum_touchdown_tilt: float  # rad
    maximum_touchdown_tilt_rate: float  # rad/s
    # legs (1-based, in LunarSim leg order) that carry a surface probe
    legs_with_surface_probes: tuple[int, ...] = ()

    # ---- derived quantities ----

    @property
    def usable_descent_propellant_at_start(self) -> float:
        """Return the descent propellant the engine can burn, kg."""
        return self.descent_propellant_at_start - self.unusable_descent_propellant

    @property
    def mass_without_usable_propellant(self) -> float:
        """Return the vehicle mass when the usable descent propellant is gone, kg."""
        return self.mass_at_start - self.usable_descent_propellant_at_start

    @property
    def engine_minimum_thrust(self) -> float:
        """Return the lowest thrust of the running engine, N."""
        return self.engine_rated_thrust * self.engine_minimum_throttle_fraction

    @property
    def engine_maximum_thrust(self) -> float:
        """Return the highest throttleable thrust, N."""
        return self.engine_rated_thrust * self.engine_maximum_throttle_fraction

    @property
    def engine_exhaust_velocity(self) -> float:
        """Return Isp * g0 of the descent engine, m/s."""
        return self.engine_specific_impulse * STANDARD_GRAVITY

    @property
    def jet_exhaust_velocity(self) -> float:
        """Return Isp * g0 of one reaction-control jet, m/s."""
        return self.jet_specific_impulse * STANDARD_GRAVITY

    @property
    def footpad_radius(self) -> float:
        """Return the radius of one footpad, m."""
        return 0.5 * self.footpad_diameter

    @property
    def strut_damping(self) -> float:
        """Return the strut damping, N s/m: ratio * 2 * sqrt(stiffness * vehicle mass / legs)."""
        mass_per_leg = self.mass_at_start / self.number_of_legs
        return self.strut_damping_ratio * 2.0 * math.sqrt(self.strut_stiffness * mass_per_leg)

    @property
    def vehicle_axis_position(self) -> tuple[float, float]:
        """Return where the vehicle axis (thrust axis) crosses the body x-y plane, m.

        The body-frame origin is the centre of mass, which sits slightly off the vehicle axis,
        so every part of the vehicle is shifted by minus the centre-of-mass offset.
        """
        return (-self.center_of_mass_offset_x, -self.center_of_mass_offset_y)

    def height_above_center_of_mass(self, height_above_footpad_plane: float) -> float:
        """Convert a height above the footpad plane (struts not stroked) to a body-frame z, m."""
        return height_above_footpad_plane - self.center_of_mass_height_above_footpad_plane

    @property
    def leg_positions(self) -> list[tuple[float, float, float]]:
        """Return the leg-link origins in the body frame (struts not stroked), one per leg.

        Legs sit on the vehicle's +x, +y, -x, -y axes. The footpad is a dish (spherical cap,
        curved side down); the link origin is the centre of its flat top, ``footpad_depth``
        above the footpad plane, so the lowest point of the dish lies in the footpad plane.
        """
        height = self.height_above_center_of_mass(self.footpad_depth)
        axis_x, axis_y = self.vehicle_axis_position
        positions = []
        for leg_index in range(self.number_of_legs):
            azimuth = 2.0 * math.pi * leg_index / self.number_of_legs
            positions.append(
                (
                    axis_x + self.footpad_radius_from_thrust_axis * math.cos(azimuth),
                    axis_y + self.footpad_radius_from_thrust_axis * math.sin(azimuth),
                    height,
                )
            )
        return positions

    @property
    def surface_probe_tip_offset_from_leg_link(self) -> tuple[float, float, float]:
        """Return the probe tip relative to its leg-link origin, in body axes, m.

        The probe hangs straight down (body -z) from the lowest point of the dish, so its tip lies
        ``footpad_depth + surface_probe_length`` below the leg-link origin (the dish's flat top).
        """
        return (0.0, 0.0, -(self.footpad_depth + self.surface_probe_length))

    @property
    def footpad_sphere_radius(self) -> float:
        """Return the radius of the sphere whose cap forms the dish footpad, m.

        A cap of base radius a and depth h belongs to a sphere of radius (a^2 + h^2) / (2 h).
        """
        base_radius = self.footpad_radius
        return (base_radius**2 + self.footpad_depth**2) / (2.0 * self.footpad_depth)

    @property
    def descent_stage_center(self) -> tuple[float, float, float]:
        """Return the centre of the descent-stage cylinder in the body frame, m."""
        axis_x, axis_y = self.vehicle_axis_position
        middle = self.descent_stage_bottom_above_footpad_plane + 0.5 * self.descent_stage_height
        return (axis_x, axis_y, self.height_above_center_of_mass(middle))

    @property
    def ascent_stage_center(self) -> tuple[float, float, float]:
        """Return the centre of the ascent-stage box in the body frame, m. It sits on the descent stage."""
        axis_x, axis_y = self.vehicle_axis_position
        bottom = self.descent_stage_bottom_above_footpad_plane + self.descent_stage_height
        return (axis_x, axis_y, self.height_above_center_of_mass(bottom + 0.5 * self.ascent_stage_height))

    @property
    def nozzle_exit_center(self) -> tuple[float, float, float]:
        """Return the centre of the descent-engine nozzle exit in the body frame, m."""
        axis_x, axis_y = self.vehicle_axis_position
        return (axis_x, axis_y, self.height_above_center_of_mass(self.nozzle_exit_height_above_footpad_plane))

    @property
    def body_mass(self) -> float:
        """Return the mass of the root body at the start: total mass minus the legs, kg."""
        return self.mass_at_start - self.number_of_legs * self.leg_mass

    @property
    def body_center_of_mass(self) -> tuple[float, float, float]:
        """Return where the root body's own centre of mass sits in the body frame, m.

        The origin of the body frame is the centre of mass of the whole vehicle (body plus
        legs, struts extended). The legs hang below it (and follow the small offset of the
        vehicle axis), so the root body's centre of mass sits a little above and beside it.
        """
        return tuple(
            -sum(self.leg_mass * position[axis] for position in self.leg_positions) / self.body_mass for axis in range(3)
        )

    @property
    def body_inertia(self) -> tuple[float, float, float]:
        """Return the root-body inertia about its own centre of mass (body axes).

        It is chosen so that body plus legs give the preset's total inertia about the vehicle
        centre of mass. Each leg is treated as a point mass at its leg-link origin, and the
        body's offset from the vehicle centre of mass is removed with the parallel-axis rule.
        Products of inertia are neglected.
        """
        legs_about_x = legs_about_y = legs_about_z = 0.0
        for x, y, z in self.leg_positions:
            legs_about_x += self.leg_mass * (y * y + z * z)
            legs_about_y += self.leg_mass * (x * x + z * z)
            legs_about_z += self.leg_mass * (x * x + y * y)
        body_x, body_y, body_z = self.body_center_of_mass
        inertia = (
            self.inertia_about_x - legs_about_x - self.body_mass * (body_y**2 + body_z**2),
            self.inertia_about_y - legs_about_y - self.body_mass * (body_x**2 + body_z**2),
            self.inertia_about_z - legs_about_z - self.body_mass * (body_x**2 + body_y**2),
        )
        if min(inertia) <= 0.0:
            raise LanderPresetError("The legs alone have more inertia than the whole vehicle; lower the leg mass")
        return inertia

    @property
    def jets(self) -> list[ReactionControlJet]:
        """Return the reaction-control jets: four clusters of four, on the 45 degree diagonals.

        Each cluster sits ``jet_cluster_offset_along_body_axes`` from the vehicle axis along
        both body x and y. In each cluster: one jet pushes up (+z), one down (-z), and two push
        horizontally along the body x and y axes, toward the vehicle (their nozzles point away
        from it). A horizontal jet sits ``horizontal_jet_offset_along_firing_axis`` out along
        its firing axis and ``horizontal_jet_offset_across_firing_axis`` out across it.
        """
        axis_x, axis_y = self.vehicle_axis_position
        height = self.jet_cluster_height_above_center_of_mass
        along = self.horizontal_jet_offset_along_firing_axis
        across = self.horizontal_jet_offset_across_firing_axis
        jets = []
        for cluster_index in range(4):
            azimuth = math.pi / 4.0 + 0.5 * math.pi * cluster_index
            sign_x = math.copysign(1.0, math.cos(azimuth))
            sign_y = math.copysign(1.0, math.sin(azimuth))
            offset = self.jet_cluster_offset_along_body_axes
            cluster_position = (axis_x + sign_x * offset, axis_y + sign_y * offset, height)
            x_jet_position = (axis_x + sign_x * along, axis_y + sign_y * across, height)
            y_jet_position = (axis_x + sign_x * across, axis_y + sign_y * along, height)
            cluster = f"cluster_{cluster_index + 1}"
            jets.append(ReactionControlJet(f"{cluster}_push_up", cluster_position, (0.0, 0.0, 1.0)))
            jets.append(ReactionControlJet(f"{cluster}_push_down", cluster_position, (0.0, 0.0, -1.0)))
            jets.append(ReactionControlJet(f"{cluster}_push_along_x", x_jet_position, (-sign_x, 0.0, 0.0)))
            jets.append(ReactionControlJet(f"{cluster}_push_along_y", y_jet_position, (0.0, -sign_y, 0.0)))
        return jets

    @property
    def number_of_jets(self) -> int:
        """Return the number of reaction-control jets."""
        return len(self.jets)

    def jet_force_matrix(self) -> list[list[float]]:
        """Return the 3 x n_jets matrix of body-frame force per fully open jet, N."""
        return [[jet.force_direction[axis] * self.jet_thrust for jet in self.jets] for axis in range(3)]

    def jet_torque_matrix(self) -> list[list[float]]:
        """Return the 3 x n_jets matrix of body-frame torque about the centre of mass per open jet, N m."""
        columns = []
        for jet in self.jets:
            px, py, pz = jet.position
            fx, fy, fz = (component * self.jet_thrust for component in jet.force_direction)
            columns.append((py * fz - pz * fy, pz * fx - px * fz, px * fy - py * fx))
        return [[column[axis] for column in columns] for axis in range(3)]


# (section, key in the YAML file, field of LanderPreset, factor to SI)
_FIELD_MAP = (
    ("mass", "mass_at_start_kg", "mass_at_start", 1.0),
    ("mass", "descent_propellant_at_start_kg", "descent_propellant_at_start", 1.0),
    ("mass", "unusable_descent_propellant_kg", "unusable_descent_propellant", 1.0),
    ("inertia", "about_x_kg_m2", "inertia_about_x", 1.0),
    ("inertia", "about_y_kg_m2", "inertia_about_y", 1.0),
    ("inertia", "about_z_kg_m2", "inertia_about_z", 1.0),
    ("geometry", "center_of_mass_height_above_footpad_plane_m", "center_of_mass_height_above_footpad_plane", 1.0),
    ("geometry", "center_of_mass_offset_x_m", "center_of_mass_offset_x", 1.0),
    ("geometry", "center_of_mass_offset_y_m", "center_of_mass_offset_y", 1.0),
    ("geometry", "descent_stage_width_m", "descent_stage_width", 1.0),
    ("geometry", "descent_stage_height_m", "descent_stage_height", 1.0),
    ("geometry", "descent_stage_bottom_above_footpad_plane_m", "descent_stage_bottom_above_footpad_plane", 1.0),
    ("geometry", "ascent_stage_width_m", "ascent_stage_width", 1.0),
    ("geometry", "ascent_stage_depth_m", "ascent_stage_depth", 1.0),
    ("geometry", "ascent_stage_height_m", "ascent_stage_height", 1.0),
    ("geometry", "nozzle_exit_diameter_m", "nozzle_exit_diameter", 1.0),
    ("geometry", "nozzle_exit_height_above_footpad_plane_m", "nozzle_exit_height_above_footpad_plane", 1.0),
    ("descent_engine", "rated_thrust_n", "engine_rated_thrust", 1.0),
    ("descent_engine", "minimum_throttle_fraction", "engine_minimum_throttle_fraction", 1.0),
    ("descent_engine", "maximum_throttle_fraction", "engine_maximum_throttle_fraction", 1.0),
    ("descent_engine", "specific_impulse_s", "engine_specific_impulse", 1.0),
    ("descent_engine", "thrust_lag_time_constant_s", "engine_thrust_lag_time_constant", 1.0),
    ("descent_engine", "maximum_throttle_rate_per_s", "engine_maximum_throttle_rate", 1.0),
    ("descent_engine", "engine_off_below_fraction", "engine_off_below_fraction", 1.0),
    ("descent_engine", "shutdown_thrust_decay_time_s", "engine_shutdown_thrust_decay_time", 1.0),
    ("reaction_control_system", "jet_thrust_n", "jet_thrust", 1.0),
    ("reaction_control_system", "specific_impulse_s", "jet_specific_impulse", 1.0),
    ("reaction_control_system", "minimum_pulse_duration_s", "jet_minimum_pulse_duration", 1.0),
    ("reaction_control_system", "valve_delay_s", "jet_valve_delay", 1.0),
    ("reaction_control_system", "cluster_offset_along_body_axes_m", "jet_cluster_offset_along_body_axes", 1.0),
    (
        "reaction_control_system",
        "horizontal_jet_offset_across_firing_axis_m",
        "horizontal_jet_offset_across_firing_axis",
        1.0,
    ),
    (
        "reaction_control_system",
        "horizontal_jet_offset_along_firing_axis_m",
        "horizontal_jet_offset_along_firing_axis",
        1.0,
    ),
    (
        "reaction_control_system",
        "cluster_height_above_center_of_mass_m",
        "jet_cluster_height_above_center_of_mass",
        1.0,
    ),
    ("landing_gear", "number_of_legs", "number_of_legs", 1),
    ("landing_gear", "footpad_radius_from_thrust_axis_m", "footpad_radius_from_thrust_axis", 1.0),
    ("landing_gear", "footpad_diameter_m", "footpad_diameter", 1.0),
    ("landing_gear", "footpad_depth_m", "footpad_depth", 1.0),
    ("landing_gear", "strut_stroke_m", "strut_stroke", 1.0),
    ("landing_gear", "strut_stiffness_n_per_m", "strut_stiffness", 1.0),
    ("landing_gear", "strut_damping_ratio", "strut_damping_ratio", 1.0),
    ("landing_gear", "leg_mass_kg", "leg_mass", 1.0),
    ("landing_gear", "surface_probe_length_m", "surface_probe_length", 1.0),
    ("touchdown_limits", "maximum_vertical_speed_mps", "maximum_vertical_touchdown_speed", 1.0),
    ("touchdown_limits", "maximum_horizontal_speed_mps", "maximum_horizontal_touchdown_speed", 1.0),
    (
        "touchdown_limits",
        "vertical_speed_where_horizontal_limit_starts_to_drop_mps",
        "vertical_speed_where_horizontal_limit_starts_to_drop",
        1.0,
    ),
    ("touchdown_limits", "maximum_tilt_deg", "maximum_touchdown_tilt", math.pi / 180.0),
    ("touchdown_limits", "maximum_tilt_rate_deg_per_s", "maximum_touchdown_tilt_rate", math.pi / 180.0),
)


# (section, key in the YAML file, field of LanderPreset): entries whose value is a list of integers.
_LIST_FIELD_MAP = (("landing_gear", "legs_with_surface_probes", "legs_with_surface_probes"),)


def lander_preset_from_dictionary(data: dict[str, Any]) -> LanderPreset:
    """Build a LanderPreset from the parsed YAML. Every entry needs a ``value`` and a ``source``."""
    values: dict[str, Any] = {"name": str(data.get("name", "unnamed"))}
    for section, key, field_name, factor in _FIELD_MAP:
        try:
            entry = data[section][key]
        except (KeyError, TypeError) as error:
            raise LanderPresetError(f"Lander preset misses '{section}.{key}'") from error
        if not isinstance(entry, dict) or "value" not in entry or not str(entry.get("source", "")).strip():
            raise LanderPresetError(f"Lander preset entry '{section}.{key}' needs a 'value' and a 'source'")
        values[field_name] = int(entry["value"]) if isinstance(factor, int) else float(entry["value"]) * factor
    for section, key, field_name in _LIST_FIELD_MAP:
        entry = (data.get(section) or {}).get(key)
        if not isinstance(entry, dict) or "value" not in entry or not str(entry.get("source", "")).strip():
            raise LanderPresetError(f"Lander preset entry '{section}.{key}' needs a 'value' and a 'source'")
        values[field_name] = tuple(int(item) for item in entry["value"])
    preset = LanderPreset(**values)
    for leg in preset.legs_with_surface_probes:
        if not 1 <= leg <= preset.number_of_legs:
            raise LanderPresetError(f"legs_with_surface_probes names leg {leg}, but the lander has {preset.number_of_legs} legs")
    return preset


def lander_preset_sources(data: dict[str, Any]) -> dict[str, str]:
    """Return ``section.key -> source text`` for every number in the parsed YAML."""
    sources = {f"{section}.{key}": str(data[section][key]["source"]) for section, key, _, _ in _FIELD_MAP}
    sources.update({f"{section}.{key}": str(data[section][key]["source"]) for section, key, _ in _LIST_FIELD_MAP})
    return sources


def load_lander_preset(path: Path | str) -> LanderPreset:
    """Load a preset YAML file."""
    with open(path) as file:
        return lander_preset_from_dictionary(yaml.safe_load(file))
