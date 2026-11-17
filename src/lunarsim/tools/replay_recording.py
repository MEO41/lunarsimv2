# SPDX-License-Identifier: BSD-3-Clause
"""Record what a run did, frame by frame, so chosen episodes can be rendered later.

The recorder runs inside the simulation process (no camera). It keeps, for every environment
and until its first episode ends, the world pose, strut strokes, speeds and height above ground,
and optionally the footpad contacts: the world position of each leg, the contact force vector
from each leg's contact sensor, and the gap between each dish footpad and the ground.

The footpad gap is the smallest vertical distance from the dish's lower surface (the points of
its mesh) to the ground lookup. A footpad that carries load must have a gap of about zero (PhysX
keeps a contact offset of 2 cm); one that carries none must have a positive gap. Comparing the
two is the check that the collision surface and the contact sensors agree.

With ``grain_landers`` (a list of environment indices; needs a surface with the L2 plume grains,
``surface.grains``) it also keeps, for those landers only, every live grain parcel per frame: its
offset from the lander's body position, its height above the ground and its size bin; and, for
every lander, the plume readings of both levels (nozzle height, L1 dust density and eroded mass,
the grains' dust density, optical depth, eroded mass, densest cloud cell and parcel count).

The recorder reads TruthState, so it declares ``reads_truth`` like a metrics binding. Nothing in
it flows back to the agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy
import torch

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey
from lunarsim.core.rotation_math import rotate_vector_body_to_parent, tilt_angle_from_vertical

BASE_COLUMNS = (
    "position_x", "position_y", "position_z",  # world, m (centre of mass)
    "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z",
    "strut_1", "strut_2", "strut_3", "strut_4",  # m
    "speed_vertical", "speed_horizontal",  # m/s
    "height_above_ground", "tilt_deg", "time", "body_force",
)
NUMBER_OF_LEGS = 4
GRAIN_COLUMNS = (
    "nozzle_height",  # m above the ground under the nozzle
    "l1_dust_density", "l1_eroded_mass",  # 0..1; kg since the episode start (the dust metric's ejecta mass)
    "grain_dust_density", "grain_optical_depth", "grain_eroded_mass",  # 0..1; -; kg
    "grain_peak_number_density", "grain_particles",  # grains per m^3 in the densest cloud cell; live parcels
)


def contact_columns() -> tuple[str, ...]:
    """Return the per-leg contact column names: position, force vector, gap."""
    names = []
    for leg in range(1, NUMBER_OF_LEGS + 1):
        names += [f"leg_{leg}_x", f"leg_{leg}_y", f"leg_{leg}_z"]
        names += [f"leg_{leg}_force_x", f"leg_{leg}_force_y", f"leg_{leg}_force_z"]
        names += [f"leg_{leg}_gap"]
    return tuple(names)


class ReplayRecorder:
    """Keeps the first episode of every environment, every ``every_n_steps`` control steps."""

    slot = SlotName.METRICS
    requires = frozenset({MessageKey.TRUTH_STATE, MessageKey.SURFACE_GEOMETRY})
    provides = frozenset()
    reads_truth = True

    def __init__(self, environment: Any, every_n_steps: int = 2, include_contacts: bool = False, grain_landers: list[int] | None = None) -> None:
        """Prepare the frame list; with ``include_contacts`` also the footpad sample points; with ``grain_landers`` the grain buffers."""
        from lunarsim.assets.lander_builder import footpad_dish_mesh

        self.environment = environment
        self.plant = environment.bindings[SlotName.PLANT]
        self.every_n_steps = every_n_steps
        self.include_contacts = include_contacts
        self.grain_landers = list(grain_landers) if grain_landers else []
        if self.grain_landers:
            self.grain_model = getattr(environment.bindings[SlotName.SURFACE], "grain_model", None)
            if self.grain_model is None:
                raise ValueError("grain_landers needs a surface with the L2 plume grains (surface.grains.enabled)")
            self.requires = ReplayRecorder.requires | frozenset({MessageKey.REGOLITH_STATE})
            self.dust_metric = environment.bindings[SlotName.METRICS]
            self.grain_offsets: list[numpy.ndarray] = []  # per frame (G, P, 4): offset x, y, z from the body, height above ground; NaN if dead
            self.grain_bins: list[numpy.ndarray] = []  # per frame (G, P) uint8
        self.columns = BASE_COLUMNS + (contact_columns() if include_contacts else ()) + (GRAIN_COLUMNS if self.grain_landers else ())
        count, device = environment.num_envs, environment.device
        self.still_recording = torch.ones(count, dtype=torch.bool, device=device)
        self.previous_time = torch.full((count,), -1.0, device=device)
        self.frames: list[numpy.ndarray] = []
        self.valid: list[numpy.ndarray] = []
        self.site_origin = None
        if include_contacts:
            preset = self.plant.preset
            points, _, _ = footpad_dish_mesh(preset)
            # Lower surface of the dish: every mesh point except the centre of the flat top.
            self.dish_points = torch.tensor(points[:-1], dtype=torch.float32, device=device)
            self.leg_ids = [self.plant.asset.find_bodies(f"leg_{index + 1}")[0][0] for index in range(NUMBER_OF_LEGS)]

    def footpad_gaps(self, orientation: torch.Tensor, leg_positions: torch.Tensor, geometry) -> torch.Tensor:
        """Return the smallest vertical gap of each dish to the ground, (N, legs), m."""
        count, points = orientation.shape[0], self.dish_points.shape[0]
        rotation = orientation.unsqueeze(1).unsqueeze(1).expand(count, NUMBER_OF_LEGS, points, 4)
        offsets = rotate_vector_body_to_parent(rotation, self.dish_points.expand(count, NUMBER_OF_LEGS, points, 3))
        world = leg_positions.unsqueeze(2) + offsets
        ground = geometry.ground_height_at(world[..., 0], world[..., 1])
        return (world[..., 2] - ground).min(dim=2).values

    def record(self, step_index: int, command: Any) -> None:
        """Store one frame of every environment that is still in its first episode."""
        truth = self.environment.bus.read(MessageKey.TRUTH_STATE, self)
        geometry = self.environment.bus.read(MessageKey.SURFACE_GEOMETRY, self)
        if self.site_origin is None:
            self.site_origin = geometry.site_origin.clone()
        # The episode time jumps back when an environment resets: its first episode is over.
        self.still_recording &= truth.time > self.previous_time
        self.previous_time = truth.time.clone()
        if step_index % self.every_n_steps != 0:
            return
        parts = [
            truth.position + self.site_origin,
            truth.orientation_quaternion,
            self.plant.strut_compression(),
            truth.velocity[:, 2:3],
            truth.velocity[:, :2].norm(dim=1, keepdim=True),
            truth.footpad_height_above_ground.unsqueeze(1),
            torch.rad2deg(tilt_angle_from_vertical(truth.orientation_quaternion)).unsqueeze(1),
            truth.time.unsqueeze(1),
            truth.body_contact.unsqueeze(1),
        ]
        if self.include_contacts:
            leg_positions = self.plant.asset.data.body_link_pos_w.torch[:, self.leg_ids]  # (N, 4, 3)
            forces = self.plant.contact_sensor.data.net_forces_w.torch[:, self.plant.contact_body_ids]  # (N, 4, 3)
            gaps = self.footpad_gaps(truth.orientation_quaternion, leg_positions, geometry)
            per_leg = torch.cat((leg_positions, forces, gaps.unsqueeze(2)), dim=2)  # (N, 4, 7)
            parts.append(per_leg.reshape(per_leg.shape[0], -1))
        if self.grain_landers:
            parts.append(self.grain_readings())
            self.record_grain_parcels(geometry)
        frame = torch.cat(parts, dim=1)
        self.frames.append(frame.cpu().numpy().astype(numpy.float32))
        self.valid.append(self.still_recording.cpu().numpy())

    def grain_readings(self) -> torch.Tensor:
        """Return the plume readings of both levels for every lander, (N, len(GRAIN_COLUMNS))."""
        regolith = self.environment.bus.read(MessageKey.REGOLITH_STATE, self)
        grains = regolith.grains
        # In shadow mode the published dust is L1's; the dust metric integrates L1's ejecta mass.
        values = (
            regolith.nozzle_height, regolith.dust_density, self.dust_metric.ejecta_mass,
            grains.dust_density, grains.optical_depth, grains.eroded_mass, grains.peak_number_density, grains.particles,
        )
        return torch.stack([value.float() for value in values], dim=1)

    def record_grain_parcels(self, geometry) -> None:
        """Store every live parcel of the grain landers: offset from the body, height above ground, size bin."""
        model, landers = self.grain_model, self.grain_landers
        body_position, _ = self.plant.body_pose()
        position = model.position[landers]  # (G, P, 3), the frame the plant reports
        above = position[..., 2] - geometry.ground_height_at(position[..., 0], position[..., 1])
        offsets = torch.cat((position - body_position[landers].unsqueeze(1), above.unsqueeze(2)), dim=2)
        offsets = torch.where(model.alive[landers].unsqueeze(2), offsets, float("nan"))
        self.grain_offsets.append(offsets.cpu().numpy().astype(numpy.float32))
        self.grain_bins.append(model.bin[landers].to(torch.uint8).cpu().numpy())

    def grain_arrays(self, environment_indices: list[int]) -> dict[str, numpy.ndarray]:
        """Return the grain arrays of the chosen environments (each must be a grain lander), in their order."""
        if not self.grain_landers:
            return {}
        missing = [index for index in environment_indices if index not in self.grain_landers]
        if missing:
            raise ValueError(f"environments {missing} were not recorded with grains (grain_landers={self.grain_landers})")
        order = [self.grain_landers.index(index) for index in environment_indices]
        model = self.grain_model
        return {
            "grain_offsets": numpy.stack(self.grain_offsets)[:, order],
            "grain_bins": numpy.stack(self.grain_bins)[:, order],
            "grain_bin_diameters_m": model.diameters.cpu().numpy(),
            "grain_bin_mass_fractions": model.mass_fractions.cpu().numpy(),
            "grain_capacity": numpy.array(model.capacity),
            "grain_line_of_sight_depression_deg": numpy.array(numpy.degrees(model.line_of_sight_depression)),
            "grain_cloud_grid_m": numpy.array([model.grid_radius, model.grid_height]),
        }

    def save(
        self,
        path: Path | str,
        environment_indices: list[int],
        scenario_name: str,
        scenario_overrides: list[str],
        clip_titles: list[str],
        clip_details: list[str],
        frames_per_second: float,
        notes: str = "",
        clip_start_times: list[float] | None = None,
    ) -> Path:
        """Write the chosen environments to an ``.npz`` file that ``scripts/render_replay.py`` reads."""
        frames = numpy.stack(self.frames)[:, environment_indices]
        valid = numpy.stack(self.valid)[:, environment_indices]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        numpy.savez_compressed(
            path,
            frames=frames,
            valid=valid,
            columns=numpy.array(self.columns),
            environment_index=numpy.array(environment_indices),
            site_origin=self.site_origin.cpu().numpy()[environment_indices],
            scenario_name=numpy.array(scenario_name),
            scenario_overrides=numpy.array(scenario_overrides if scenario_overrides else [""]),
            clip_titles=numpy.array(clip_titles),
            clip_details=numpy.array(clip_details),
            clip_start_times=numpy.array(clip_start_times or [0.0] * len(environment_indices)),
            frames_per_second=frames_per_second,
            notes=numpy.array(notes),
            **self.grain_arrays(environment_indices),
        )
        return path
