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

    def __init__(self, environment: Any, every_n_steps: int = 2, include_contacts: bool = False) -> None:
        """Prepare the frame list; with ``include_contacts`` also the footpad sample points."""
        from lunarsim.assets.lander_builder import footpad_dish_mesh

        self.environment = environment
        self.plant = environment.bindings[SlotName.PLANT]
        self.every_n_steps = every_n_steps
        self.include_contacts = include_contacts
        self.columns = BASE_COLUMNS + (contact_columns() if include_contacts else ())
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
        frame = torch.cat(parts, dim=1)
        self.frames.append(frame.cpu().numpy().astype(numpy.float32))
        self.valid.append(self.still_recording.cpu().numpy())

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
        )
        return path
