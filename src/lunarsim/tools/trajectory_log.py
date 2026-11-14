# SPDX-License-Identifier: BSD-3-Clause
"""Trajectory log: the state of a few environments at every control step, for plots.

The log reads TruthState, so it declares ``reads_truth`` like a metrics binding. It is a
recording path only; nothing in it flows back to the agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas
import torch

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey
from lunarsim.core.rotation_math import tilt_angle_from_vertical
from lunarsim.tools.episode_records import TRAJECTORIES_FILE_NAME


class TrajectoryLog:
    """Records the first episode of the first ``number_of_environments`` environments."""

    slot = SlotName.METRICS
    requires = frozenset({MessageKey.TRUTH_STATE})
    provides = frozenset()
    reads_truth = True

    def __init__(self, environment: Any, number_of_environments: int = 16) -> None:
        """Remember which environments to record."""
        self.environment = environment
        self.count = min(number_of_environments, environment.num_envs)
        self.center_of_mass_height = environment.binding_context.shared_data[
            "lander_preset"
        ].center_of_mass_height_above_footpad_plane
        self.still_recording = torch.ones(self.count, dtype=torch.bool, device=environment.device)
        self.previous_time = torch.full((self.count,), -1.0, device=environment.device)
        self._rows: list[dict[str, torch.Tensor]] = []

    def record(self, step_index: int, command: Any) -> None:
        """Store one row per recorded environment. Call after every control step."""
        if not bool(self.still_recording.any()):
            return
        truth = self.environment.bus.read(MessageKey.TRUTH_STATE, self)
        selection = slice(0, self.count)
        time = truth.time[selection]
        # The episode time jumps back to zero when an environment resets: stop recording it then.
        self.still_recording &= time > self.previous_time
        self.previous_time = time.clone()
        keep = self.still_recording
        row = {
            "environment_index": torch.arange(self.count, device=time.device),
            "time_s": time,
            "position_x_m": truth.position[selection, 0],
            "position_y_m": truth.position[selection, 1],
            "altitude_m": truth.position[selection, 2] - self.center_of_mass_height,
            "velocity_x_mps": truth.velocity[selection, 0],
            "velocity_y_mps": truth.velocity[selection, 1],
            "velocity_z_mps": truth.velocity[selection, 2],
            "tilt_deg": torch.rad2deg(tilt_angle_from_vertical(truth.orientation_quaternion[selection])),
            "mass_kg": truth.mass[selection],
            "leg_force_total_n": truth.contact[selection].sum(dim=1),
            "throttle_command": command.throttle[selection],
            "jets_commanded_open": command.reaction_control_jets[selection].sum(dim=1),
        }
        self._rows.append({name: values[keep].cpu() for name, values in row.items()})

    def to_frame(self) -> pandas.DataFrame:
        """Return all recorded rows as one table."""
        if not self._rows:
            return pandas.DataFrame()
        return pandas.DataFrame({name: torch.cat([row[name] for row in self._rows]).numpy() for name in self._rows[0]})

    def write(self, run_directory: Path | str) -> Path:
        """Write ``trajectories.parquet`` into the run directory and return its path."""
        path = Path(run_directory) / TRAJECTORIES_FILE_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_parquet(path, index=False)
        return path
