# SPDX-License-Identifier: BSD-3-Clause
"""Record one lander's trajectory through every phase of a mission, for the mission video (S7).

Passed as ``after_step`` to ``collect_episodes`` (``scripts/run_mission.py record=<lander>``). Each
control step it keeps the chosen lander's state until its first episode ends (the phase gate or
touchdown): episode time, position and velocity in the SITE frame (east, north, up of the landing
site) and in MCMF (Moon-fixed, for the orbit picture), attitude in the SITE frame, throttle,
mass and height above the site's sphere (or above the ground in the PhysX phase). It reads
TruthState like a metrics binding; nothing flows back to the agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy
import torch

from lunarsim.core.binding import SlotName
from lunarsim.core.frames import SiteLocation, convert_truth_state
from lunarsim.core.message_types import Frame, MessageKey

COLUMNS = (
    "time",
    "site_x", "site_y", "site_z", "site_velocity_x", "site_velocity_y", "site_velocity_z",
    "mcmf_x", "mcmf_y", "mcmf_z",
    "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z",
    "throttle", "mass", "altitude",
)


class MissionTrajectoryRecorder:
    """Keeps one lander's first episode, every ``every_n_steps`` control steps (see the module text)."""

    slot = SlotName.METRICS
    requires = frozenset({MessageKey.TRUTH_STATE})
    provides = frozenset()
    reads_truth = True

    def __init__(self, environment: Any, lander_index: int, site: SiteLocation, every_n_steps: int = 1) -> None:
        """Prepare the row list for lander ``lander_index``."""
        self.environment = environment
        self.lander_index = int(lander_index)
        self.site = site
        self.every_n_steps = int(every_n_steps)
        self.rows: list[list[float]] = []
        self.previous_time = -1.0
        self.recording = True

    def record(self, step_index: int, command: Any) -> None:
        """Store one row while the lander is still in its first episode."""
        if not self.recording:
            return
        truth = self.environment.bus.read(MessageKey.TRUTH_STATE, self)
        index = self.lander_index
        time = float(truth.time[index])
        if time < self.previous_time:  # the episode ended and the lander was reset
            self.recording = False
            return
        self.previous_time = time
        if step_index % self.every_n_steps != 0:
            return
        one = truth.__class__(**{name: getattr(truth, name)[index : index + 1] if isinstance(getattr(truth, name), torch.Tensor) else getattr(truth, name) for name in truth.__dataclass_fields__})
        site_state = convert_truth_state(one, Frame.SITE, self.site)
        fixed = convert_truth_state(one, Frame.MOON_CENTERED_MOON_FIXED, self.site)
        if truth.frame is Frame.SITE and getattr(truth, "footpad_height_above_ground", None) is not None:
            altitude = float(truth.footpad_height_above_ground[index])
        else:
            altitude = float(fixed.position[0].norm()) - self.site.radius_m
        self.rows.append(
            [time]
            + site_state.position[0].double().tolist() + site_state.velocity[0].double().tolist()
            + fixed.position[0].double().tolist()
            + site_state.orientation_quaternion[0].double().tolist()
            + [float(command.throttle[index]), float(truth.mass[index]), altitude]
        )

    def save(self, path: Path | str, phase: str) -> Path:
        """Write the rows to ``path`` (.npz with ``rows``, ``columns``, ``phase``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        numpy.savez_compressed(path, rows=numpy.asarray(self.rows, dtype=numpy.float64), columns=numpy.array(COLUMNS), phase=numpy.array(phase))
        return path
