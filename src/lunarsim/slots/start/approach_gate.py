# SPDX-License-Identifier: BSD-3-Clause
"""``start: approach_gate`` - start states at the end of the approach phase, flying toward the site (S6).

Apollo flew the approach phase from "high gate" to "low gate", where the landing phase began
(NASA TN D-6846, Bennett 1972). At low gate the LM was about 500 ft above the surface and
2,000 ft from the landing site, with 60 ft/s forward and 16 ft/s vertical speed and an attitude
of about 16 deg from the vertical ([D-6846] p. 10, Table I p. 9). Scenario S6 starts there,
so the hazard-detection LiDAR sees the landing area while there is still time and propellant to
divert.

Each quantity is drawn uniformly from its ``[low, high]`` range:

- ``downrange_m``: horizontal distance behind the target along the approach direction;
- ``crossrange_m``: sideways offset from the approach line;
- ``altitude_m``: footpad plane above the target height;
- ``forward_speed_mps``: horizontal speed toward the target (along the approach direction);
- ``vertical_speed_mps``: vertical speed (negative is down);
- ``pitch_deg``: thrust axis tilted back from the vertical, against the motion (braking);
- ``approach_azimuth_deg`` (default ``[0, 0]``): direction of flight, from +x toward +y;
- ``nav_error`` (optional): error of the first navigation estimate, as for ``terminal_box``.

The lander's heading about its thrust axis is zero. Positions are in the site frame (the target
at the origin); the plant puts the lander above the ground of its own site.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.core.binding import SlotName
from lunarsim.core.registry import register_binding
from lunarsim.core.rotation_math import quaternion_from_axis_angle
from lunarsim.slots.start.terminal_box import TerminalBoxStart, sample_uniform


def braking_attitude(azimuth: Tensor, pitch: Tensor) -> Tensor:
    """Return quaternions (N, 4) whose body +z axis is tilted by ``pitch`` against the flight direction ``azimuth``.

    Rotating +z about ``a = z x w`` (``w`` the unit vector against the motion) by ``pitch`` gives
    ``cos(pitch) z + sin(pitch) w``: the thrust pushes back and up.
    """
    against_x, against_y = -torch.cos(azimuth), -torch.sin(azimuth)
    axis = torch.stack((-against_y, against_x, torch.zeros_like(azimuth)), dim=1)
    return quaternion_from_axis_angle(axis, pitch)


@register_binding(SlotName.START, "approach_gate")
class ApproachGateStart(TerminalBoxStart):
    """Samples low-gate start states (see the module text for the parameters)."""

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Draw a new approach start for the given environments, using only ``gen``."""
        if self.fixed_start_states is not None:
            super().reset(env_ids, gen)
            return
        count, parameters = len(env_ids), self.parameters

        def draw(name: str, default=None) -> Tensor:
            """Draw one uniform sample per environment from the range named ``name``."""
            low, high = parameters.get(name, default)
            return sample_uniform(float(low), float(high), count, gen, self.device)

        downrange = draw("downrange_m")
        crossrange = draw("crossrange_m", [0.0, 0.0])
        altitude = draw("altitude_m")
        forward_speed = draw("forward_speed_mps")
        vertical_speed = draw("vertical_speed_mps")
        pitch = draw("pitch_deg") * (math.pi / 180.0)
        azimuth = draw("approach_azimuth_deg", [0.0, 0.0]) * (math.pi / 180.0)
        along_x, along_y = torch.cos(azimuth), torch.sin(azimuth)
        # Behind the target along the flight direction, plus the sideways offset (left of the line).
        position_x = -downrange * along_x - crossrange * along_y
        position_y = -downrange * along_y + crossrange * along_x
        self.position[env_ids] = torch.stack((position_x, position_y, altitude + self.center_of_mass_height), dim=1)
        self.velocity[env_ids] = torch.stack((forward_speed * along_x, forward_speed * along_y, vertical_speed), dim=1)
        self.orientation[env_ids] = braking_attitude(azimuth, pitch)
        self.angular_velocity[env_ids] = 0.0
        self.navigation_error.reset(env_ids, gen)
