# SPDX-License-Identifier: BSD-3-Clause
"""Mapping between a learned policy and the lander environment (milestone 1b task 3).

Both directions live here, outside the environment, so training and evaluation use the same
mapping and the environment code does not change.

**Actions.** Every number is meant to lie in [-1, 1] (values outside are clipped). Number 0 is
always the throttle, piecewise linear: -1 gives 0, 0 gives the hover throttle at the start mass,
+1 gives ``maximum_throttle_fraction``. At -1 the command is 0, which shuts the engine down
(below ``engine_off_below_fraction``), so the policy decides its own engine cut-off. Between the
off point and the minimum throttle the engine runs at minimum. Centring on hover lets
exploration start near a hover; a first trial centred on 0.325 learned to climb away. The
rest depends on ``action_mode``:

- ``attitude_commands`` (default, 3 numbers): numbers 1 and 2 tilt the wanted thrust direction
  toward site +x and +y, up to ``maximum_tilt_deg`` (15 deg, the ZEM/ZEV guided tilt limit). The
  classical attitude controller (the one ZEM/ZEV uses, zero heading change) fires the jets to
  reach that attitude. The policy learns guidance; attitude control is shared with ZEM/ZEV.
- ``axis_commands`` (4 numbers): numbers 1 to 3 are turn commands about body x, y, z, turned
  into jet duties by the ``JetCoupleTable`` (a pure couple per axis, duty = |command|). The
  policy then also learns attitude control; a first trial (300 iterations, 1,024 envs) did not
  learn to hold the attitude.

The gimbal command stays zero (the gimbal is not modelled).

**Observations.** Each value of the 16-value layout is divided by a fixed typical size, so all
inputs are of order one before rsl_rl's running normalisation: position 20 m, velocity 5 m/s,
quaternion 1, body rate 0.5 rad/s, mass as (m - 7000 kg) / 300 kg, height 20 m, contact light 1.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.agents.classical.attitude_controller import AttitudeController, quaternion_for_thrust_direction
from lunarsim.agents.classical.jet_selection import JetCoupleTable
from lunarsim.assets.lander_preset import LanderPreset
from lunarsim.core.message_types import Command, NavigationStateLayout
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY

ACTION_DIMENSION_BY_MODE = {"attitude_commands": 3, "axis_commands": 4}
MASS_OFFSET_KG = 7000.0
MASS_SCALE_KG = 300.0
MAXIMUM_DATA_AGE_S = 2.0  # model choice: a sensor silent for longer looks the same to the policy


def observation_scale(device) -> tuple[Tensor, Tensor]:
    """Return (offset, scale) so that ``(observation - offset) / scale`` is of order one, (16,) each."""
    layout = NavigationStateLayout
    offset = torch.zeros(layout.DIMENSION, device=device)
    scale = torch.ones(layout.DIMENSION, device=device)
    scale[layout.POSITION] = 20.0
    scale[layout.VELOCITY] = 5.0
    scale[layout.ANGULAR_VELOCITY] = 0.5
    offset[layout.MASS] = MASS_OFFSET_KG
    scale[layout.MASS] = MASS_SCALE_KG
    scale[layout.HEIGHT_ABOVE_GROUND] = 20.0
    return offset, scale


class PolicyActionAdapter:
    """Turns policy outputs into Commands and observations into policy inputs, batched."""

    def __init__(self, preset: LanderPreset, device, action_mode: str = "attitude_commands", maximum_tilt_deg: float = 15.0) -> None:
        """Build the jet table, the attitude controller and the observation scaling from the preset."""
        if action_mode not in ACTION_DIMENSION_BY_MODE:
            raise ValueError(f"action_mode must be one of {sorted(ACTION_DIMENSION_BY_MODE)}, not '{action_mode}'")
        self.preset = preset
        self.device = device
        self.action_mode = action_mode
        self.action_dimension = ACTION_DIMENSION_BY_MODE[action_mode]
        self.maximum_tilt_tangent = math.tan(math.radians(float(maximum_tilt_deg)))
        self.jet_table = JetCoupleTable(preset, device=device)
        self.attitude_controller = AttitudeController(preset, device=device)
        self.maximum_throttle = preset.engine_maximum_throttle_fraction
        self.hover_throttle = preset.mass_at_start * MOON_SURFACE_GRAVITY / preset.engine_rated_thrust
        self.offset, self.scale = observation_scale(device)

    def throttle_from_action(self, action: Tensor) -> Tensor:
        """Return the throttle command for a clipped throttle action in [-1, 1] (0 = hover)."""
        below = (action + 1.0) * self.hover_throttle
        above = self.hover_throttle + action * (self.maximum_throttle - self.hover_throttle)
        return torch.where(action < 0.0, below, above)

    def scale_observation(self, observation: Tensor) -> Tensor:
        """Return the scaled observation: the 16 navigation values, then any health columns.

        Health columns come in pairs per aided sensor: the failed flag (kept as 0 or 1) and the
        data age in seconds (clipped to ``MAXIMUM_DATA_AGE_S``, so a long-lost sensor stays of
        order one).
        """
        dimension = self.offset.shape[0]
        navigation = (observation[:, :dimension] - self.offset) / self.scale
        if observation.shape[1] == dimension:
            return navigation
        health = observation[:, dimension:].clone()
        health[:, 1::2] = health[:, 1::2].clamp(max=MAXIMUM_DATA_AGE_S)
        return torch.cat((navigation, health), dim=1)

    @classmethod
    def from_agent_parameters(cls, preset: LanderPreset, device, parameters: dict) -> PolicyActionAdapter:
        """Build the adapter from the ``ppo_mlp`` agent parameters (``action_mode``, ``maximum_tilt_deg``)."""
        return cls(
            preset,
            device,
            action_mode=str(parameters.get("action_mode", "attitude_commands")),
            maximum_tilt_deg=float(parameters.get("maximum_tilt_deg", 15.0)),
        )

    def command_from_policy_actions(self, actions: Tensor, observation: Tensor, control_step_duration: float) -> Command:
        """Return the Command for policy actions, (N, action_dimension), given the unscaled observation."""
        clipped = actions.clamp(-1.0, 1.0)
        throttle = self.throttle_from_action(clipped[:, 0])
        if self.action_mode == "axis_commands":
            jets = self.jet_table.jets_for_axis_commands(clipped[:, 1:4])
        else:
            layout = NavigationStateLayout
            direction = torch.cat(
                (clipped[:, 1:3] * self.maximum_tilt_tangent, torch.ones_like(clipped[:, :1])), dim=1
            )
            wanted_orientation = quaternion_for_thrust_direction(direction)
            jets = self.attitude_controller.compute_jet_duties(
                observation[:, layout.ORIENTATION_QUATERNION],
                observation[:, layout.ANGULAR_VELOCITY],
                wanted_orientation,
                control_step_duration,
            )
        return Command(
            throttle=throttle,
            gimbal=torch.zeros(actions.shape[0], 2, device=actions.device),
            reaction_control_jets=jets,
        )

    def environment_actions(self, actions: Tensor, observation: Tensor, control_step_duration: float) -> Tensor:
        """Return the environment's flat action tensor, (N, 3 + jets), for policy actions."""
        return self.command_from_policy_actions(actions, observation, control_step_duration).to_action_tensor()
