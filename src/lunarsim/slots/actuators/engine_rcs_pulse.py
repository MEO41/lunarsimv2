# SPDX-License-Identifier: BSD-3-Clause
"""``actuators: engine_rcs_pulse`` - descent engine plus pulsed reaction-control jets."""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.assets.lander_preset import LanderPreset
from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import Command, MessageKey, Wrench
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.core.registry import register_binding
from lunarsim.slots.actuators.propulsion_models import DescentEngineModel, ReactionControlPulseModel


@register_binding(SlotName.ACTUATORS, "engine_rcs_pulse")
class EngineAndPulsedJetsActuators(BindingBase):
    """Turns a Command into a body-frame Wrench, every PhysX substep.

    Engine: throttle range, command slew limit, first-order thrust lag, thrust tail-off after a
    shutdown, propellant use.
    Jets: minimum pulse width, valve delay, propellant use. The gimbal command is accepted but
    ignored (the ``gimbal`` actuator is a later binding); the thrust acts along body +z through
    the centre of mass.

    Parameters: ``initial_throttle`` - ``hover`` (default: the engine starts at the throttle
    that balances the lunar weight) or a fraction of rated thrust.
    """

    requires = frozenset({MessageKey.COMMAND})
    provides = frozenset({MessageKey.ACTUATOR_WRENCH})

    def __init__(self, parameters, context) -> None:
        """Build the engine and jet models from the lander preset."""
        super().__init__(parameters, context)
        preset: LanderPreset = context.shared_data["lander_preset"]
        self.preset = preset
        count = self.number_of_environments
        self.engine = DescentEngineModel(
            count,
            rated_thrust=preset.engine_rated_thrust,
            minimum_throttle_fraction=preset.engine_minimum_throttle_fraction,
            maximum_throttle_fraction=preset.engine_maximum_throttle_fraction,
            lag_time_constant=preset.engine_thrust_lag_time_constant,
            maximum_throttle_rate=preset.engine_maximum_throttle_rate,
            off_below_fraction=preset.engine_off_below_fraction,
            exhaust_velocity=preset.engine_exhaust_velocity,
            device=self.device,
            shutdown_decay_time=preset.engine_shutdown_thrust_decay_time,
        )
        self.jets = ReactionControlPulseModel(
            count,
            preset.number_of_jets,
            minimum_pulse_duration=preset.jet_minimum_pulse_duration,
            valve_delay=preset.jet_valve_delay,
            physics_step_duration=context.clock.physics_step_duration,
            device=self.device,
        )
        self.jet_force_matrix = torch.tensor(preset.jet_force_matrix(), device=self.device).T  # (n_jets, 3)
        self.jet_torque_matrix = torch.tensor(preset.jet_torque_matrix(), device=self.device).T  # (n_jets, 3)
        self.jet_mass_flow_per_open_jet = preset.jet_thrust / preset.jet_exhaust_velocity
        self.throttle_command = torch.zeros(count, device=self.device)
        initial_throttle = self.parameters.get("initial_throttle", "hover")
        if initial_throttle == "hover":
            initial_throttle = preset.mass_at_start * MOON_SURFACE_GRAVITY / preset.engine_rated_thrust
        self.initial_throttle = float(initial_throttle)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Start the episode with a running engine, closed jets and the preset's propellant load."""
        self.engine.reset(env_ids, self.initial_throttle, self.preset.usable_descent_propellant_at_start)
        self.jets.reset(env_ids)
        self.throttle_command[env_ids] = self.initial_throttle

    def begin_control_step(self, bus, dt: float) -> None:
        """Take the new Command: latch the throttle and schedule the jet pulses of this control step."""
        command: Command = bus.read(MessageKey.COMMAND, self)
        self.throttle_command = command.throttle
        self.jets.begin_control_step(command.reaction_control_jets, dt)

    def step(self, bus, dt: float) -> None:
        """Advance engine and jets by one PhysX substep and write the actuator wrench."""
        thrust, engine_mass_flow = self.engine.step(self.throttle_command, dt)
        open_fraction = self.jets.step()
        force = open_fraction @ self.jet_force_matrix
        force[:, 2] += thrust
        torque = open_fraction @ self.jet_torque_matrix
        mass_flow = engine_mass_flow + open_fraction.sum(dim=1) * self.jet_mass_flow_per_open_jet
        bus.write(MessageKey.ACTUATOR_WRENCH, Wrench(force=force, torque=torque, mass_flow_rate=mass_flow), self)
