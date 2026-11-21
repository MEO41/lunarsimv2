# SPDX-License-Identifier: BSD-3-Clause
"""``actuators: engine_rcs_pulse`` - descent engine plus pulsed reaction-control jets."""

from __future__ import annotations

import math

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
    that balances the lunar weight) or a fraction of rated thrust; ``ideal_response`` (default
    false) - true removes the response limits for the model ablation of the transfer study
    (test T8): no thrust lag, no throttle slew limit, no shutdown tail-off, no jet minimum pulse
    and no valve delay. The throttle range, rated thrust and propellant use stay.
    """

    requires = frozenset({MessageKey.COMMAND, MessageKey.START_STATE})
    provides = frozenset({MessageKey.ACTUATOR_WRENCH})

    def __init__(self, parameters, context) -> None:
        """Build the engine and jet models from the lander preset."""
        super().__init__(parameters, context)
        preset: LanderPreset = context.shared_data["lander_preset"]
        self.preset = preset
        count = self.number_of_environments
        self.ideal_response = bool(self.parameters.get("ideal_response", False))
        ideal = self.ideal_response
        self.engine = DescentEngineModel(
            count,
            rated_thrust=preset.engine_rated_thrust,
            minimum_throttle_fraction=preset.engine_minimum_throttle_fraction,
            maximum_throttle_fraction=preset.engine_maximum_throttle_fraction,
            lag_time_constant=0.0 if ideal else preset.engine_thrust_lag_time_constant,
            maximum_throttle_rate=math.inf if ideal else preset.engine_maximum_throttle_rate,
            off_below_fraction=preset.engine_off_below_fraction,
            exhaust_velocity=preset.engine_exhaust_velocity,
            device=self.device,
            shutdown_decay_time=0.0 if ideal else preset.engine_shutdown_thrust_decay_time,
        )
        self.jets = ReactionControlPulseModel(
            count,
            preset.number_of_jets,
            minimum_pulse_duration=0.0 if ideal else preset.jet_minimum_pulse_duration,
            valve_delay=0.0 if ideal else preset.jet_valve_delay,
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
        """Start the episode with a running engine, closed jets and the propellant load.

        The load is the start state's mass minus the dry mass (the vehicle without usable descent
        propellant), so a phase that starts from a saved state (milestone 6) has the propellant
        that is left; with no start mass it is the preset's load.
        """
        load = torch.full((len(env_ids),), self.preset.usable_descent_propellant_at_start, device=self.device)
        if self.context.bus.has(MessageKey.START_STATE):
            start_mass = self.context.bus.read(MessageKey.START_STATE, self).mass[env_ids].to(torch.float32)
            from_mass = (start_mass - self.preset.mass_without_usable_propellant).clamp_min(0.0)
            load = torch.where(torch.isfinite(start_mass), from_mass, load)
        self.engine.reset(env_ids, self.initial_throttle, load)
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
        bus.write(MessageKey.ACTUATOR_WRENCH, Wrench(force=force, torque=torque, mass_flow_rate=mass_flow, engine_thrust=thrust.clone()), self)
