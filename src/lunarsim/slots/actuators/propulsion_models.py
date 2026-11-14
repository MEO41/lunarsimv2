# SPDX-License-Identifier: BSD-3-Clause
"""Pure-torch models of the descent engine and the reaction-control jets.

No simulator is needed, so the propellant and pulse-timing tests run on the CPU.
All state tensors have the environment count ``N`` as their first dimension.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


class DescentEngineModel:
    """Throttleable engine: throttle range, command slew limit, first-order thrust lag, propellant.

    The throttle command is a fraction of rated thrust. A command below
    ``off_below_fraction`` shuts the engine down; any other command is clamped to the
    throttleable range. The commanded fraction slews at ``maximum_throttle_rate`` and the
    thrust follows it with a first-order lag of ``lag_time_constant``.
    """

    def __init__(
        self,
        number_of_environments: int,
        rated_thrust: float,
        minimum_throttle_fraction: float,
        maximum_throttle_fraction: float,
        lag_time_constant: float,
        maximum_throttle_rate: float,
        off_below_fraction: float,
        exhaust_velocity: float,
        device: torch.device | str = "cpu",
    ) -> None:
        """Store the engine constants and allocate the per-environment state."""
        self.rated_thrust = float(rated_thrust)
        self.minimum_throttle_fraction = float(minimum_throttle_fraction)
        self.maximum_throttle_fraction = float(maximum_throttle_fraction)
        self.lag_time_constant = float(lag_time_constant)
        self.maximum_throttle_rate = float(maximum_throttle_rate)
        self.off_below_fraction = float(off_below_fraction)
        self.exhaust_velocity = float(exhaust_velocity)
        self.slewed_fraction = torch.zeros(number_of_environments, device=device)
        self.thrust_fraction = torch.zeros(number_of_environments, device=device)
        self.propellant_mass = torch.zeros(number_of_environments, device=device)

    def reset(self, env_ids: Tensor, initial_fraction: Tensor | float, propellant_mass: Tensor | float) -> None:
        """Set the engine of the given environments to a steady throttle and a propellant load."""
        self.slewed_fraction[env_ids] = initial_fraction
        self.thrust_fraction[env_ids] = initial_fraction
        self.propellant_mass[env_ids] = propellant_mass

    def step(self, throttle_command: Tensor, dt: float) -> tuple[Tensor, Tensor]:
        """Advance by ``dt`` and return (mean thrust over the step in N, mass flow in kg/s), shape (N,).

        The lag is integrated exactly for a constant slewed command, and the returned thrust is
        the mean over the step, so the impulse is right for any step size.
        """
        engine_on = (throttle_command >= self.off_below_fraction) & (self.propellant_mass > 0.0)
        target = throttle_command.clamp(self.minimum_throttle_fraction, self.maximum_throttle_fraction)
        # A running engine cannot go below the minimum throttle, so a start jumps to it.
        slewed = torch.maximum(self.slewed_fraction, torch.full_like(target, self.minimum_throttle_fraction))
        maximum_change = self.maximum_throttle_rate * dt
        slewed = slewed + (target - slewed).clamp(-maximum_change, maximum_change)
        slewed = torch.where(engine_on, slewed, torch.zeros_like(slewed))

        decay = math.exp(-dt / self.lag_time_constant) if self.lag_time_constant > 0.0 else 0.0
        mean_weight = (self.lag_time_constant / dt) * (1.0 - decay) if self.lag_time_constant > 0.0 else 0.0
        difference = self.thrust_fraction - slewed
        mean_fraction = slewed + difference * mean_weight
        self.thrust_fraction = slewed + difference * decay
        self.slewed_fraction = slewed

        thrust = mean_fraction * self.rated_thrust
        mass_flow = thrust / self.exhaust_velocity
        # Do not burn more propellant than is left: scale the last step down.
        burned = mass_flow * dt
        scale = torch.where(burned > self.propellant_mass, self.propellant_mass / burned.clamp_min(1e-12), 1.0)
        thrust, mass_flow = thrust * scale, mass_flow * scale
        self.propellant_mass = (self.propellant_mass - mass_flow * dt).clamp_min(0.0)
        ran_dry = self.propellant_mass <= 0.0
        self.thrust_fraction = torch.where(ran_dry, torch.zeros_like(self.thrust_fraction), self.thrust_fraction)
        self.slewed_fraction = torch.where(ran_dry, torch.zeros_like(self.slewed_fraction), self.slewed_fraction)
        return thrust, mass_flow


class ReactionControlPulseModel:
    """On/off jets with a minimum pulse width and a valve delay.

    Once per control step the agent gives a duty (0..1) per jet. The jet is commanded on for
    ``duty * control_step`` seconds from the start of the control step. An on-time shorter
    than the minimum pulse is dropped (the electronics cannot make a shorter pulse). The valve
    opens and closes ``valve_delay`` seconds after its command. Per physics substep the model
    returns the fraction of the substep that each valve was open, so the impulse of every pulse
    is exact even when the pulse edges fall between substeps.
    """

    def __init__(
        self,
        number_of_environments: int,
        number_of_jets: int,
        minimum_pulse_duration: float,
        valve_delay: float,
        physics_step_duration: float,
        device: torch.device | str = "cpu",
    ) -> None:
        """Store the timing constants and allocate the delay line."""
        self.minimum_pulse_duration = float(minimum_pulse_duration)
        self.physics_step_duration = float(physics_step_duration)
        whole_steps = int(math.floor(valve_delay / physics_step_duration + 1e-9))
        remainder = valve_delay - whole_steps * physics_step_duration
        self._whole_delay_steps = whole_steps
        self._older_weight = max(0.0, remainder / physics_step_duration)
        # history[..., 0] is the newest commanded open-fraction; index k is k substeps old.
        self._history = torch.zeros(number_of_environments, number_of_jets, whole_steps + 2, device=device)
        self.on_time = torch.zeros(number_of_environments, number_of_jets, device=device)
        self.elapsed_in_control_step = torch.zeros(number_of_environments, device=device)

    def reset(self, env_ids: Tensor) -> None:
        """Close all valves of the given environments and clear their delay lines."""
        self._history[env_ids] = 0.0
        self.on_time[env_ids] = 0.0
        self.elapsed_in_control_step[env_ids] = 0.0

    def begin_control_step(self, duty: Tensor, control_step_duration: float) -> None:
        """Take the duty command (N, n_jets) for the coming control step."""
        on_time = duty.clamp(0.0, 1.0) * control_step_duration
        # 1 microsecond of slack so that a duty of exactly min_pulse / control_step is kept.
        too_short = on_time < self.minimum_pulse_duration - 1e-6
        self.on_time = torch.where(too_short, torch.zeros_like(on_time), on_time)
        self.elapsed_in_control_step[:] = 0.0

    def step(self) -> Tensor:
        """Advance one physics substep and return the open fraction of every valve, (N, n_jets)."""
        dt = self.physics_step_duration
        window_start = self.elapsed_in_control_step.unsqueeze(1)
        commanded = ((self.on_time - window_start).clamp(0.0, dt)) / dt
        self._history = torch.roll(self._history, shifts=1, dims=2)
        self._history[:, :, 0] = commanded
        newer = self._history[:, :, self._whole_delay_steps]
        older = self._history[:, :, self._whole_delay_steps + 1]
        self.elapsed_in_control_step += dt
        return (1.0 - self._older_weight) * newer + self._older_weight * older
