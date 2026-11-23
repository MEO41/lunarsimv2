# SPDX-License-Identifier: BSD-3-Clause
"""Measure how fast the closed-loop attitude follows a tilt command (transfer study, T8 lag step).

The lagged point mass of test T8 (``plant=analytic_point_mass plant.attitude_time_constant_s=...``)
turns its thrust axis towards the commanded direction with a first-order lag. Its time constant
is not made up: it is measured here on the analytic rigid-body lander (``analytic_rigid_body``,
standard actuators) flown by the same classical attitude controller that turns the PPO policy's
attitude commands into jet duties (``PolicyActionAdapter`` in ``attitude_commands`` mode).

The lander starts upright and hovers at the hover throttle; at t = 0 the commanded thrust
direction steps to a tilt of ``--tilts`` degrees towards site +x and stays there. Per step the
script prints the time at which the tilt reaches 63.2 % of the step (the time constant of a
first-order lag with the same rise), the 10-90 % rise time and the overshoot.

Run (CPU, no Isaac Sim):
    python scripts/measure_attitude_time_constant.py --tilts 5 10 15
"""

from __future__ import annotations

import argparse
import math

import torch

from lunarsim.agents.classical.attitude_controller import quaternion_for_thrust_direction
from lunarsim.agents.rsl_rl.policy_action_adapter import PolicyActionAdapter
from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import Command, MessageKey, NavigationStateLayout
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.core.rotation_math import tilt_angle_from_vertical
from lunarsim.env.analytic_lander_environment import AnalyticLanderEnvironment
from lunarsim.env.scenario_loader import load_scenario


class TruthReader:
    """Reader that may look at TruthState on the bus (measurement only)."""

    requires = frozenset({MessageKey.TRUTH_STATE})
    provides = frozenset()
    reads_truth = True


def first_time_at_or_above(times: list[float], values: list[float], level: float) -> float:
    """Return the first time at which ``values`` reaches ``level``, interpolated between samples (nan if never)."""
    for index in range(1, len(values)):
        if values[index] >= level:
            previous, current = values[index - 1], values[index]
            fraction = (level - previous) / (current - previous) if current != previous else 0.0
            return times[index - 1] + fraction * (times[index] - times[index - 1])
    return math.nan


def measure_step_response(tilt_degrees: float, duration_s: float) -> dict[str, float]:
    """Fly one tilt step on the analytic rigid body and return its 63.2 % time, 10-90 % rise time and overshoot."""
    scenario = load_scenario("s1_terminal_truth", ["plant=analytic_rigid_body", "num_envs=1"])
    environment = AnalyticLanderEnvironment(scenario, device="cpu")
    preset = environment.bindings[SlotName.PLANT].preset
    start = environment.bindings[SlotName.START]
    position = torch.tensor([[0.0, 0.0, 60.0 + start.center_of_mass_height]])
    velocity = torch.zeros(1, 3)
    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    start.set_fixed_start_states(position, velocity, upright)
    observations, _ = environment.reset()

    adapter = PolicyActionAdapter(preset, "cpu")
    hover = preset.mass_at_start * MOON_SURFACE_GRAVITY / preset.engine_rated_thrust
    tilt = math.radians(tilt_degrees)
    direction = torch.tensor([[math.sin(tilt), 0.0, math.cos(tilt)]])
    layout = NavigationStateLayout
    step_duration = environment.clock.control_step_duration
    reader = TruthReader()

    times, tilts = [0.0], [0.0]
    for step in range(1, int(round(duration_s / step_duration)) + 1):
        observation = observations["policy"]
        orientation = observation[:, layout.ORIENTATION_QUATERNION]
        wanted = quaternion_for_thrust_direction(direction, orientation)
        jets = adapter.attitude_controller.compute_jet_duties(
            orientation, observation[:, layout.ANGULAR_VELOCITY], wanted, step_duration
        )
        command = Command(throttle=torch.tensor([hover]), gimbal=torch.zeros(1, 2), reaction_control_jets=jets)
        observations, _, _, _, _ = environment.step(command.to_action_tensor())
        truth = environment.bus.read(MessageKey.TRUTH_STATE, reader)
        times.append(step * step_duration)
        tilts.append(math.degrees(float(tilt_angle_from_vertical(truth.orientation_quaternion)[0])))

    return {
        "tilt_step_deg": tilt_degrees,
        "time_to_63_percent_s": first_time_at_or_above(times, tilts, 0.632 * tilt_degrees),
        "rise_time_10_to_90_percent_s": first_time_at_or_above(times, tilts, 0.9 * tilt_degrees)
        - first_time_at_or_above(times, tilts, 0.1 * tilt_degrees),
        "overshoot_percent": 100.0 * (max(tilts) - tilt_degrees) / tilt_degrees,
        "final_tilt_deg": tilts[-1],
    }


def main() -> None:
    """Parse the tilt steps, measure each and print one line per step."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tilts", type=float, nargs="+", default=[5.0, 10.0, 15.0], help="tilt steps in degrees")
    parser.add_argument("--duration", type=float, default=6.0, help="flight time per step, s")
    arguments = parser.parse_args()
    for tilt_degrees in arguments.tilts:
        result = measure_step_response(tilt_degrees, arguments.duration)
        print(
            f"tilt step {result['tilt_step_deg']:5.1f} deg: 63.2 % after {result['time_to_63_percent_s']:.3f} s, "
            f"10-90 % rise {result['rise_time_10_to_90_percent_s']:.3f} s, overshoot {result['overshoot_percent']:.1f} %, "
            f"tilt after {arguments.duration:.0f} s {result['final_tilt_deg']:.2f} deg"
        )


if __name__ == "__main__":
    main()
