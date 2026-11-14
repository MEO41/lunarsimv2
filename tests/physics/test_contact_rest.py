# SPDX-License-Identifier: BSD-3-Clause
"""Physics validation (GPU): contact rest.

Known answer: the lander settles, and the sum of the leg forces equals m g.

64 landers touch down with the engine off, in four groups of 16:

- A "gentle":  dropped from rest 0.3 m above the ground (about 1 m/s at contact)
- B "hard":    3.0 m/s vertical at contact (the 10 ft/s design limit)
- C "sliding": 2.1 m/s vertical with 1.2 m/s horizontal (corner of the design envelope)
- D "tilted":  2.1 m/s vertical with a 6 degree tilt (the design limit)

Tolerances and why:
- No bounce: once the legs carry more than 1 % of the weight, the total leg force never falls
  below 0.1 % of the weight again. Any lift-off would give exactly zero force.
- Leg force sum within 0.5 % of m g (mean of the last 2 s). At rest this is Newton's first
  law; 0.5 % covers the force noise of the solver (its standard deviation is about 1.5 %
  per sample, averaged over 100 samples).
- Each leg within 3 % of m g / 4 in the symmetric groups A and B: four spring legs on a flat
  plane share the load equally, except for the 4 cm centre-of-mass offset of the Apollo 11 LM,
  which moves about 2 % of the load toward the +x leg; 3 % covers that.
- Strut compression at rest within 2 % of each leg's static share of the body weight divided
  by the stiffness (groups A and B). The share follows from a rigid body on four equal springs:
  W/4 + W (c . p) / (2 r^2), with c the body's centre of mass and p the leg position, both
  measured from the vehicle axis. The PGS solver with 16 iterations reproduces the spring to 0.2 %.
- At rest: the position changes by less than 1 mm over the last 4 s; the height standard
  deviation is below 0.1 mm; the reported speed is below 1 cm/s and the body rate below
  0.2 deg/s. The solver leaves a vibration of about 10 micrometres in the struts, which shows
  up as up to 5 mm/s of velocity noise. That is 0.2 % of the 3 m/s touchdown limit.
- Upright at rest: tilt below 0.5 degree on the flat ground.
- The strut never reaches the end of its 0.813 m stroke.
- Group C slides and stops within 0.1 to 1.2 m. Pure sliding at the static weight with a
  friction coefficient of 0.4 would give v^2 / (2 mu g) = 1.1 m; the impact raises the
  normal force, so the real distance is shorter.
"""

import math

import pytest
import torch
from isaac_test_support import make_actions, make_environment, read_truth, set_start_states

from lunarsim.core.binding import SlotName
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY
from lunarsim.core.rotation_math import quaternion_from_axis_angle, tilt_angle_from_vertical

GROUP_SIZE = 16
GENTLE, HARD, SLIDING, TILTED = (slice(index * GROUP_SIZE, (index + 1) * GROUP_SIZE) for index in range(4))
SETTLE_TIME = 12.0


@pytest.fixture(scope="module")
def touchdown():
    """Drop the four groups, record 12 s, and return the histories and the preset."""
    count = 4 * GROUP_SIZE
    environment = make_environment("s1_terminal_truth", [f"num_envs={count}", "actuators.initial_throttle=0.0", "metrics=none"])
    device = environment.device
    plant = environment.bindings[SlotName.PLANT]
    preset = plant.preset

    altitude = torch.full((count,), 0.05, device=device)
    altitude[GENTLE] = 0.3
    altitude[TILTED] = 0.05 + preset.footpad_radius_from_thrust_axis * math.sin(math.radians(6.0))
    vertical_speed = torch.zeros(count, device=device)
    vertical_speed[HARD] = -3.0
    vertical_speed[SLIDING] = -2.1
    vertical_speed[TILTED] = -2.1
    set_start_states(environment, altitude=altitude, vertical_speed=vertical_speed)
    start = environment.bindings[SlotName.START]
    position, velocity, orientation = start.fixed_start_states
    velocity[SLIDING, 0] = 1.2
    tilt_axis = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(GROUP_SIZE, 1)
    orientation[TILTED] = quaternion_from_axis_angle(tilt_axis, torch.full((GROUP_SIZE,), math.radians(6.0), device=device))
    start.set_fixed_start_states(position, velocity, orientation)
    environment.reset()

    history = {"position": [], "speed": [], "rate": [], "leg_forces": [], "strut": [], "tilt": []}
    for _ in range(int(SETTLE_TIME / environment.clock.control_step_duration)):
        environment.step(make_actions(environment, throttle=0.0))
        truth = read_truth(environment)
        history["position"].append(truth.position.clone())
        history["speed"].append(truth.velocity.norm(dim=1))
        history["rate"].append(truth.angular_velocity.norm(dim=1))
        history["leg_forces"].append(truth.contact.clone())
        history["strut"].append(plant.strut_compression().clone())
        history["tilt"].append(tilt_angle_from_vertical(truth.orientation_quaternion))
    history = {name: torch.stack(values) for name, values in history.items()}  # (steps, N, ...)
    weight = preset.mass_at_start * MOON_SURFACE_GRAVITY
    yield history, preset, weight
    environment.close()


def test_no_bounce(touchdown):
    """After first contact, the footpads never leave the ground in any group."""
    history, _, weight = touchdown
    total_force = history["leg_forces"].sum(dim=2)  # (steps, N)
    in_contact = total_force > 0.01 * weight
    assert bool(in_contact.any(dim=0).all()), "some lander never touched the ground"
    after_first_contact = in_contact.cumsum(dim=0) > 0
    lost_contact = after_first_contact & (total_force < 0.001 * weight)
    assert int(lost_contact.sum()) == 0


def test_sum_of_leg_forces_equals_weight(touchdown):
    """Mean total leg force over the last 2 s is m g within 0.5 %, in every environment."""
    history, _, weight = touchdown
    mean_total = history["leg_forces"][-100:].sum(dim=2).mean(dim=0)
    assert float((mean_total / weight - 1.0).abs().max()) < 0.005


def test_symmetric_landings_share_the_load_equally(touchdown):
    """Groups A and B: each leg carries m g / 4 within 3 %."""
    history, _, weight = touchdown
    mean_per_leg = history["leg_forces"][-100:].mean(dim=0)  # (N, 4)
    for group in (GENTLE, HARD):
        assert float((mean_per_leg[group] / (0.25 * weight) - 1.0).abs().max()) < 0.03


def test_strut_compression_matches_the_spring(touchdown):
    """Groups A and B: rest compression is each leg's static load / stiffness within 2 %; no strut bottoms out."""
    history, preset, _ = touchdown
    body_weight = preset.body_mass * MOON_SURFACE_GRAVITY
    axis_x, axis_y = preset.vehicle_axis_position
    body_x, body_y, _ = preset.body_center_of_mass
    radius = preset.footpad_radius_from_thrust_axis
    shares = [
        0.25 + ((body_x - axis_x) * (leg_x - axis_x) + (body_y - axis_y) * (leg_y - axis_y)) / (2.0 * radius**2)
        for leg_x, leg_y, _ in preset.leg_positions
    ]
    expected = torch.tensor(shares, device=history["strut"].device) * body_weight / preset.strut_stiffness
    rest_compression = history["strut"][-100:].mean(dim=0)
    for group in (GENTLE, HARD):
        assert float((rest_compression[group] / expected - 1.0).abs().max()) < 0.02
    assert float(history["strut"].max()) < 0.95 * preset.strut_stroke


def test_lander_comes_to_rest_without_jitter(touchdown):
    """All groups: no drift, no height jitter, small velocity noise, upright."""
    history, _, weight = touchdown
    position = history["position"]
    drift = (position[-1] - position[-200]).norm(dim=1)
    assert float(drift.max()) < 1e-3
    assert float(position[-100:, :, 2].std(dim=0).max()) < 1e-4
    assert float(history["speed"][-100:].max()) < 0.01
    assert float(history["rate"][-100:].max()) < math.radians(0.2)
    assert float(history["tilt"][-1].max()) < math.radians(0.5)
    assert float(history["leg_forces"][-100:].sum(dim=2).std(dim=0).max()) < 0.03 * weight


def test_sliding_landing_stops_by_friction(touchdown):
    """Group C: the lander slides between 0.1 m and 1.2 m and stops."""
    history, _, _ = touchdown
    slide = history["position"][-1, SLIDING, 0] - history["position"][0, SLIDING, 0]
    assert float(slide.min()) > 0.1
    assert float(slide.max()) < 1.2
