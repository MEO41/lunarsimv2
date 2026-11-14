# SPDX-License-Identifier: BSD-3-Clause
"""Touchdown metric on synthetic truth states (CPU)."""

import math

import pytest
import torch
from helpers import make_binding_context, make_truth_state

from lunarsim.assets.lander_preset import load_lander_preset
from lunarsim.core.message_types import MessageKey, SurfaceGeometry
from lunarsim.core.rotation_math import quaternion_from_axis_angle
from lunarsim.env.scenario_loader import default_config_directory
from lunarsim.slots.metrics.touchdown import OUTCOME_CODE, OUTCOME_NAMES, TouchdownMetrics
from lunarsim.tools.episode_records import records_to_frame

CONTROL_STEP = 0.02


class _World:
    provides = frozenset({MessageKey.TRUTH_STATE, MessageKey.START_STATE, MessageKey.SURFACE_GEOMETRY})
    requires = frozenset({MessageKey.EPISODE_EVENTS})
    reads_truth = False


WORLD = _World()


def make_metrics(count=1, episode_length=60.0):
    """Build the metric with the Apollo preset on the CPU and reset all environments from 20 m."""
    context = make_binding_context(count)
    preset = load_lander_preset(default_config_directory() / "landers" / "apollo_lunar_module.yaml")
    context.shared_data.update({"lander_preset": preset, "episode_length_seconds": episode_length})
    metrics = TouchdownMetrics({}, context)
    context.bus.write(MessageKey.SURFACE_GEOMETRY, SurfaceGeometry(site_origin=torch.zeros(count, 3)), WORLD)
    start = state(count, preset, altitude=20.0)
    context.bus.write(MessageKey.START_STATE, start, WORLD)
    metrics.reset(torch.arange(count), torch.Generator())
    return metrics, context, preset


def state(
    count, preset, altitude, vertical_speed=0.0, horizontal_speed=0.0, tilt_deg=0.0, leg_force=0.0, time=0.0, mass=7327.0, body_force=0.0
):
    """Build a TruthState with the given simple values for all environments (flat ground at z = 0)."""
    truth = make_truth_state(count)
    truth.footpad_height_above_ground[:] = altitude
    truth.body_contact[:] = body_force
    truth.position[:] = torch.tensor([0.0, 0.0, altitude + preset.center_of_mass_height_above_footpad_plane])
    truth.velocity[:] = torch.tensor([horizontal_speed, 0.0, vertical_speed])
    axis = torch.tensor([[1.0, 0.0, 0.0]]).repeat(count, 1)
    truth.orientation_quaternion = quaternion_from_axis_angle(axis, torch.full((count,), math.radians(tilt_deg)))
    truth.contact[:] = leg_force
    truth.time[:] = time
    truth.mass[:] = mass
    return truth


def run(metrics, context, states):
    """Feed truth states one control step apart until the episode ends; return the last events.

    The environment resets a finished episode at once, so feeding stops at the first end.
    """
    for truth in states:
        context.bus.write(MessageKey.TRUTH_STATE, truth, WORLD)
        metrics.step(context.bus, CONTROL_STEP)
        events = context.bus.read(MessageKey.EPISODE_EVENTS, WORLD)
        if bool((events.terminated | events.truncated).all()):
            break
    return events


def land(metrics, context, preset, vertical_speed, horizontal_speed=0.0, tilt_deg=0.0, final_tilt_deg=0.0):
    """Simulate an approach step, a contact step, then rest until the episode ends; return the record."""
    states = [state(1, preset, 0.1, vertical_speed, horizontal_speed, tilt_deg, time=1.0)]
    states.append(state(1, preset, 0.0, 0.3 * vertical_speed, 0.0, tilt_deg, leg_force=5000.0, time=1.02))
    states += [state(1, preset, 0.0, 0.0, 0.0, final_tilt_deg, leg_force=3000.0, time=1.04 + 0.02 * k) for k in range(60)]
    events = run(metrics, context, states)
    assert bool(events.terminated[0])
    return records_to_frame(metrics.drain_episode_records(), OUTCOME_NAMES).iloc[0]


def test_soft_landing_is_a_success_and_uses_the_approach_speed():
    """Touchdown values come from the step before contact; a gentle landing that settles succeeds."""
    metrics, context, preset = make_metrics()
    record = land(metrics, context, preset, vertical_speed=-1.0, horizontal_speed=0.5, tilt_deg=2.0)
    assert record["outcome"] == "success" and bool(record["success"])
    assert record["touchdown_vertical_speed_mps"] == pytest.approx(-1.0)
    assert record["touchdown_horizontal_speed_mps"] == pytest.approx(0.5)
    assert record["touchdown_tilt_deg"] == pytest.approx(2.0, abs=1e-3)
    assert record["start_altitude_m"] == pytest.approx(20.0, abs=1e-4)
    assert record["episode_duration_s"] == pytest.approx(1.02 + 1.0, abs=0.03)  # settled after the 1 s hold


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"vertical_speed": -3.5}, "hard_vertical"),
        ({"vertical_speed": -1.0, "horizontal_speed": 1.5}, "hard_horizontal"),
        ({"vertical_speed": -2.8, "horizontal_speed": 1.0}, "hard_horizontal"),  # limit shrinks above 7 ft/s
        ({"vertical_speed": -1.0, "tilt_deg": 8.0}, "tilt_at_contact"),
        ({"vertical_speed": -1.0, "final_tilt_deg": 60.0}, "tip_over"),
    ],
)
def test_each_limit_gives_its_own_failure_cause(arguments, expected):
    """One broken limit at a time maps to the matching outcome."""
    metrics, context, preset = make_metrics()
    record = land(metrics, context, preset, **arguments)
    assert record["outcome"] == expected and not bool(record["success"])


def test_horizontal_limit_follows_the_design_envelope():
    """4 ft/s up to 7 ft/s vertical, then linear to zero at 10 ft/s."""
    metrics, _, _ = make_metrics()
    limits = metrics.horizontal_speed_limit(torch.tensor([0.0, -2.134, -2.591, -3.048, -4.0]))
    assert limits.tolist() == pytest.approx([1.219, 1.219, 0.6095, 0.0, 0.0], abs=2e-3)


def test_time_limit_without_contact_is_truncated():
    """An episode that never touches down is truncated and recorded as timeout_no_contact."""
    metrics, context, preset = make_metrics(episode_length=1.0)
    events = run(metrics, context, [state(1, preset, 20.0, time=0.5), state(1, preset, 20.0, time=1.0)])
    assert bool(events.truncated[0]) and not bool(events.terminated[0])
    record = records_to_frame(metrics.drain_episode_records(), OUTCOME_NAMES).iloc[0]
    assert record["outcome"] == "timeout_no_contact"


def test_out_of_bounds_and_propellant_depleted():
    """Flying away terminates; a hard landing with empty tanks is blamed on the propellant."""
    metrics, context, preset = make_metrics(count=2)
    far = state(2, preset, 20.0, time=1.0)
    far.position[0, 0] = 400.0
    context.bus.write(MessageKey.TRUTH_STATE, far, WORLD)
    metrics.step(context.bus, CONTROL_STEP)
    events = context.bus.read(MessageKey.EPISODE_EVENTS, WORLD)
    assert events.terminated.tolist() == [True, False]
    record = records_to_frame(metrics.drain_episode_records(), OUTCOME_NAMES).iloc[0]
    assert record["outcome"] == "out_of_bounds"

    metrics, context, preset = make_metrics()
    dry_mass = preset.mass_without_usable_propellant
    states = [state(1, preset, 0.1, -4.0, time=1.0, mass=dry_mass)]
    states.append(state(1, preset, 0.0, -1.0, leg_force=9000.0, time=1.02, mass=dry_mass))
    states += [state(1, preset, 0.0, 0.0, leg_force=3000.0, time=1.04 + 0.02 * k, mass=dry_mass) for k in range(60)]
    run(metrics, context, states)
    record = records_to_frame(metrics.drain_episode_records(), OUTCOME_NAMES).iloc[0]
    assert record["outcome"] == "propellant_depleted"
    assert record["propellant_used_kg"] == pytest.approx(preset.usable_descent_propellant_at_start, abs=0.01)


def test_episode_numbers_count_up_and_reset_clears_contact():
    """After a reset the next episode starts clean and gets the next episode number."""
    metrics, context, preset = make_metrics()
    land(metrics, context, preset, vertical_speed=-1.0)
    context.bus.write(MessageKey.START_STATE, state(1, preset, 30.0), WORLD)
    metrics.reset(torch.tensor([0]), torch.Generator())
    assert not bool(metrics.has_contact[0])
    record = land(metrics, context, preset, vertical_speed=-3.5)
    assert int(record["episode_number"]) == 1 and record["outcome"] == "hard_vertical"
    assert OUTCOME_CODE["success"] == 0


def test_body_contact_after_a_soft_touchdown_is_a_failure():
    """A stage that touches the ground (for example on a rock) fails a landing that is inside every limit."""
    metrics, context, preset = make_metrics()
    states = [state(1, preset, 0.1, -1.0, time=1.0)]
    states.append(state(1, preset, 0.0, -0.3, leg_force=5000.0, body_force=800.0, time=1.02))
    states += [state(1, preset, 0.0, 0.0, leg_force=3000.0, time=1.04 + 0.02 * k) for k in range(60)]
    run(metrics, context, states)
    record = records_to_frame(metrics.drain_episode_records(), OUTCOME_NAMES).iloc[0]
    assert record["outcome"] == "body_contact"
    assert record["peak_body_contact_force_n"] == pytest.approx(800.0)


def test_nozzle_clearance_and_flat_footprint_on_flat_ground():
    """On flat ground the nozzle rim clears by its height above the pads; slope and relief are zero."""
    metrics, context, preset = make_metrics()
    record = land(metrics, context, preset, vertical_speed=-1.0)
    assert record["minimum_nozzle_clearance_m"] == pytest.approx(preset.nozzle_exit_height_above_footpad_plane, abs=1e-4)
    assert record["touchdown_ground_slope_deg"] == pytest.approx(0.0, abs=1e-4)
    assert record["touchdown_ground_relief_m"] == pytest.approx(0.0, abs=1e-6)
