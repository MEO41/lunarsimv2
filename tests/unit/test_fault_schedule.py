# SPDX-License-Identifier: BSD-3-Clause
"""Fault schedule (CPU, milestone 3): onset, every fault type, seeding, fault status."""

import pytest
import torch
import yaml
from helpers import make_binding_context, make_truth_state

from lunarsim.core.message_types import Measurement, MessageKey
from lunarsim.env.scenario_loader import default_config_directory
from lunarsim.slots.faults.fault_schedule import FAULT_CODE, TARGET_NAMES, FaultSchedule
from lunarsim.slots.sensors.terminal_suite import DOPPLER_VELOCIMETER_KEY, RADAR_ALTIMETER_KEY

STEP = 0.02
COUNT = 2


class _World:
    provides = frozenset({MessageKey.TRUTH_STATE, RADAR_ALTIMETER_KEY, DOPPLER_VELOCIMETER_KEY})
    requires = frozenset({RADAR_ALTIMETER_KEY, DOPPLER_VELOCIMETER_KEY, MessageKey.FAULT_STATUS})
    reads_truth = False


WORLD = _World()


def make_schedule(fault: str, target: str = "radar_altimeter", onset: float = 20.0, size_fraction: float = 1.0):
    """Return a schedule with one fixed fault on environment 0 and none on environment 1."""
    with open(default_config_directory() / "faults" / "schedule.yaml") as file:
        parameters = yaml.safe_load(file)
    parameters.pop("binding")
    context = make_binding_context(COUNT)
    context.shared_data["control_step_duration"] = STEP
    schedule = FaultSchedule(parameters, context)
    schedule.set_fixed_faults(
        torch.tensor([FAULT_CODE[fault], FAULT_CODE["none"]]),
        torch.tensor([TARGET_NAMES.index(target), -1]),
        torch.tensor([onset, onset]),
        torch.tensor([size_fraction, size_fraction]),
    )
    schedule.reset(torch.arange(COUNT), torch.Generator().manual_seed(3))
    return schedule, context.bus


def fly(schedule, bus, heights, values=None):
    """Feed one step per height (altimeter value = 10 + step, always a new sample); return the outputs."""
    outputs = []
    for step, height in enumerate(heights):
        truth = make_truth_state(COUNT)
        truth.time[:] = step * STEP
        truth.footpad_height_above_ground[:] = height
        bus.write(MessageKey.TRUTH_STATE, truth, WORLD)
        value = torch.full((COUNT, 1), 10.0 + step) if values is None else values[step]
        bus.write(RADAR_ALTIMETER_KEY, Measurement(value=value, valid=torch.ones(COUNT, dtype=torch.bool), stamp=truth.time.clone(), covariance=torch.ones(COUNT, 1)), WORLD)
        bus.write(DOPPLER_VELOCIMETER_KEY, Measurement(value=torch.ones(COUNT, 6), valid=torch.ones(COUNT, dtype=torch.bool), stamp=truth.time.clone(), covariance=torch.ones(COUNT, 6)), WORLD)
        schedule.step(bus, STEP)
        outputs.append((bus.read(RADAR_ALTIMETER_KEY, WORLD), bus.read(DOPPLER_VELOCIMETER_KEY, WORLD), bus.read(MessageKey.FAULT_STATUS, WORLD)))
    return outputs


def test_bias_starts_at_the_onset_altitude_and_leaves_the_other_lander_alone():
    """Above 20 m nothing changes; from the first step at or below 20 m the bias (5 m) is added."""
    schedule, bus = make_schedule("bias")
    outputs = fly(schedule, bus, [25.0, 21.0, 20.0, 15.0])
    values = [float(altimeter.value[0, 0]) for altimeter, _, _ in outputs]
    assert values[:2] == [10.0, 11.0]
    assert abs(values[2] - 12.0) == pytest.approx(5.0) and abs(values[3] - 13.0) == pytest.approx(5.0)
    assert [float(altimeter.value[1, 0]) for altimeter, _, _ in outputs] == [10.0, 11.0, 12.0, 13.0]
    status = outputs[-1][2]
    assert bool(status.active[0]) and not bool(status.active[1])
    assert float(status.onset_time[0]) == pytest.approx(2 * STEP)


def test_drift_grows_with_time_since_the_onset():
    """Drift at 1 m/s (top of the range): the error after 0.1 s of fault is 0.1 m."""
    schedule, bus = make_schedule("drift")
    outputs = fly(schedule, bus, [10.0] * 6)
    errors = [abs(float(altimeter.value[0, 0]) - (10.0 + step)) for step, (altimeter, _, _) in enumerate(outputs)]
    assert errors[0] == pytest.approx(0.0, abs=1e-6)
    assert errors[5] == pytest.approx(5 * STEP * 1.0, rel=1e-4)


def test_stuck_repeats_the_last_value_before_the_onset_as_new():
    """The value freezes at the last sample above the onset; valid stays True."""
    schedule, bus = make_schedule("stuck")
    outputs = fly(schedule, bus, [30.0, 25.0, 15.0, 10.0, 5.0])
    assert [float(a.value[0, 0]) for a, _, _ in outputs] == [10.0, 11.0, 11.0, 11.0, 11.0]
    assert all(bool(a.valid[0]) for a, _, _ in outputs)


def test_scale_and_total_loss():
    """Scale multiplies by 1 +/- 0.2; total loss gives NaN and valid False."""
    schedule, bus = make_schedule("scale")
    altimeter = fly(schedule, bus, [10.0])[0][0]
    assert float(altimeter.value[0, 0]) in (pytest.approx(8.0), pytest.approx(12.0))
    schedule, bus = make_schedule("total_loss", target="doppler_velocimeter")
    _, doppler, status = fly(schedule, bus, [10.0])[0]
    assert not bool(doppler.valid[0]) and bool(torch.isnan(doppler.value[0]).all())
    assert bool(doppler.valid[1])
    assert bool(status.corrupting_now[0])


def test_doppler_bias_uses_velocity_and_range_sizes():
    """A Doppler bias adds 0.5 m/s to the three velocities and 5 m to the three ranges."""
    schedule, bus = make_schedule("bias", target="doppler_velocimeter")
    _, doppler, _ = fly(schedule, bus, [10.0])[0]
    error = (doppler.value[0] - 1.0).abs()
    assert torch.allclose(error[:3], torch.full((3,), 0.5)) and torch.allclose(error[3:], torch.full((3,), 5.0))


def test_dropout_and_delay_bursts_and_outliers():
    """With events nearly certain: dropouts hide samples, delays deliver old stamps, outliers jump."""
    schedule, bus = make_schedule("dropout_burst")
    schedule.event_rate[:] = 1000.0
    outputs = fly(schedule, bus, [10.0] * 5)
    assert not any(bool(a.valid[0]) for a, _, _ in outputs[1:])
    schedule, bus = make_schedule("delay_spike")
    schedule.event_rate[:] = 1000.0
    schedule.delay_steps[:] = 2
    outputs = fly(schedule, bus, [10.0] * 6)
    altimeter = outputs[5][0]
    assert float(altimeter.stamp[0]) == pytest.approx(3 * STEP) and float(altimeter.value[0, 0]) == 13.0
    schedule, bus = make_schedule("outlier_spike")
    schedule.event_rate[:] = 1000.0
    outputs = fly(schedule, bus, [10.0] * 3)
    assert abs(float(outputs[0][0].value[0, 0]) - 10.0) == pytest.approx(20.0)


def test_same_seed_draws_the_same_faults():
    """Two schedules reset with the same seed draw identical faults; a different seed differs."""
    with open(default_config_directory() / "faults" / "schedule.yaml") as file:
        parameters = yaml.safe_load(file)
    parameters.pop("binding")

    def draw(seed):
        """Return the drawn fault codes, targets and onsets of 256 environments."""
        context = make_binding_context(256)
        context.shared_data["control_step_duration"] = STEP
        schedule = FaultSchedule(parameters, context)
        schedule.reset(torch.arange(256), torch.Generator().manual_seed(seed))
        return torch.cat((schedule.fault_type.float(), schedule.target.float(), schedule.onset_altitude))

    assert torch.equal(draw(1), draw(1))
    assert not torch.equal(draw(1), draw(2))
    codes = draw(1)[:256]
    assert set(codes.long().tolist()) == set(range(1, len(FAULT_CODE)))


def test_fault_grid_covers_every_cell_evenly():
    """160 environments: each of the 80 cells twice, with the grid's onsets and mid sizes."""
    from lunarsim.slots.faults.fault_grid import GRID_ONSET_ALTITUDES_M, apply_fault_grid, grid_cells

    with open(default_config_directory() / "faults" / "schedule.yaml") as file:
        parameters = yaml.safe_load(file)
    parameters.pop("binding")
    context = make_binding_context(160)
    context.shared_data["control_step_duration"] = STEP
    schedule = FaultSchedule(parameters, context)
    apply_fault_grid(schedule, 160, "cpu")
    schedule.reset(torch.arange(160), torch.Generator().manual_seed(1000))
    cells = {(int(code), int(target), float(onset)) for code, target, onset in zip(schedule.fault_type, schedule.target, schedule.onset_altitude)}
    assert len(grid_cells()) == 80 and len(cells) == 80
    assert set(schedule.onset_altitude.tolist()) == set(GRID_ONSET_ALTITUDES_M)
    bias = schedule.fault_type == FAULT_CODE["bias"]
    altimeter = schedule.target == TARGET_NAMES.index("radar_altimeter")
    assert torch.allclose(schedule.size[bias & altimeter].abs(), torch.tensor(2.75))  # middle of 0.5-5 m
