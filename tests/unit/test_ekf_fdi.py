# SPDX-License-Identifier: BSD-3-Clause
"""Fault detection (CPU, milestone 3): ekf_fdi on the synthetic flight of test_ekf_consistency.

Same 200 landers and sensors as the consistency test, with the fault schedule between sensors
and filter. Known answers:
- Without faults, no sensor is declared failed (gate 99.9 %, three rejections in a row).
- A 5 m altimeter bias that starts at 20 m (mid-flight, after the filter has converged) is
  declared within 0.5 s (three samples at 20 Hz: about 0.15 s), and the filter's height error
  stays below 0.5 m for 95 % of the landers, while the plain ``ekf`` takes part of the bias in
  (above 0.5 m; the three Doppler ranges outvote the altimeter only partly).
- A total loss of the Doppler velocimeter is declared 0.5 s after its last sample, which comes
  up to one sample period (50 ms) before the onset: 0.45 to 0.6 s after the onset.

A fault present from the first sample is a different case: with a 1 m first-estimate error the
filter cannot tell which of the two aiding sensors is wrong and may isolate the good one. The
fault_response metric records such faults separately.
"""

import pytest
import torch
import yaml
from helpers import make_binding_context
from test_ekf_consistency import CONFIG, COUNT, CONTROL_STEP, WORLD, load, truth_at

from lunarsim.assets.lander_preset import load_lander_preset
from lunarsim.core.message_types import MessageKey, SurfaceGeometry
from lunarsim.slots.faults.fault_schedule import FAULT_CODE, TARGET_NAMES, FaultSchedule
from lunarsim.slots.navigation.ekf import ExtendedKalmanFilterNavigation
from lunarsim.slots.navigation.ekf_fdi import FaultDetectingKalmanFilterNavigation
from lunarsim.slots.sensors.terminal_suite import TerminalSensorSuite
from lunarsim.slots.start.initial_navigation_error import InitialNavigationError

DURATION_S = 12.0


class _Faulted:
    provides = frozenset({MessageKey.TRUTH_STATE, MessageKey.SURFACE_GEOMETRY, MessageKey.INITIAL_NAVIGATION_STATE})
    requires = frozenset({MessageKey.NAVIGATION_ESTIMATE, MessageKey.FAULT_STATUS})
    reads_truth = False


def fly(navigation_class, fault: str | None, target: str = "radar_altimeter", onset_m: float = 20.0):
    """Fly the synthetic trajectory through sensors, faults and a filter; return per-step records."""
    context = make_binding_context(COUNT)
    preset = load_lander_preset(CONFIG / "landers" / "apollo_lunar_module.yaml")
    suite_parameters = load("sensors/terminal_suite.yaml")
    context.shared_data.update({"lander_preset": preset, "sensor_suite_parameters": suite_parameters, "control_step_duration": CONTROL_STEP})
    suite = TerminalSensorSuite(suite_parameters, context)
    with open(CONFIG / "nav" / "ekf.yaml") as file:
        navigation_parameters = yaml.safe_load(file)
    navigation_parameters["detection"] = load("nav/ekf_fdi.yaml")["detection"]
    navigation = navigation_class(navigation_parameters, context)
    schedule = FaultSchedule(load("faults/schedule.yaml"), context)
    code = FAULT_CODE[fault] if fault else FAULT_CODE["none"]
    schedule.set_fixed_faults(
        torch.full((COUNT,), code), torch.full((COUNT,), TARGET_NAMES.index(target) if fault else -1),
        torch.full((COUNT,), onset_m), torch.ones(COUNT),
    )
    bus = context.bus
    bus.write(MessageKey.SURFACE_GEOMETRY, SurfaceGeometry(site_origin=torch.zeros(COUNT, 3)), WORLD)
    error = InitialNavigationError({"position_m": 1.0, "velocity_mps": 0.2, "attitude_deg": 0.1}, COUNT, "cpu")
    generator = torch.Generator().manual_seed(7)
    everyone = torch.arange(COUNT)
    error.reset(everyone, generator)
    bus.write(MessageKey.INITIAL_NAVIGATION_STATE, error.estimate(truth_at(0.0)), WORLD)
    for binding in (suite, schedule, navigation):
        binding.reset(everyone, generator)
    records = []
    for step in range(int(DURATION_S / CONTROL_STEP) + 1):
        truth = truth_at(step * CONTROL_STEP)
        bus.write(MessageKey.TRUTH_STATE, truth, WORLD)
        suite.step(bus, CONTROL_STEP)
        schedule.step(bus, CONTROL_STEP)
        navigation.step(bus, CONTROL_STEP)
        estimate = bus.read(MessageKey.NAVIGATION_ESTIMATE, _Faulted())
        status = bus.read(MessageKey.FAULT_STATUS, _Faulted())
        records.append((truth.time[0].item(), estimate.health.clone(), (navigation.filter.position[:, 2] - truth.position[:, 2]).abs(), status.active.clone()))
    return records


def test_no_false_failures_without_faults():
    """200 landers, 12 s, no fault: no sensor is ever declared failed."""
    records = fly(FaultDetectingKalmanFilterNavigation, None)
    failed = torch.stack([health[:, [0, 2]] for _, health, _, _ in records]).amax()
    assert float(failed) == 0.0


def test_altimeter_bias_is_detected_and_kept_out():
    """5 m altimeter bias from 20 m: declared within 0.5 s; height error p95 < 0.5 m, plain ekf above 0.5 m."""
    records = fly(FaultDetectingKalmanFilterNavigation, "bias")
    onset = torch.stack([active for *_, active in records]).float().argmax(dim=0) * CONTROL_STEP
    declared = torch.stack([health[:, 0] for _, health, _, _ in records]).argmax(dim=0) * CONTROL_STEP
    delay = declared - onset
    print("detection delay median", float(delay.median()), "max", float(delay.max()))
    assert float(delay.max()) < 0.5
    worst_detecting = torch.stack([error for _, _, error, _ in records]).amax(dim=0)
    worst_plain = torch.stack([error for _, _, error, _ in fly(ExtendedKalmanFilterNavigation, "bias")]).amax(dim=0)
    print("height error p95: ekf_fdi", float(worst_detecting.quantile(0.95)), "ekf", float(worst_plain.quantile(0.95)))
    assert float(worst_detecting.quantile(0.95)) < 0.5
    assert float(worst_plain.quantile(0.95)) > 0.5


def test_doppler_total_loss_is_declared_after_the_loss_time():
    """Doppler lost from 20 m: declared failed 0.45 to 0.6 s after the onset."""
    records = fly(FaultDetectingKalmanFilterNavigation, "total_loss", target="doppler_velocimeter")
    onset = torch.stack([active for *_, active in records]).float().argmax(dim=0) * CONTROL_STEP
    declared = torch.stack([health[:, 2] for _, health, _, _ in records]).argmax(dim=0) * CONTROL_STEP
    delay = declared - onset
    assert float(delay.min()) >= 0.45 - 1e-6 and float(delay.max()) <= 0.6 + 1e-6
