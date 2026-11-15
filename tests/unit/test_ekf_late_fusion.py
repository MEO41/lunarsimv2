# SPDX-License-Identifier: BSD-3-Clause
"""Late fusion (CPU, milestone 3, S4): Doppler samples arrive 100-200 ms late on the synthetic flight.

Three ways to handle late data (design S4): use it as if it were new (plain ``ekf``), drop it
(``ekf_fdi``'s age check), and apply it at its capture time (``ekf_late_fusion``). Known answers:
- Without delays, ``ekf_late_fusion`` gives exactly the ``ekf`` estimate (it never replays).
- With delays, late fusion keeps the velocity error close to the no-delay filter (within a
  factor 1.5 at the 95th percentile) and clearly below the filter that treats late data as new.
- Its covariance stays consistent: mean NEES (9 components) over the last 6 s between 6 and 12.
"""

import pytest
import torch
import yaml
from helpers import make_binding_context
from test_ekf_consistency import CONFIG, COUNT, CONTROL_STEP, WORLD, load, normalised_error, truth_at

from lunarsim.assets.lander_preset import load_lander_preset
from lunarsim.core.message_types import MessageKey, SurfaceGeometry
from lunarsim.core.rotation_math import quaternion_conjugate, quaternion_multiply, quaternion_to_rotation_vector
from lunarsim.slots.faults.delayed_delivery import DelayedDelivery
from lunarsim.slots.navigation.ekf import ExtendedKalmanFilterNavigation
from lunarsim.slots.navigation.ekf_fdi import FaultDetectingKalmanFilterNavigation
from lunarsim.slots.navigation.ekf_late_fusion import LateFusionKalmanFilterNavigation
from lunarsim.slots.sensors.terminal_suite import TerminalSensorSuite
from lunarsim.slots.start.initial_navigation_error import InitialNavigationError

DURATION_S = 10.0


def fly(navigation_class, delayed: bool):
    """Fly the synthetic trajectory with (optionally) late Doppler samples; return per-step errors and covariances."""
    context = make_binding_context(COUNT)
    preset = load_lander_preset(CONFIG / "landers" / "apollo_lunar_module.yaml")
    suite_parameters = load("sensors/terminal_suite.yaml")
    context.shared_data.update({"lander_preset": preset, "sensor_suite_parameters": suite_parameters, "control_step_duration": CONTROL_STEP})
    suite = TerminalSensorSuite(suite_parameters, context)
    with open(CONFIG / "nav" / "ekf.yaml") as file:
        navigation_parameters = yaml.safe_load(file)
    navigation_parameters["detection"] = load("nav/ekf_fdi.yaml")["detection"]
    navigation_parameters["late_fusion"] = {"buffer_seconds": 0.6}
    navigation = navigation_class(navigation_parameters, context)
    delay = DelayedDelivery({"delay_ms": {"doppler_velocimeter": [100, 200]}, "jitter_ms": 20}, context) if delayed else None
    bus = context.bus
    bus.write(MessageKey.SURFACE_GEOMETRY, SurfaceGeometry(site_origin=torch.zeros(COUNT, 3)), WORLD)
    error = InitialNavigationError({"position_m": 1.0, "velocity_mps": 0.2, "attitude_deg": 0.1}, COUNT, "cpu")
    generator = torch.Generator().manual_seed(7)
    everyone = torch.arange(COUNT)
    error.reset(everyone, generator)
    bus.write(MessageKey.INITIAL_NAVIGATION_STATE, error.estimate(truth_at(0.0)), WORLD)
    suite.reset(everyone, generator)
    if delay is not None:
        delay.reset(everyone, torch.Generator().manual_seed(11))
    navigation.reset(everyone, generator)
    records = []
    for step in range(int(DURATION_S / CONTROL_STEP) + 1):
        truth = truth_at(step * CONTROL_STEP)
        bus.write(MessageKey.TRUTH_STATE, truth, WORLD)
        suite.step(bus, CONTROL_STEP)
        if delay is not None:
            delay.step(bus, CONTROL_STEP)
        navigation.step(bus, CONTROL_STEP)
        state = navigation.filter
        attitude = quaternion_to_rotation_vector(quaternion_multiply(quaternion_conjugate(state.orientation), truth.orientation_quaternion))
        errors = torch.cat((truth.position - state.position, truth.velocity - state.velocity, attitude), dim=1).to(torch.float64)
        records.append((step * CONTROL_STEP, errors, state.covariance[:, :9, :9].clone()))
    return records, navigation


def velocity_error_p95(records, after_s: float = 4.0) -> float:
    """Return the 95th percentile of the velocity error norm over the steps after ``after_s``."""
    errors = torch.cat([errors[:, 3:6].norm(dim=1) for time, errors, _ in records if time >= after_s])
    return float(errors.quantile(0.95))


def test_without_delays_late_fusion_is_the_ekf():
    """No late samples: identical estimates, no replay."""
    plain, _ = fly(ExtendedKalmanFilterNavigation, delayed=False)
    late, navigation = fly(LateFusionKalmanFilterNavigation, delayed=False)
    assert navigation.replays == 0
    assert torch.allclose(plain[-1][1], late[-1][1], atol=1e-4)  # float32: different rounding paths


def test_late_fusion_beats_treating_late_data_as_new():
    """Velocity error p95: late fusion within 1.5x of no delay, and below the as-if-new filter."""
    reference = velocity_error_p95(fly(ExtendedKalmanFilterNavigation, delayed=False)[0])
    as_new = velocity_error_p95(fly(ExtendedKalmanFilterNavigation, delayed=True)[0])
    dropped = velocity_error_p95(fly(FaultDetectingKalmanFilterNavigation, delayed=True)[0])
    late_records, navigation = fly(LateFusionKalmanFilterNavigation, delayed=True)
    late = velocity_error_p95(late_records)
    print(f"velocity error p95 (m/s): no delay {reference:.4f}, as new {as_new:.4f}, dropped {dropped:.4f}, late fusion {late:.4f}; replays {navigation.replays}")
    assert navigation.replays > 0
    assert late < 1.5 * reference
    assert late < as_new


def test_late_fusion_stays_consistent():
    """Mean NEES of position, velocity and attitude over the last 6 s between 6 and 12."""
    records, _ = fly(LateFusionKalmanFilterNavigation, delayed=True)
    values = [normalised_error(errors, covariance).mean() for time, errors, covariance in records if time >= DURATION_S - 6.0]
    mean = float(torch.stack(values).mean())
    print("late fusion mean NEES", round(mean, 2))
    assert 6.0 < mean < 12.0
