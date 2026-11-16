# SPDX-License-Identifier: BSD-3-Clause
"""The dust fault (milestone 5): optical sensors degrade with the dust density, radar does not."""

import torch
from helpers import make_binding_context

from lunarsim.core.message_types import Measurement, MessageKey, RegolithState
from lunarsim.slots.faults.dust import DustFault
from lunarsim.slots.sensors.terminal_suite import DOPPLER_VELOCIMETER_KEY, LIDAR_KEY


class _Writer:
    """Test-side writer of the sensor and regolith messages."""

    provides = frozenset({LIDAR_KEY, DOPPLER_VELOCIMETER_KEY, MessageKey.REGOLITH_STATE})
    requires = frozenset({LIDAR_KEY, DOPPLER_VELOCIMETER_KEY})
    reads_truth = False


def run_fault(density: float, count: int = 200, pixels: int = 100):
    """Return the degraded LiDAR and Doppler messages for a constant dust density."""
    context = make_binding_context(count)
    context.shared_data["sensor_suite_parameters"] = {"lidar": {"minimum_range_m": 2.8}}
    fault = DustFault({}, context)
    generator = torch.Generator().manual_seed(0)
    fault.reset(torch.arange(count), generator)
    writer, bus = _Writer(), context.bus
    regolith = RegolithState.clear(count, 4, "cpu")
    regolith.dust_density[:] = density
    bus.write(MessageKey.REGOLITH_STATE, regolith, writer)
    ranges = torch.full((count, pixels), 50.0)
    bus.write(LIDAR_KEY, Measurement(ranges, torch.ones(count, dtype=torch.bool), torch.zeros(count), torch.ones(count, pixels)), writer)
    doppler = torch.cat((torch.full((count, 3), -1.0), torch.full((count, 3), 40.0)), dim=1)
    bus.write(DOPPLER_VELOCIMETER_KEY, Measurement(doppler, torch.ones(count, dtype=torch.bool), torch.zeros(count), torch.ones(count, 6)), writer)
    fault.step(bus, 0.02)
    return bus.read(LIDAR_KEY, writer), bus.read(DOPPLER_VELOCIMETER_KEY, writer)


def test_clear_air_leaves_the_sensors_alone():
    """With no dust every LiDAR pixel and Doppler beam is unchanged."""
    lidar, doppler = run_fault(0.0)
    assert bool(torch.isfinite(lidar.value).all()) and bool((lidar.value == 50.0).all())
    assert bool(torch.isfinite(doppler.value).all())


def test_full_dust_loses_most_pixels_and_gives_some_short_returns():
    """At full dust about 90 % of pixels are lost and about 5 % return short (between 2.8 m and the true range)."""
    lidar, doppler = run_fault(1.0)
    lost = torch.isnan(lidar.value).float().mean()
    short = ((lidar.value < 50.0) & torch.isfinite(lidar.value)).float().mean()
    assert 0.87 < float(lost) < 0.93
    assert 0.03 < float(short) < 0.07
    returns = lidar.value[torch.isfinite(lidar.value)]
    assert float(returns.min()) >= 2.8
    beams_lost = torch.isnan(doppler.value[:, :3]).float().mean()
    assert 0.4 < float(beams_lost) < 0.6
    # Velocity and range of a beam are lost together.
    assert bool((torch.isnan(doppler.value[:, :3]) == torch.isnan(doppler.value[:, 3:])).all())
