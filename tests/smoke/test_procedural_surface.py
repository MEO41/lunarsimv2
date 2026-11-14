# SPDX-License-Identifier: BSD-3-Clause
"""Smoke test (GPU): S1 with ``surface=procedural`` runs, and landers come to rest on the rough ground."""

import pytest
import torch
from isaac_test_support import make_actions, make_environment, read_truth, set_start_states

from lunarsim.core.binding import SlotName
from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY


@pytest.fixture(scope="module")
def environment():
    """Build S1 on the procedural surface with 64 environments and the engine off."""
    environment = make_environment(
        "s1_terminal_truth", ["num_envs=64", "surface=procedural", "metrics=none", "actuators.initial_throttle=0.0"]
    )
    yield environment
    environment.close()


def test_landers_rest_on_rough_ground(environment):
    """Dropped from 1 m, every lander rests near the mean ground level, carried by its legs.

    Rest height: the centre of mass sits 3.62 m above the footpads, minus about 4 cm of strut
    compression, on ground that is within +-0.15 m of zero: so between 3.3 m and 3.9 m.
    The leg forces, averaged over the last second, sum to the weight within 3 %: the ground is
    not level, so the legs share the load unequally, and the sensor reports only the normal part
    of each contact force; on facets tilted by a few degrees, friction carries the rest. A
    single sample also has about 1.5 % solver noise, so the sum is averaged.
    """
    set_start_states(environment, altitude=1.0)
    leg_force_sums = []
    for step in range(500):
        environment.step(make_actions(environment))
        if step >= 450:
            leg_force_sums.append(read_truth(environment).contact.sum(dim=1))
    truth = read_truth(environment)
    preset = environment.bindings[SlotName.PLANT].preset
    weight = preset.mass_at_start * MOON_SURFACE_GRAVITY
    assert torch.isfinite(truth.position).all()
    assert float(truth.position[:, 2].min()) > 3.3 and float(truth.position[:, 2].max()) < 3.9
    assert float((torch.stack(leg_force_sums).mean(dim=0) / weight - 1.0).abs().max()) < 0.03
    assert float(truth.velocity.norm(dim=1).max()) < 0.02
    # The ground is really rough: rest heights differ between environments.
    assert float(truth.position[:, 2].std()) > 0.01
