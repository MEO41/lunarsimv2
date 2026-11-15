# SPDX-License-Identifier: BSD-3-Clause
"""Surface probes (GPU): the contact light comes on when a probe tip reaches the ground.

Known answer: three probes, 1.727 m long, hang along body -z from the lowest point of the dishes
of legs 2, 3 and 4 (none on leg 1, the ladder leg; NASA TN D-6850 p. 8, Rogers 1973 p. 124).
On flat ground the light must be on exactly when the lowest probe tip is at or below z = 0.

Cases, each checked right after a reset (struts extended, nothing has moved yet):
- upright landers at footpad heights from 1.60 m to 1.85 m: on below 1.727 m, off above;
- landers tilted about body y so that the ladder leg points down: its missing probe must not
  switch the light on, while tilting the other way (leg 3 down) does;
- the observation column ``SURFACE_PROBE_CONTACT`` carries the same signal to the agents.

Expected tip heights are computed here from the preset geometry and the start pose, not from
PhysX, so the test also checks that the leg links sit where the preset puts them.

Tolerance and why: environments whose expected lowest tip lies within 5 mm of the ground are
not judged (float32 positions and the reset write).
"""

import math

import pytest
import torch
from isaac_test_support import make_environment, read_truth, set_start_states

from lunarsim.core.message_types import NavigationStateLayout
from lunarsim.core.rotation_math import quaternion_from_axis_angle, rotate_vector_body_to_parent

COUNT = 16
UNDECIDED_BAND_M = 0.005


def expected_lowest_probe_tip(preset, position: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
    """Return the world height of the lowest probe tip for each environment, from the preset alone."""
    offset = torch.tensor(preset.surface_probe_tip_offset_from_leg_link, device=position.device)
    tips = []
    for leg in preset.legs_with_surface_probes:
        leg_position = torch.tensor(preset.leg_positions[leg - 1], device=position.device)
        tip_body = (leg_position + offset).expand(position.shape[0], 3)
        tips.append(position[:, 2] + rotate_vector_body_to_parent(orientation, tip_body)[:, 2])
    return torch.stack(tips, dim=1).min(dim=1).values


@pytest.fixture(scope="module")
def environment():
    """One environment set for all cases: flat ground, engine off, no metrics."""
    environment = make_environment(
        "s1_terminal_truth", [f"num_envs={COUNT}", "metrics=none", "actuators.initial_throttle=0.0"]
    )
    yield environment
    environment.close()


def check_light(environment, observations):
    """Compare the plant's light and the observation with the geometric answer; return the light."""
    preset = environment.bindings["plant"].preset
    truth = read_truth(environment)
    # Flat surface: the SITE origin lies on the ground (z = 0), so SITE heights are ground heights.
    lowest_tip = expected_lowest_probe_tip(preset, truth.position, truth.orientation_quaternion)
    light = truth.surface_probe_contact
    decided = lowest_tip.abs() > UNDECIDED_BAND_M
    print("lowest tip", lowest_tip.tolist(), "light", light.tolist())
    assert torch.equal(light[decided] > 0.5, lowest_tip[decided] <= 0.0)
    observed = observations["policy"][:, NavigationStateLayout.SURFACE_PROBE_CONTACT].squeeze(1)
    assert torch.equal(observed, light)
    return light, lowest_tip


def test_light_switches_at_the_probe_length_when_upright(environment):
    """Upright: on at footpad heights below 1.727 m, off above."""
    heights = torch.linspace(1.60, 1.85, COUNT, device=environment.device)
    observations = set_start_states(environment, altitude=heights)
    light, _ = check_light(environment, observations)
    assert bool((light[heights < 1.70] > 0.5).all())
    assert bool((light[heights > 1.75] < 0.5).all())


@pytest.mark.parametrize("tilt_deg, light_expected", [(5.0, False), (-5.0, True)])
def test_ladder_leg_has_no_probe(environment, tilt_deg, light_expected):
    """Tilted 5 degrees about body y at 1.9 m: leg 1 down gives no light, leg 3 down does."""
    count, device = environment.num_envs, environment.device
    set_start_states(environment, altitude=1.9)
    start = environment.bindings["start"]
    position, velocity, _ = start.fixed_start_states
    axis = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(count, 1)
    orientation = quaternion_from_axis_angle(axis, torch.full((count,), math.radians(tilt_deg), device=device))
    start.set_fixed_start_states(position, velocity, orientation)
    observations, _ = environment.reset()
    light, lowest_tip = check_light(environment, observations)
    assert bool(((light > 0.5) == light_expected).all())
    if not light_expected:
        # The ladder leg's dish bottom is lower than the probe-leg tips would need: a probe there
        # would have switched the light on.
        preset = environment.bindings["plant"].preset
        ladder_leg_dish_bottom = 1.9 - preset.footpad_radius_from_thrust_axis * math.sin(math.radians(tilt_deg))
        assert ladder_leg_dish_bottom < preset.surface_probe_length
