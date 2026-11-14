# SPDX-License-Identifier: BSD-3-Clause
"""Smoke and physics check (GPU): the Apollo 17 surface, its targets and its height lookup.

Needs the height-map files (``python scripts/download_height_maps.py``); skipped without them.

32 landers are dropped with the engine off from 1.5 m above the ground at their own targets.
The height is measured straight below the centre of mass; the ground under the 9 m footprint
varies by up to about 1 m, so a lower drop could start a footpad inside the terrain.
Craters and rocks are switched off here, so the ground under each lander is the smooth 2 m
measured surface and every lander can come to rest on four legs.

Pass criteria and why:
- Every environment gets its own target inside the target square, at the ground height there.
- Right after the reset, the observed height above ground equals the commanded 1.5 m.
  The start slot places the lander relative to the target, whose height is the lookup.
- At rest after 8 s, at least three footpads of every lander touch the looked-up ground: the
  distance from the centre of the dish's sphere to the local ground plane equals the sphere
  radius within 3 cm (the contact offset is 2 cm). This compares the PhysX collision surface
  with the lookup at 96 or more points. (The height above ground itself is measured straight
  below the centre of mass, so on uneven ground it can differ from zero at rest; it is printed.)
- Every lander settles upright to within 15 degrees: it rests on the measured slope, not on
  a phantom surface below or above it.
- A second reset moves the targets: they are drawn again.
"""

import math

import pytest
import torch

pytest.importorskip("isaaclab")
from isaac_test_support import make_actions, make_environment, read_truth, set_start_states  # noqa: E402

from lunarsim.core.binding import SlotName  # noqa: E402
from lunarsim.core.message_types import MessageKey  # noqa: E402
from lunarsim.core.rotation_math import body_up_axis_in_parent, tilt_angle_from_vertical  # noqa: E402
from lunarsim.slots.surface.measured_terrain import resolve_repository_path  # noqa: E402

COUNT = 32
DROP_HEIGHT = 1.5
DATA = resolve_repository_path("assets/terrain/apollo17/NAC_DTM_APOLLO17_2M.TIF")
pytestmark = pytest.mark.skipif(not DATA.is_file(), reason="height maps not downloaded")


@pytest.fixture(scope="module")
def dropped():
    """Drop the landers on the smooth measured surface and record the start and the rest state."""
    environment = make_environment(
        "apollo17_terminal_truth",
        [
            f"num_envs={COUNT}",
            "metrics=none",
            "actuators.initial_throttle=0.0",
            "surface.craters.enabled=false",
            "surface.rocks.enabled=false",
        ],
    )
    surface = environment.bindings[SlotName.SURFACE]
    observations = set_start_states(environment, altitude=DROP_HEIGHT)
    first_targets = surface.geometry.site_origin.clone()
    start_height = observations["policy"][:, 14].clone()
    for _ in range(int(8.0 / environment.clock.control_step_duration)):
        observations, _, _, _, _ = environment.step(make_actions(environment, throttle=0.0))
    truth = read_truth(environment)
    plant = environment.bindings[SlotName.PLANT]
    preset = plant.preset
    leg_ids = [plant.asset.find_bodies(f"leg_{index + 1}")[0][0] for index in range(preset.number_of_legs)]
    leg_origins = plant.asset.data.body_link_pos_w.torch[:, leg_ids]  # (N, 4, 3)
    body_up = body_up_axis_in_parent(truth.orientation_quaternion).unsqueeze(1)
    # Centre of the sphere whose cap is the dish: (R - depth) above the dish bottom, i.e. above the leg origin.
    sphere_centres = leg_origins + (preset.footpad_sphere_radius - preset.footpad_depth) * body_up
    ground = surface.height_field.height_at(sphere_centres[..., 0], sphere_centres[..., 1])
    slope_x, slope_y = surface.height_field.gradient_at(sphere_centres[..., 0], sphere_centres[..., 1])
    normal_z = 1.0 / torch.sqrt(1.0 + slope_x**2 + slope_y**2)
    footpad_gap = (sphere_centres[..., 2] - ground) * normal_z - preset.footpad_sphere_radius
    rest = {
        "footpad_gap": footpad_gap,
        "height": truth.footpad_height_above_ground.clone(),
        "tilt": tilt_angle_from_vertical(truth.orientation_quaternion),
        "speed": truth.velocity.norm(dim=1),
        "body_contact": truth.body_contact.clone(),
    }
    environment.reset()
    second_targets = surface.geometry.site_origin.clone()
    yield environment, surface, first_targets, second_targets, start_height, rest
    environment.close()


def test_targets_lie_on_the_ground_inside_the_square(dropped):
    """Each target is inside the target square and at the looked-up ground height."""
    _, surface, first_targets, _, _, _ = dropped
    assert float(first_targets[:, :2].abs().max()) <= surface.target_half_size + 1e-4
    ground = surface.height_field.height_at(first_targets[:, 0], first_targets[:, 1])
    assert torch.allclose(first_targets[:, 2], ground)
    assert float(torch.cdist(first_targets[:, :2], first_targets[:, :2]).sum()) > 0.0


def test_observed_height_starts_at_the_commanded_altitude(dropped):
    """The 15th observation value is the drop height right after the reset."""
    _, _, _, _, start_height, _ = dropped
    assert torch.allclose(start_height, torch.full_like(start_height, DROP_HEIGHT), atol=1e-3)


def test_landers_rest_on_the_collision_surface_of_the_lookup(dropped):
    """At rest three or more dishes per lander touch the looked-up ground within 3 cm; upright; no body strike."""
    _, _, _, _, _, rest = dropped
    print("rest heights above ground", [round(value, 3) for value in rest["height"].tolist()])
    print("footpad gaps", rest["footpad_gap"].tolist())
    three_lowest = rest["footpad_gap"].sort(dim=1).values[:, :3]
    assert float(three_lowest.abs().max()) < 0.03
    assert float(rest["tilt"].max()) < math.radians(15.0)
    assert float(rest["speed"].max()) < 0.05
    assert float(rest["body_contact"].max()) == 0.0


def test_targets_are_drawn_again_at_reset(dropped):
    """A reset gives new targets."""
    _, _, first_targets, second_targets, _, _ = dropped
    assert float((first_targets - second_targets)[:, :2].norm(dim=1).min()) > 0.0


def test_geometry_message_is_on_the_bus(dropped):
    """The surface publishes its geometry under the declared key."""
    environment, surface, _, _, _, _ = dropped
    assert environment.bus.has(MessageKey.SURFACE_GEOMETRY)
    assert surface.geometry.height_field is surface.height_field
