# SPDX-License-Identifier: BSD-3-Clause
"""Frame changes MCI <-> MCMF <-> SITE (milestone 6; design test "MCI -> MCMF -> SITE -> back gives the start state")."""

import math

import pytest
import torch
from helpers import make_truth_state

from lunarsim.core.frames import SiteLocation, convert_truth_state, quaternion_from_matrix, specific_orbital_energy
from lunarsim.core.message_types import Frame, TruthState
from lunarsim.core.physical_constants import MOON_GRAVITATIONAL_PARAMETER, MOON_MEAN_RADIUS, MOON_ROTATION_RATE
from lunarsim.core.rotation_math import quaternion_from_axis_angle

SITE = SiteLocation(latitude_deg=20.19091, longitude_deg=30.77168, radius_m=MOON_MEAN_RADIUS - 2000.0)


def inertial_state(count: int = 5) -> TruthState:
    """Return float64 MCI states at different times with random orientations."""
    generator = torch.Generator().manual_seed(3)
    position = torch.randn(count, 3, generator=generator, dtype=torch.float64) * 1.0e5 + torch.tensor([1.8e6, 0.0, 0.0], dtype=torch.float64)
    velocity = torch.randn(count, 3, generator=generator, dtype=torch.float64) * 100.0 + torch.tensor([0.0, 1600.0, 0.0], dtype=torch.float64)
    axis = torch.nn.functional.normalize(torch.randn(count, 3, generator=generator, dtype=torch.float64), dim=1)
    orientation = quaternion_from_axis_angle(axis, torch.rand(count, generator=generator, dtype=torch.float64) * 3.0)
    return TruthState(
        frame=Frame.MOON_CENTERED_INERTIAL, time=torch.linspace(0.0, 7200.0, count, dtype=torch.float64), position=position,
        velocity=velocity, orientation_quaternion=orientation, angular_velocity=torch.zeros(count, 3, dtype=torch.float64),
        mass=torch.full((count,), 1.0e4, dtype=torch.float64), contact=torch.zeros(count, 4, dtype=torch.float64),
    )


def test_same_frame_conversion_returns_the_same_state():
    """SITE -> SITE changes nothing."""
    state = make_truth_state()
    assert convert_truth_state(state, Frame.SITE, SITE) is state


def test_mci_mcmf_site_and_back_gives_the_start_state():
    """MCI -> MCMF -> SITE -> MCMF -> MCI returns the same position, velocity and orientation."""
    start = inertial_state()
    site_state = convert_truth_state(convert_truth_state(start, Frame.MOON_CENTERED_MOON_FIXED), Frame.SITE, SITE)
    back = convert_truth_state(convert_truth_state(site_state, Frame.MOON_CENTERED_MOON_FIXED, SITE), Frame.MOON_CENTERED_INERTIAL)
    assert torch.allclose(back.position, start.position, atol=1e-6)
    assert torch.allclose(back.velocity, start.velocity, atol=1e-9)
    same_rotation = (back.orientation_quaternion * start.orientation_quaternion).sum(dim=1).abs()
    assert torch.allclose(same_rotation, torch.ones_like(same_rotation), atol=1e-12)


def test_site_origin_and_up_axis():
    """The site origin maps to SITE (0, 0, 0); a point 100 m above it to (0, 0, 100)."""
    origin = SITE.origin_mcmf()
    up = SITE.east_north_up()[:, 2]
    state = TruthState(
        frame=Frame.MOON_CENTERED_MOON_FIXED, time=torch.zeros(2, dtype=torch.float64),
        position=torch.stack((origin, origin + 100.0 * up)), velocity=torch.zeros(2, 3, dtype=torch.float64),
        orientation_quaternion=torch.tensor([[1.0, 0, 0, 0]] * 2, dtype=torch.float64), angular_velocity=torch.zeros(2, 3, dtype=torch.float64),
        mass=torch.ones(2, dtype=torch.float64), contact=torch.zeros(2, 4, dtype=torch.float64),
    )
    site_state = convert_truth_state(state, Frame.SITE, SITE)
    assert torch.allclose(site_state.position, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]], dtype=torch.float64), atol=1e-6)


def test_a_point_fixed_to_the_moon_moves_in_mci_at_the_rotation_rate():
    """A body at rest in MCMF has the inertial speed omega * distance from the spin axis."""
    position = torch.tensor([[MOON_MEAN_RADIUS, 0.0, 0.0]], dtype=torch.float64)
    state = TruthState(
        frame=Frame.MOON_CENTERED_MOON_FIXED, time=torch.zeros(1, dtype=torch.float64), position=position,
        velocity=torch.zeros(1, 3, dtype=torch.float64), orientation_quaternion=torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64),
        angular_velocity=torch.zeros(1, 3, dtype=torch.float64), mass=torch.ones(1, dtype=torch.float64), contact=torch.zeros(1, 4, dtype=torch.float64),
    )
    inertial = convert_truth_state(state, Frame.MOON_CENTERED_INERTIAL)
    assert abs(float(inertial.velocity[0, 1]) - MOON_ROTATION_RATE * MOON_MEAN_RADIUS) < 1e-9


def test_energy_is_the_same_whatever_frame_the_state_is_given_in():
    """Specific energy computed from MCI, MCMF and SITE copies of one state agrees."""
    start = inertial_state()
    fixed = convert_truth_state(start, Frame.MOON_CENTERED_MOON_FIXED)
    site_state = convert_truth_state(fixed, Frame.SITE, SITE)
    energies = [specific_orbital_energy(state, SITE) for state in (start, fixed, site_state)]
    assert torch.allclose(energies[0], energies[1], rtol=1e-12) and torch.allclose(energies[0], energies[2], rtol=1e-12)
    circular = math.sqrt(MOON_GRAVITATIONAL_PARAMETER / 1.8e6)
    assert circular > 1600.0  # the test orbit is near circular at 1,800 km


def test_quaternion_from_matrix_handles_every_branch():
    """Rotations of 0, 90 and 180 deg about each axis round-trip through the matrix."""
    from lunarsim.core.rotation_math import rotate_vector_body_to_parent

    for angle in (0.0, math.pi / 2.0, math.pi):
        for axis in torch.eye(3, dtype=torch.float64):
            quaternion = quaternion_from_axis_angle(axis.unsqueeze(0), torch.tensor([angle], dtype=torch.float64))
            matrix = torch.stack([rotate_vector_body_to_parent(quaternion, basis.unsqueeze(0))[0] for basis in torch.eye(3, dtype=torch.float64)], dim=1)
            back = quaternion_from_matrix(matrix.unsqueeze(0))
            assert abs(abs(float((back * quaternion).sum())) - 1.0) < 1e-9


def test_site_changes_need_the_site_and_frames_must_be_frames():
    """A SITE change without a SiteLocation fails; a string target fails."""
    with pytest.raises(ValueError):
        convert_truth_state(make_truth_state(), Frame.MOON_CENTERED_INERTIAL)
    with pytest.raises(TypeError):
        convert_truth_state(make_truth_state(), "SITE")
