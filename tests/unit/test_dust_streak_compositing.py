# SPDX-License-Identifier: BSD-3-Clause
"""Dust streaks of the grain video (CPU): cross-section of a parcel, projection, opacity bookkeeping, hiding."""

import math

import numpy

from lunarsim.tools.plume_grain_video import DUST_COLOR, camera_basis, composite_dust_streaks, parcel_cross_sections, project

WIDTH, HEIGHT, FOCAL_PX = 320, 240, 250.0
EYE, TARGET = numpy.array([0.0, 0.0, 0.0]), numpy.array([10.0, 0.0, 0.0])  # looking along +x


def test_parcel_cross_section_is_grain_count_times_extinction_area():
    """A = Q N pi D^2 / 4 with N grains of diameter D and density rho in the parcel mass."""
    mass, diameter, density, efficiency = numpy.array([1e-3]), numpy.array([100e-6]), 3100.0, 2.0
    grains = mass[0] / (density * math.pi / 6.0 * diameter[0] ** 3)
    expected = efficiency * grains * math.pi / 4.0 * diameter[0] ** 2
    assert abs(float(parcel_cross_sections(mass, diameter, density, efficiency)[0]) / expected - 1.0) < 1e-12


def test_projection_puts_the_target_in_the_middle_and_right_to_the_right():
    """The look-at target lands on the image centre; a point to the camera's right lands right of it."""
    basis = camera_basis(EYE, TARGET)
    column, row, depth = project(numpy.array([[10.0, 0.0, 0.0], [10.0, -1.0, 0.5]]), EYE, basis, FOCAL_PX, WIDTH, HEIGHT)
    assert abs(column[0] - WIDTH / 2) < 1e-9 and abs(row[0] - HEIGHT / 2) < 1e-9 and abs(depth[0] - 10.0) < 1e-12
    # Looking along +x with +z up, the camera's right is -y; +z is up, so the row number falls.
    assert abs(column[1] - (WIDTH / 2 + FOCAL_PX * 0.1)) < 1e-9 and abs(row[1] - (HEIGHT / 2 - FOCAL_PX * 0.05)) < 1e-9


def test_a_streak_deposits_its_projected_cross_section():
    """Over open sky, the summed optical depth equals A (f / d)^2 in pixel areas, wherever the streak goes."""
    image = numpy.zeros((HEIGHT, WIDTH, 3), dtype=numpy.uint8)
    sky = numpy.full((HEIGHT, WIDTH), numpy.inf)
    start, end = numpy.array([[20.0, 0.5, 0.0]]), numpy.array([[20.0, -0.5, 0.2]])  # 1 m sideways at 20 m
    cross_section = numpy.array([2e-3])
    new_image, optical_depth = composite_dust_streaks(image, sky, EYE, TARGET, FOCAL_PX, start, end, cross_section)
    expected = 2e-3 * (FOCAL_PX / 20.0) ** 2
    assert abs(optical_depth.sum() / expected - 1.0) < 1e-6
    # Opacity 1 - exp(-tau) of the dust colour on a black image.
    brightest = numpy.unravel_index(optical_depth.argmax(), optical_depth.shape)
    assert abs(int(new_image[brightest][0]) - round(DUST_COLOR[0] * (1.0 - math.exp(-optical_depth[brightest])))) <= 1


def test_streaks_behind_the_scene_are_hidden_and_gain_scales_opacity():
    """A wall at 10 m hides a parcel at 20 m; in front of it, a gain of 100 darkens more than a gain of 1."""
    image = numpy.zeros((HEIGHT, WIDTH, 3), dtype=numpy.uint8)
    wall = numpy.full((HEIGHT, WIDTH), 10.0)
    behind = composite_dust_streaks(image, wall, EYE, TARGET, FOCAL_PX, numpy.array([[20.0, 0.2, 0.0]]), numpy.array([[20.0, -0.2, 0.0]]), numpy.array([1e-2]))
    assert behind[1].sum() == 0.0 and int(behind[0].max()) == 0
    start, end, cross_section = numpy.array([[5.0, 0.2, 0.0]]), numpy.array([[5.0, -0.2, 0.0]]), numpy.array([1e-5])
    faint, _ = composite_dust_streaks(image, wall, EYE, TARGET, FOCAL_PX, start, end, cross_section, gain=1.0)
    boosted, _ = composite_dust_streaks(image, wall, EYE, TARGET, FOCAL_PX, start, end, cross_section, gain=100.0)
    assert int(boosted.max()) > int(faint.max())
