# SPDX-License-Identifier: BSD-3-Clause
"""Small craters and rocks that a 2 m height map cannot show, drawn from published statistics.

Populations (every constant comes from the surface config file, with its source there):

- Craters: cumulative number per square metre N(>D) = k D^-b, between ``minimum`` and
  ``maximum`` diameter. Default k = 0.0336, b = 2 (equilibrium on the Apollo 15 mare, Minton et
  al. 2019); NASA's design upper limit is k = 0.0794 (SLS-SPEC-159, 3.4.1.1).
- Crater depth: one of five degradation classes, each with its share and depth/diameter range
  (SLS-SPEC-159 Table 3.4.1.2-2).
- Rocks: N(>D) = C D^gamma per square metre below 2 m (Surveyor 1: C = 0.00234, gamma = -2.11,
  SLS-SPEC-159 3.4.1.4); height = diameter * N(0.6, 0.28), clipped (same section).

Model choices (no fetched source gives them, so they are parameters and listed as such):

- Crater profile: a parabolic bowl inside the rim, a raised rim of ``rim_height_to_depth`` times
  the depth, and an ejecta blanket that falls off as (R / r)^3 to zero at three radii. Craters
  of the last (most worn) class have no rim.
- Rock profile: half of an ellipsoid (a dome) on top of the local ground; rocks cannot roll.
- Shapes add up where they overlap.

All of this is built once per run on the CPU with numpy; a loop over craters and rocks is fine
here (it is not a loop over environments).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy


@dataclass(frozen=True)
class CraterPopulation:
    """Size and depth statistics of small craters."""

    cumulative_coefficient_per_m2: float  # k in N(>D) = k D^-b, D in metres
    cumulative_exponent: float  # b
    minimum_diameter_m: float
    maximum_diameter_m: float
    class_shares: tuple[float, ...]  # share of each degradation class
    class_depth_to_diameter: tuple[tuple[float, float], ...]  # (low, high) per class
    rim_height_to_depth: float  # model choice
    classes_without_rim: tuple[int, ...]  # indices of classes that have no rim

    def expected_count(self, area_m2: float) -> float:
        """Return the mean number of craters between the minimum and maximum diameter in ``area_m2``."""
        b = self.cumulative_exponent
        return area_m2 * self.cumulative_coefficient_per_m2 * (
            self.minimum_diameter_m ** (-b) - self.maximum_diameter_m ** (-b)
        )


@dataclass(frozen=True)
class RockPopulation:
    """Size and shape statistics of rocks."""

    cumulative_coefficient_per_m2: float  # C in N(>D) = C D^gamma
    cumulative_exponent: float  # gamma (negative)
    minimum_diameter_m: float
    maximum_diameter_m: float
    height_to_diameter_mean: float
    height_to_diameter_standard_deviation: float
    height_to_diameter_limits: tuple[float, float]  # clip range of the drawn ratio (model choice)

    def expected_count(self, area_m2: float) -> float:
        """Return the mean number of rocks between the minimum and maximum diameter in ``area_m2``."""
        exponent = self.cumulative_exponent
        return area_m2 * self.cumulative_coefficient_per_m2 * (
            self.minimum_diameter_m**exponent - self.maximum_diameter_m**exponent
        )


@dataclass
class Craters:
    """Drawn craters: centres, diameters, depths and rim heights (one entry per crater)."""

    center_x: numpy.ndarray
    center_y: numpy.ndarray
    diameter: numpy.ndarray
    depth: numpy.ndarray
    rim_height: numpy.ndarray
    degradation_class: numpy.ndarray


@dataclass
class Rocks:
    """Drawn rocks: centres, diameters and heights."""

    center_x: numpy.ndarray
    center_y: numpy.ndarray
    diameter: numpy.ndarray
    height: numpy.ndarray


def draw_power_law_sizes(
    random: numpy.random.Generator, count: int, minimum: float, maximum: float, cumulative_exponent: float
) -> numpy.ndarray:
    """Draw sizes whose cumulative number N(>D) is proportional to D^-b between minimum and maximum.

    ``cumulative_exponent`` is b > 0. Inverse transform: D = (Dmin^-b - u (Dmin^-b - Dmax^-b))^(-1/b).
    """
    b = cumulative_exponent
    uniform = random.random(count)
    return (minimum ** (-b) - uniform * (minimum ** (-b) - maximum ** (-b))) ** (-1.0 / b)


def draw_craters(
    random: numpy.random.Generator, population: CraterPopulation, bounds: tuple[float, float, float, float]
) -> Craters:
    """Draw a Poisson number of craters with centres uniform in ``bounds = (x0, x1, y0, y1)``."""
    x0, x1, y0, y1 = bounds
    count = int(random.poisson(population.expected_count((x1 - x0) * (y1 - y0))))
    diameter = draw_power_law_sizes(
        random, count, population.minimum_diameter_m, population.maximum_diameter_m, population.cumulative_exponent
    )
    shares = numpy.asarray(population.class_shares, dtype=numpy.float64)
    degradation_class = random.choice(len(shares), size=count, p=shares / shares.sum())
    ranges = numpy.asarray(population.class_depth_to_diameter, dtype=numpy.float64)
    ratio = ranges[degradation_class, 0] + random.random(count) * (ranges[degradation_class, 1] - ranges[degradation_class, 0])
    depth = ratio * diameter
    has_rim = ~numpy.isin(degradation_class, population.classes_without_rim)
    return Craters(
        center_x=random.uniform(x0, x1, count),
        center_y=random.uniform(y0, y1, count),
        diameter=diameter,
        depth=depth,
        rim_height=numpy.where(has_rim, population.rim_height_to_depth * depth, 0.0),
        degradation_class=degradation_class,
    )


def draw_rocks(random: numpy.random.Generator, population: RockPopulation, bounds: tuple[float, float, float, float]) -> Rocks:
    """Draw a Poisson number of rocks with centres uniform in ``bounds = (x0, x1, y0, y1)``."""
    x0, x1, y0, y1 = bounds
    count = int(random.poisson(population.expected_count((x1 - x0) * (y1 - y0))))
    diameter = draw_power_law_sizes(
        random, count, population.minimum_diameter_m, population.maximum_diameter_m, -population.cumulative_exponent
    )
    low, high = population.height_to_diameter_limits
    ratio = numpy.clip(
        random.normal(population.height_to_diameter_mean, population.height_to_diameter_standard_deviation, count), low, high
    )
    return Rocks(
        center_x=random.uniform(x0, x1, count),
        center_y=random.uniform(y0, y1, count),
        diameter=diameter,
        height=ratio * diameter,
    )


def crater_profile(normalized_radius: numpy.ndarray, depth: float, rim_height: float) -> numpy.ndarray:
    """Return the height change of one crater at r / R (model choice, see the module docstring)."""
    inside = rim_height - (depth + rim_height) * (1.0 - normalized_radius**2)
    outside_limit = 3.0
    falloff = (normalized_radius.clip(1.0, None) ** -3 - outside_limit**-3) / (1.0 - outside_limit**-3)
    outside = rim_height * falloff
    return numpy.where(normalized_radius < 1.0, inside, numpy.where(normalized_radius < outside_limit, outside, 0.0))


def rock_profile(normalized_radius: numpy.ndarray, height: float) -> numpy.ndarray:
    """Return the height of one rock dome at r / R (half ellipsoid, model choice)."""
    return height * numpy.sqrt(numpy.clip(1.0 - normalized_radius**2, 0.0, None))


def _window(
    center: float, reach: float, origin: float, cell_size: float, count: int
) -> tuple[int, int]:
    """Return the index range [first, last) of grid lines within ``reach`` of ``center``."""
    first = max(int(numpy.floor((center - reach - origin) / cell_size)), 0)
    last = min(int(numpy.ceil((center + reach - origin) / cell_size)) + 1, count)
    return first, last


def surface_detail_heights(
    origin_x: float,
    origin_y: float,
    cell_size: float,
    shape: tuple[int, int],
    craters: Craters,
    rocks: Rocks,
) -> numpy.ndarray:
    """Return the height change (ny, nx) that the craters and rocks add to a grid."""
    number_y, number_x = shape
    detail = numpy.zeros(shape, dtype=numpy.float64)
    for center_x, center_y, diameter, depth, rim_height in zip(
        craters.center_x, craters.center_y, craters.diameter, craters.depth, craters.rim_height
    ):
        radius = 0.5 * diameter
        first_i, last_i = _window(center_x, 3.0 * radius, origin_x, cell_size, number_x)
        first_j, last_j = _window(center_y, 3.0 * radius, origin_y, cell_size, number_y)
        if first_i >= last_i or first_j >= last_j:
            continue
        x = origin_x + numpy.arange(first_i, last_i) * cell_size
        y = origin_y + numpy.arange(first_j, last_j) * cell_size
        distance = numpy.hypot(x[numpy.newaxis, :] - center_x, y[:, numpy.newaxis] - center_y)
        detail[first_j:last_j, first_i:last_i] += crater_profile(distance / radius, depth, rim_height)
    for center_x, center_y, diameter, height in zip(rocks.center_x, rocks.center_y, rocks.diameter, rocks.height):
        radius = 0.5 * diameter
        first_i, last_i = _window(center_x, radius, origin_x, cell_size, number_x)
        first_j, last_j = _window(center_y, radius, origin_y, cell_size, number_y)
        if first_i >= last_i or first_j >= last_j:
            continue
        x = origin_x + numpy.arange(first_i, last_i) * cell_size
        y = origin_y + numpy.arange(first_j, last_j) * cell_size
        distance = numpy.hypot(x[numpy.newaxis, :] - center_x, y[:, numpy.newaxis] - center_y)
        detail[first_j:last_j, first_i:last_i] += rock_profile(distance / radius, height)
    return detail
