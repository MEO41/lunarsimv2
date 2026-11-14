# SPDX-License-Identifier: BSD-3-Clause
"""Build the two-level terrain of a landing site: a measured base, plus a fine detail patch.

- **Outer grid:** the measured height map, resampled on a regular grid (default 2 m spacing,
  the native spacing of the Apollo 17 2 m product) over a square around the site. The cells
  under the detail patch are left out of its mesh.
- **Detail patch:** a finer grid (default 0.25 m) over a smaller square around the site. Its base
  heights are the outer grid's triangles, read exactly at the finer points; craters and rocks
  are added on top. Their effect fades to zero over ``detail_taper_width_m`` at the patch edge,
  so the patch border lies exactly on the outer mesh's edges: the two meshes meet without a gap.

The patch edges and the outer grid lines coincide because the patch half-size is a multiple of
the outer spacing, and the outer spacing is a multiple of the fine spacing (both are checked).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy
import torch

from lunarsim.terrain.surface_detail import (
    CraterPopulation,
    Craters,
    RockPopulation,
    Rocks,
    draw_craters,
    draw_rocks,
    surface_detail_heights,
)
from lunarsim.terrain.terrain_height_field import RegularHeightGrid, TerrainHeightField, grid_triangles, grid_vertices

# A function that returns base heights (m) at local east/north points (numpy arrays).
BaseHeightFunction = Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]


class TerrainBuildError(ValueError):
    """The terrain settings do not fit together."""


@dataclass(frozen=True)
class MeasuredTerrainSettings:
    """Sizes and spacings of the two grids, and the surface detail to add."""

    outer_half_size_m: float = 600.0
    outer_cell_size_m: float = 2.0
    detail_half_size_m: float = 80.0
    detail_cell_size_m: float = 0.25
    detail_taper_width_m: float = 5.0
    craters: CraterPopulation | None = None
    rocks: RockPopulation | None = None
    seed: int = 0

    def check(self) -> None:
        """Raise :class:`TerrainBuildError` if the grids cannot meet without a gap."""

        def is_multiple(value: float, step: float) -> bool:
            """Return True if ``value`` is an integer multiple of ``step`` (to 1e-9)."""
            ratio = value / step
            return abs(ratio - round(ratio)) < 1e-9 and round(ratio) >= 1

        if not is_multiple(self.outer_half_size_m, self.outer_cell_size_m):
            raise TerrainBuildError("outer_half_size_m must be a multiple of outer_cell_size_m")
        if not is_multiple(self.detail_half_size_m, self.outer_cell_size_m):
            raise TerrainBuildError("detail_half_size_m must be a multiple of outer_cell_size_m")
        if not is_multiple(self.outer_cell_size_m, self.detail_cell_size_m):
            raise TerrainBuildError("outer_cell_size_m must be a multiple of detail_cell_size_m")
        if self.detail_half_size_m >= self.outer_half_size_m:
            raise TerrainBuildError("the detail patch must be smaller than the outer grid")
        if 2.0 * self.detail_taper_width_m >= 2.0 * self.detail_half_size_m:
            raise TerrainBuildError("the taper is wider than the detail patch")


@dataclass
class MeasuredTerrain:
    """The built terrain: two grids (on the CPU), the hole in the outer mesh, and what was drawn."""

    outer: RegularHeightGrid
    detail: RegularHeightGrid
    outer_hole: tuple[int, int, int, int]  # (i0, j0, ni, nj) cells of the outer grid under the patch
    craters: Craters | None
    rocks: Rocks | None

    def height_field(self) -> TerrainHeightField:
        """Return the lookup of the collision surface: detail patch first, outer grid second."""
        return TerrainHeightField([self.detail, self.outer])

    def meshes(self) -> dict[str, tuple[numpy.ndarray, numpy.ndarray]]:
        """Return ``name -> (vertices (n, 3), triangles (m, 3))`` of the outer and detail meshes."""
        outer_rows, outer_columns = self.outer.heights.shape
        detail_rows, detail_columns = self.detail.heights.shape
        return {
            "outer": (grid_vertices(self.outer), grid_triangles(outer_rows, outer_columns, self.outer_hole)),
            "detail": (grid_vertices(self.detail), grid_triangles(detail_rows, detail_columns)),
        }


def smooth_step(value: numpy.ndarray) -> numpy.ndarray:
    """Return 3 t^2 - 2 t^3 of ``value`` clipped to [0, 1]: 0 at 0, 1 at 1, flat at both ends."""
    clipped = numpy.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def build_measured_terrain(base_heights: BaseHeightFunction, settings: MeasuredTerrainSettings) -> MeasuredTerrain:
    """Build the outer grid from ``base_heights`` and the detail patch on top of it."""
    settings.check()
    outer_cell, outer_half = settings.outer_cell_size_m, settings.outer_half_size_m
    outer_count = int(round(2.0 * outer_half / outer_cell)) + 1
    outer_lines = -outer_half + numpy.arange(outer_count) * outer_cell
    east, north = numpy.meshgrid(outer_lines, outer_lines)
    outer_heights = base_heights(east.ravel(), north.ravel()).reshape(east.shape)
    outer = RegularHeightGrid(-outer_half, -outer_half, outer_cell, torch.as_tensor(outer_heights, dtype=torch.float32))

    detail_cell, detail_half = settings.detail_cell_size_m, settings.detail_half_size_m
    detail_count = int(round(2.0 * detail_half / detail_cell)) + 1
    detail_lines = -detail_half + numpy.arange(detail_count) * detail_cell
    detail_east, detail_north = numpy.meshgrid(detail_lines, detail_lines)
    base = outer.height_at(
        torch.as_tensor(detail_east, dtype=torch.float32), torch.as_tensor(detail_north, dtype=torch.float32)
    ).numpy().astype(numpy.float64)

    random = numpy.random.default_rng(settings.seed)
    bounds = (-detail_half, detail_half, -detail_half, detail_half)
    craters = draw_craters(random, settings.craters, bounds) if settings.craters else None
    rocks = draw_rocks(random, settings.rocks, bounds) if settings.rocks else None
    empty_craters = Craters(*(numpy.zeros(0) for _ in range(6)))
    empty_rocks = Rocks(*(numpy.zeros(0) for _ in range(4)))
    detail_heights = surface_detail_heights(
        -detail_half, -detail_half, detail_cell, base.shape, craters or empty_craters, rocks or empty_rocks
    )
    distance_to_edge = numpy.minimum(detail_half - numpy.abs(detail_east), detail_half - numpy.abs(detail_north))
    taper = smooth_step(distance_to_edge / settings.detail_taper_width_m)
    detail = RegularHeightGrid(
        -detail_half,
        -detail_half,
        detail_cell,
        torch.as_tensor(base + taper * detail_heights, dtype=torch.float32),
    )
    first = int(round((outer_half - detail_half) / outer_cell))
    width = int(round(2.0 * detail_half / outer_cell))
    return MeasuredTerrain(outer=outer, detail=detail, outer_hole=(first, first, width, width), craters=craters, rocks=rocks)
