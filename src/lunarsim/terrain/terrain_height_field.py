# SPDX-License-Identifier: BSD-3-Clause
"""Batched, exact height lookup on the triangle meshes that PhysX collides with.

A terrain is made of one or more regular height grids ("levels"). Each grid is triangulated in
the same way as its PhysX mesh, so the lookup returns the height of the collision surface itself,
not an approximation of it:

    corners of cell (i, j):  a = (i, j)   b = (i + 1, j)   c = (i, j + 1)   d = (i + 1, j + 1)
    two triangles per cell:  (a, b, d) and (a, d, c), both counter-clockwise seen from above

All lookups are torch operations on the batch, with no Python loop over points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor


@dataclass
class RegularHeightGrid:
    """Heights on a regular grid: ``heights[j, i]`` is the height at (origin_x + i d, origin_y + j d)."""

    origin_x: float
    origin_y: float
    cell_size: float
    heights: Tensor  # (ny, nx)

    @property
    def end_x(self) -> float:
        """Return the largest x of the grid, m."""
        return self.origin_x + (self.heights.shape[1] - 1) * self.cell_size

    @property
    def end_y(self) -> float:
        """Return the largest y of the grid, m."""
        return self.origin_y + (self.heights.shape[0] - 1) * self.cell_size

    def contains(self, x: Tensor, y: Tensor) -> Tensor:
        """Return True where (x, y) lies inside the grid (edges included)."""
        return (x >= self.origin_x) & (x <= self.end_x) & (y >= self.origin_y) & (y <= self.end_y)

    def to(self, device: torch.device | str) -> RegularHeightGrid:
        """Return a copy of the grid on ``device``."""
        return RegularHeightGrid(self.origin_x, self.origin_y, self.cell_size, self.heights.to(device))

    def _cell_and_fractions(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return cell indices (i, j) and the in-cell fractions (u, v), clamped to the grid.

        Non-finite coordinates (for example a LiDAR pixel without a return) are mapped to the grid
        corner, so they never index outside the grid; callers mask those points themselves.
        """
        number_y, number_x = self.heights.shape
        column = torch.nan_to_num((x - self.origin_x) / self.cell_size, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, number_x - 1.0)
        row = torch.nan_to_num((y - self.origin_y) / self.cell_size, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, number_y - 1.0)
        i = column.floor().long().clamp(max=number_x - 2)
        j = row.floor().long().clamp(max=number_y - 2)
        return i, j, column - i, row - j

    def _corner_heights(self, i: Tensor, j: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the heights at the four corners a, b, c, d of cells (i, j)."""
        heights = self.heights
        return heights[j, i], heights[j, i + 1], heights[j + 1, i], heights[j + 1, i + 1]

    def height_at(self, x: Tensor, y: Tensor) -> Tensor:
        """Return the height of the triangulated surface at (x, y); points outside are clamped to the edge."""
        i, j, u, v = self._cell_and_fractions(x, y)
        height_a, height_b, height_c, height_d = self._corner_heights(i, j)
        lower_triangle = height_a + u * (height_b - height_a) + v * (height_d - height_b)
        upper_triangle = height_a + u * (height_d - height_c) + v * (height_c - height_a)
        return torch.where(u >= v, lower_triangle, upper_triangle)

    def gradient_at(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Return (dz/dx, dz/dy) of the triangle under (x, y)."""
        i, j, u, v = self._cell_and_fractions(x, y)
        height_a, height_b, height_c, height_d = self._corner_heights(i, j)
        lower = u >= v
        slope_x = torch.where(lower, height_b - height_a, height_d - height_c) / self.cell_size
        slope_y = torch.where(lower, height_d - height_b, height_c - height_a) / self.cell_size
        return slope_x, slope_y


class TerrainHeightField:
    """Several height grids, finest first. A point uses the first grid that contains it.

    The last grid is the fallback for points outside every grid (its edge heights are used).
    """

    def __init__(self, levels: list[RegularHeightGrid]) -> None:
        """Store the grids, finest first."""
        if not levels:
            raise ValueError("A TerrainHeightField needs at least one grid")
        self.levels = levels

    def to(self, device: torch.device | str) -> TerrainHeightField:
        """Return a copy with all grids on ``device``."""
        return TerrainHeightField([level.to(device) for level in self.levels])

    def _select(self, x: Tensor, y: Tensor) -> list[Tensor]:
        """Return one boolean mask per level: which points that level answers."""
        unanswered = torch.ones_like(x, dtype=torch.bool)
        masks = []
        for level in self.levels[:-1]:
            mask = unanswered & level.contains(x, y)
            masks.append(mask)
            unanswered = unanswered & ~mask
        masks.append(unanswered)
        return masks

    def height_at(self, x: Tensor, y: Tensor) -> Tensor:
        """Return the ground height at world (x, y), any shape."""
        result = torch.zeros_like(x)
        for level, mask in zip(self.levels, self._select(x, y)):
            result = torch.where(mask, level.height_at(x, y), result)
        return result

    def gradient_at(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Return the ground slope (dz/dx, dz/dy) at world (x, y)."""
        slope_x = torch.zeros_like(x)
        slope_y = torch.zeros_like(x)
        for level, mask in zip(self.levels, self._select(x, y)):
            level_x, level_y = level.gradient_at(x, y)
            slope_x = torch.where(mask, level_x, slope_x)
            slope_y = torch.where(mask, level_y, slope_y)
        return slope_x, slope_y


def grid_triangles(number_y: int, number_x: int, hole: tuple[int, int, int, int] | None = None) -> numpy.ndarray:
    """Return the triangle vertex indices (n_triangles, 3) of an ``number_y x number_x`` vertex grid.

    Vertex (i, j) has index ``j * number_x + i``. ``hole = (i0, j0, ni, nj)`` removes the cells
    i0 <= i < i0 + ni, j0 <= j < j0 + nj (another, finer mesh fills them).
    """
    cell_i, cell_j = numpy.meshgrid(numpy.arange(number_x - 1), numpy.arange(number_y - 1))
    cell_i, cell_j = cell_i.ravel(), cell_j.ravel()
    if hole is not None:
        i0, j0, ni, nj = hole
        inside = (cell_i >= i0) & (cell_i < i0 + ni) & (cell_j >= j0) & (cell_j < j0 + nj)
        cell_i, cell_j = cell_i[~inside], cell_j[~inside]
    corner_a = cell_j * number_x + cell_i
    corner_b, corner_c, corner_d = corner_a + 1, corner_a + number_x, corner_a + number_x + 1
    return numpy.concatenate(
        (numpy.stack((corner_a, corner_b, corner_d), axis=1), numpy.stack((corner_a, corner_d, corner_c), axis=1))
    ).astype(numpy.int64)


def grid_vertices(grid: RegularHeightGrid) -> numpy.ndarray:
    """Return the (n_vertices, 3) vertex positions of a grid, in the index order of :func:`grid_triangles`."""
    heights = grid.heights.detach().cpu().numpy().astype(numpy.float64)
    number_y, number_x = heights.shape
    x = grid.origin_x + numpy.arange(number_x) * grid.cell_size
    y = grid.origin_y + numpy.arange(number_y) * grid.cell_size
    grid_x, grid_y = numpy.meshgrid(x, y)
    return numpy.stack((grid_x.ravel(), grid_y.ravel(), heights.ravel()), axis=1)
