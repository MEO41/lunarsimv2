# SPDX-License-Identifier: BSD-3-Clause
"""The onboard height grid and site map of hazard-detection navigation (scenario S6, milestone 4).

One square grid per environment, in the site frame, centred on ``center_m`` (default the nominal
target). Every LiDAR return becomes a ground point (computed by navigation from its own pose
estimate, so navigation error moves the map) and is fused into the cell it falls in with the
weight ``1 / sigma^2`` of its stated range noise. A cell's height is the weighted mean; it is
**known** once its standard error ``1 / sqrt(sum of weights)`` is at most
``maximum_cell_sigma_m``. Far returns are noisy (10 cm + 1 % of range for the flash LiDAR), so
the map fills in as the lander comes closer.

Every ``assessment_period_s`` the map judges a footprint centred on every ``candidate_spacing_m``
with :class:`~lunarsim.terrain.hazard_assessment.FootprintAssessor` and writes the site map
(:class:`~lunarsim.core.message_types.SiteMapLayout`: safe, margin, coverage).

The same class serves the truth variant of S6 ("``nav: truth`` with the true hazard map"):
:meth:`set_true_heights` fills the grid from the surface's exact height lookup, all cells known.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.core.message_types import SiteMapLayout, SurfaceGeometry
from lunarsim.terrain.hazard_assessment import FootprintAssessor, HazardCriteria, candidate_patches


class OnboardHazardMap:
    """Per-environment height grid, fused from ground points, and its site map.

    Parameters (``hazard_map`` block): ``center_m`` (site frame, default ``[0, 0]``),
    ``half_size_m`` (default 40), ``cell_size_m`` (default 0.5), ``maximum_cell_sigma_m``
    (default 0.15), ``candidate_spacing_m`` (default 2), ``assessment_period_s`` (default 1),
    ``environments_per_chunk`` (default 16, bounds the memory of one assessment), and a
    ``hazard_criteria`` block (:class:`HazardCriteria`).
    """

    def __init__(self, parameters: dict, criteria: HazardCriteria, count: int, device) -> None:
        """Size the grid and the candidate grid and allocate the fused sums."""
        get = parameters.get
        self.count, self.device = count, device
        center = [float(value) for value in get("center_m", [0.0, 0.0])]
        self.half_size = float(get("half_size_m", 40.0))
        self.cell_size = float(get("cell_size_m", 0.5))
        self.maximum_cell_sigma = float(get("maximum_cell_sigma_m", 0.15))
        self.assessment_period = float(get("assessment_period_s", 1.0))
        self.environments_per_chunk = int(get("environments_per_chunk", 16))
        self.cells = int(round(2.0 * self.half_size / self.cell_size))
        self.origin = torch.tensor([center[0] - self.half_size, center[1] - self.half_size], device=device)  # corner of cell (0, 0)
        self.assessor = FootprintAssessor(criteria, self.cell_size, device)
        self.stride = max(1, int(round(float(get("candidate_spacing_m", 2.0)) / self.cell_size)))
        patch = self.assessor.size
        if patch > self.cells:
            raise ValueError("the hazard map is smaller than one lander footprint")
        self.rows = (self.cells - patch) // self.stride + 1
        first_center = (patch - 1) // 2
        # Site-frame (x, y) of candidate (0, 0): cell centres are at origin + (index + 0.5) * cell.
        self.candidate_origin = self.origin + (first_center + 0.5) * self.cell_size
        self.candidate_spacing = self.stride * self.cell_size
        self.weight_sum = torch.zeros(count, self.cells * self.cells, device=device)
        self.weighted_height_sum = torch.zeros(count, self.cells * self.cells, device=device)
        self.site_map = torch.zeros(count, SiteMapLayout.CHANNELS, self.rows, self.rows, device=device)
        self.site_map[:, SiteMapLayout.MARGIN] = -1.0
        self.next_assessment_time = torch.zeros(count, device=device)
        self.true_heights: Tensor | None = None

    def geometry(self) -> dict:
        """Return the site-map geometry that agents need (shared data ``site_map_geometry``)."""
        return {
            "candidate_origin_m": [float(value) for value in self.candidate_origin.cpu()],
            "candidate_spacing_m": float(self.candidate_spacing),
            "rows": int(self.rows),
            "columns": int(self.rows),
            "channels": {"safe": SiteMapLayout.SAFE, "margin": SiteMapLayout.MARGIN, "coverage": SiteMapLayout.COVERAGE},
        }

    def candidate_positions(self) -> Tensor:
        """Return the site-frame (x, y) of every candidate, (rows, columns, 2)."""
        index = torch.arange(self.rows, device=self.device, dtype=torch.float32) * self.candidate_spacing
        y, x = torch.meshgrid(self.candidate_origin[1] + index, self.candidate_origin[0] + index, indexing="ij")
        return torch.stack((x, y), dim=2)

    def reset(self, env_ids: Tensor) -> None:
        """Forget the map of the given environments."""
        self.weight_sum[env_ids] = 0.0
        self.weighted_height_sum[env_ids] = 0.0
        self.site_map[env_ids] = 0.0
        self.site_map[env_ids, SiteMapLayout.MARGIN] = -1.0
        self.next_assessment_time[env_ids] = 0.0

    def add_points(self, points: Tensor, sigma: Tensor, use: Tensor) -> None:
        """Fuse ground points (N, P, 3) site frame, with range sigmas (N, P), where ``use`` (N, P)."""
        column = torch.floor((points[..., 0] - self.origin[0]) / self.cell_size).long()
        row = torch.floor((points[..., 1] - self.origin[1]) / self.cell_size).long()
        inside = use & (column >= 0) & (column < self.cells) & (row >= 0) & (row < self.cells) & torch.isfinite(points[..., 2])
        weight = torch.where(inside, 1.0 / sigma.clamp_min(1e-3) ** 2, torch.zeros_like(sigma))
        cell = torch.where(inside, row * self.cells + column, torch.zeros_like(row))
        self.weight_sum.scatter_add_(1, cell, weight)
        self.weighted_height_sum.scatter_add_(1, cell, weight * torch.nan_to_num(points[..., 2]))

    def set_true_heights(self, geometry: SurfaceGeometry, env_ids: Tensor) -> None:
        """Fill the grids of ``env_ids`` with the true ground (site frame), every cell known (truth variant)."""
        centres = self.origin + (torch.arange(self.cells, device=self.device, dtype=torch.float32) + 0.5).unsqueeze(1) * self.cell_size
        y, x = torch.meshgrid(centres[:, 1], centres[:, 0], indexing="ij")
        origin = geometry.site_origin[env_ids]
        world_x = origin[:, 0, None, None] + x.unsqueeze(0)
        world_y = origin[:, 1, None, None] + y.unsqueeze(0)
        heights = geometry.ground_height_at(world_x, world_y) - origin[:, 2, None, None]
        large_weight = 1.0e6
        self.weight_sum[env_ids] = large_weight
        self.weighted_height_sum[env_ids] = large_weight * heights.reshape(len(env_ids), -1)

    def heights_and_known(self) -> tuple[Tensor, Tensor]:
        """Return the fused heights (N, cells, cells) and the known mask."""
        known = self.weight_sum >= 1.0 / self.maximum_cell_sigma**2
        heights = self.weighted_height_sum / self.weight_sum.clamp_min(1e-12)
        shape = (self.count, self.cells, self.cells)
        return heights.reshape(shape), known.reshape(shape)

    def assess_due(self, time: Tensor) -> Tensor:
        """Judge every candidate of the environments whose assessment is due; return the mask of those."""
        due = time >= self.next_assessment_time
        if not bool(due.any()):
            return due
        heights, known = self.heights_and_known()
        indices = torch.nonzero(due).squeeze(1)
        for start in range(0, len(indices), self.environments_per_chunk):
            chunk = indices[start : start + self.environments_per_chunk]
            patches, masks, rows, columns = candidate_patches(heights[chunk], known[chunk], self.assessor.size, self.stride)
            result = self.assessor.assess(patches, masks)
            shape = (len(chunk), rows, columns)
            self.site_map[chunk, SiteMapLayout.SAFE] = result["safe"].reshape(shape).to(self.site_map.dtype)
            self.site_map[chunk, SiteMapLayout.MARGIN] = result["margin"].reshape(shape)
            self.site_map[chunk, SiteMapLayout.COVERAGE] = result["coverage"].reshape(shape)
        self.next_assessment_time = torch.where(due, time + self.assessment_period, self.next_assessment_time)
        return due


def lidar_ground_points(position: Tensor, orientation: Tensor, ranges: Tensor, directions_body: Tensor) -> Tensor:
    """Return the ground points (N, P, 3) of ranges (N, P) along body directions (N, P, 3) from a pose estimate."""
    from lunarsim.core.rotation_math import rotate_vector_body_to_parent

    count, beams = ranges.shape
    world_directions = rotate_vector_body_to_parent(orientation.unsqueeze(1).expand(count, beams, 4), directions_body)
    return position.unsqueeze(1) + torch.nan_to_num(ranges, nan=math.nan).unsqueeze(2) * world_directions
