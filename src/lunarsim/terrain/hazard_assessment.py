# SPDX-License-Identifier: BSD-3-Clause
"""Is the ground under a lander footprint safe? One batched check used on both sides of the loop.

The same function judges a footprint on the **onboard** hazard map (``nav: hazard_map``, heights
built from LiDAR returns) and on the **true** terrain (``metrics: site_safety``, heights sampled
from the surface's exact height lookup). The criteria are the Apollo LM landing-gear design
surface, NASA TN D-6850 (Rogers 1972), p. 3:

    "a mean surface slope of 6 deg or less and an effective slope of 12 deg or less, including
    the effects of depressions or protuberances (or both) and footpad penetration. [...] within
    the landing-gear footprint, the vertical distance from the top of the highest protuberance
    to the bottom of the lowest depression would be 24 inches or less."

For each footprint (a square patch of heights centred on the candidate site, ``K x K`` cells,
``K`` odd) three numbers are computed:

- **mean slope**: tilt of the least-squares plane through every known cell inside the footprint
  disk (radius = footpad radius from the vehicle axis + footpad radius);
- **effective slope**: tilt of the least-squares plane through the four footpad heights, where a
  footpad height is the highest known cell under that pad (a rock under a pad lifts it); the
  worst of several headings, because the heading at touchdown is not known in advance;
- **relief**: highest minus lowest known cell inside the disk after the mean plane is removed
  (the "protuberance top to depression bottom" distance, measured from the mean surface).

A footprint is safe when all three are inside their limits, enough of the disk is known
(``minimum_coverage``) and every footpad has at least one known cell. ``margin`` is the smallest
of the three normalised margins ``(limit - value) / limit`` (negative when unsafe), for ranking.
All work is torch on the batch; there is no Python loop over footprints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

INCH = 0.0254


@dataclass
class HazardCriteria:
    """Footprint geometry and the safety limits.

    The defaults are the Apollo LM: footpad centres 167.57 in from the thrust axis and a 37 in
    dish ([D-6850] p. 6 and App. C p. 50, the values of ``configs/landers/apollo_lunar_module.yaml``),
    and the design surface of [D-6850] p. 3.
    """

    footpad_radius_from_axis_m: float = 167.57 * INCH
    footpad_diameter_m: float = 37.0 * INCH
    maximum_mean_slope_deg: float = 6.0
    maximum_effective_slope_deg: float = 12.0
    maximum_relief_m: float = 24.0 * INCH
    minimum_coverage: float = 0.5  # model choice: half of the footprint must have been seen
    headings_deg: tuple[float, ...] = (0.0, 22.5, 45.0, 67.5)  # model choice: 4 headings (the legs repeat every 90 deg)

    @property
    def footprint_radius_m(self) -> float:
        """Return the radius of the disk that holds all four footpads, m."""
        return self.footpad_radius_from_axis_m + 0.5 * self.footpad_diameter_m

    @classmethod
    def from_parameters(cls, parameters: dict | None, preset=None) -> HazardCriteria:
        """Build the criteria from a ``hazard_criteria:`` config block; the footprint comes from the lander preset if given."""
        parameters = dict(parameters or {})
        values = {}
        if preset is not None:
            values["footpad_radius_from_axis_m"] = float(preset.footpad_radius_from_thrust_axis)
            values["footpad_diameter_m"] = float(preset.footpad_diameter)
        for name in ("footpad_radius_from_axis_m", "footpad_diameter_m", "maximum_mean_slope_deg",
                     "maximum_effective_slope_deg", "maximum_relief_m", "minimum_coverage"):
            if name in parameters:
                values[name] = float(parameters[name])
        if "headings_deg" in parameters:
            values["headings_deg"] = tuple(float(value) for value in parameters["headings_deg"])
        return cls(**values)


def patch_size_for(criteria: HazardCriteria, cell_size: float) -> int:
    """Return the odd patch width K (cells) that holds the footprint disk on a grid of ``cell_size``."""
    half = math.ceil(criteria.footprint_radius_m / cell_size)
    return 2 * half + 1


class FootprintAssessor:
    """Precomputes the masks of one patch size and judges batches of footprints.

    ``cell_size`` is the spacing of the patch cells (m). The patch is ``K x K`` with
    ``K = patch_size_for(criteria, cell_size)``; cell (row, column) sits at
    ``((column - c) * cell_size, (row - c) * cell_size)`` from the centre, ``c = (K - 1) / 2``,
    rows along +y and columns along +x.
    """

    def __init__(self, criteria: HazardCriteria, cell_size: float, device: torch.device | str = "cpu") -> None:
        """Build the footprint disk, the per-heading footpad cell lists and the pad-plane solver."""
        self.criteria = criteria
        self.cell_size = float(cell_size)
        self.size = patch_size_for(criteria, self.cell_size)
        size, centre = self.size, (self.size - 1) // 2
        offsets = (torch.arange(size, device=device, dtype=torch.float32) - centre) * self.cell_size
        self.offset_y, self.offset_x = torch.meshgrid(offsets, offsets, indexing="ij")  # (K, K)
        radius = criteria.footprint_radius_m
        self.disk = (self.offset_x**2 + self.offset_y**2) <= radius**2 + 1e-9  # (K, K)

        # Footpad cells: every cell whose centre lies within the pad radius plus half a cell of the
        # pad centre, so a pad always covers at least one cell. Padded to the same count per pad.
        pad_radius = 0.5 * criteria.footpad_diameter_m + 0.5 * self.cell_size
        headings = torch.tensor(criteria.headings_deg, device=device) * (math.pi / 180.0)
        leg_angles = torch.arange(4, device=device, dtype=torch.float32) * (math.pi / 2.0)
        angles = headings.unsqueeze(1) + leg_angles.unsqueeze(0)  # (H, 4)
        pad_x = criteria.footpad_radius_from_axis_m * torch.cos(angles)
        pad_y = criteria.footpad_radius_from_axis_m * torch.sin(angles)
        distance = torch.hypot(
            self.offset_x.reshape(1, 1, -1) - pad_x.unsqueeze(2), self.offset_y.reshape(1, 1, -1) - pad_y.unsqueeze(2)
        )  # (H, 4, K*K)
        inside = distance <= pad_radius
        count = int(inside.sum(dim=2).max())
        order = torch.argsort((~inside).to(torch.int8), dim=2, stable=True)[..., :count]  # inside cells first
        self.pad_cells = order  # (H, 4, M) flat cell indices
        self.pad_cell_used = torch.gather(inside, 2, order)  # (H, 4, M)
        # Least-squares plane z = a + b x + c y through the 4 pad centres: coefficients = P @ heights.
        design = torch.stack((torch.ones_like(pad_x), pad_x, pad_y), dim=2)  # (H, 4, 3)
        self.pad_plane_solver = torch.linalg.pinv(design)  # (H, 3, 4)

    def assess(self, heights: Tensor, known: Tensor | None = None) -> dict[str, Tensor]:
        """Judge footprints. ``heights`` (B, K, K) m; ``known`` (B, K, K) bool (default: all known).

        Returns ``mean_slope_deg``, ``effective_slope_deg``, ``relief_m``, ``coverage``, ``safe``
        and ``margin``, each (B,). Unknown cells (``known`` False or non-finite heights) are left out.
        """
        batch = heights.shape[0]
        if known is None:
            known = torch.ones_like(heights, dtype=torch.bool)
        known = known & torch.isfinite(heights)
        heights = torch.where(known, heights, torch.zeros_like(heights))
        criteria = self.criteria
        disk_known = known & self.disk.unsqueeze(0)
        weight = disk_known.to(heights.dtype)
        coverage = weight.sum(dim=(1, 2)) / self.disk.sum().to(heights.dtype)

        # Mean plane: weighted least squares over the known disk cells (normal equations, 3 x 3).
        x, y = self.offset_x.unsqueeze(0), self.offset_y.unsqueeze(0)
        s1, sx, sy = weight.sum(dim=(1, 2)), (weight * x).sum(dim=(1, 2)), (weight * y).sum(dim=(1, 2))
        sxx, syy, sxy = (weight * x * x).sum(dim=(1, 2)), (weight * y * y).sum(dim=(1, 2)), (weight * x * y).sum(dim=(1, 2))
        sh, sxh, syh = (weight * heights).sum(dim=(1, 2)), (weight * x * heights).sum(dim=(1, 2)), (weight * y * heights).sum(dim=(1, 2))
        normal = torch.stack(
            (torch.stack((s1, sx, sy), dim=1), torch.stack((sx, sxx, sxy), dim=1), torch.stack((sy, sxy, syy), dim=1)), dim=1
        )
        normal = normal + 1e-6 * torch.eye(3, device=heights.device, dtype=heights.dtype)
        coefficients = torch.linalg.solve(normal, torch.stack((sh, sxh, syh), dim=1).unsqueeze(2)).squeeze(2)
        mean_slope = torch.atan(torch.hypot(coefficients[:, 1], coefficients[:, 2]))

        # Relief about the mean plane, over the known disk cells.
        residual = heights - (coefficients[:, 0, None, None] + coefficients[:, 1, None, None] * x + coefficients[:, 2, None, None] * y)
        highest = torch.where(disk_known, residual, torch.full_like(residual, -math.inf)).amax(dim=(1, 2))
        lowest = torch.where(disk_known, residual, torch.full_like(residual, math.inf)).amin(dim=(1, 2))
        relief = torch.where(coverage > 0, highest - lowest, torch.full_like(highest, math.inf))

        # Effective slope: plane through the four pad heights (highest known cell under each pad).
        flat_heights, flat_known = heights.reshape(batch, -1), known.reshape(batch, -1)
        headings, _, cells = self.pad_cells.shape
        index = self.pad_cells.reshape(1, -1).expand(batch, -1)
        pad_values = torch.gather(flat_heights, 1, index).reshape(batch, headings, 4, cells)
        pad_known = torch.gather(flat_known, 1, index).reshape(batch, headings, 4, cells) & self.pad_cell_used.unsqueeze(0)
        pad_height = torch.where(pad_known, pad_values, torch.full_like(pad_values, -math.inf)).amax(dim=3)  # (B, H, 4)
        every_pad_known = pad_known.any(dim=3).all(dim=2).all(dim=1)  # (B,)
        pad_height = torch.where(torch.isfinite(pad_height), pad_height, torch.zeros_like(pad_height))
        pad_coefficients = (self.pad_plane_solver.unsqueeze(0) @ pad_height.unsqueeze(3)).squeeze(3)  # (B, H, 3)
        effective_slope = torch.atan(torch.hypot(pad_coefficients[..., 1], pad_coefficients[..., 2])).amax(dim=1)

        degree = 180.0 / math.pi
        mean_slope_deg, effective_slope_deg = mean_slope * degree, effective_slope * degree
        margins = torch.stack(
            (
                (criteria.maximum_mean_slope_deg - mean_slope_deg) / criteria.maximum_mean_slope_deg,
                (criteria.maximum_effective_slope_deg - effective_slope_deg) / criteria.maximum_effective_slope_deg,
                (criteria.maximum_relief_m - relief) / criteria.maximum_relief_m,
            ),
            dim=1,
        )
        margin = margins.amin(dim=1)
        enough_seen = (coverage >= criteria.minimum_coverage) & every_pad_known
        safe = (margin >= 0.0) & enough_seen
        margin = torch.where(enough_seen, margin, torch.full_like(margin, -1.0))
        return {
            "mean_slope_deg": mean_slope_deg,
            "effective_slope_deg": effective_slope_deg,
            "relief_m": relief,
            "coverage": coverage,
            "safe": safe,
            "margin": margin,
        }

    def sample_patches(self, height_lookup, center_x: Tensor, center_y: Tensor) -> Tensor:
        """Return (B, K, K) heights of a lookup on patches centred at world (x, y).

        ``height_lookup`` is a height field (``height_at``) or a ``SurfaceGeometry`` (``ground_height_at``).
        """
        x = center_x.reshape(-1, 1, 1) + self.offset_x.unsqueeze(0)
        y = center_y.reshape(-1, 1, 1) + self.offset_y.unsqueeze(0)
        lookup = getattr(height_lookup, "ground_height_at", None) or height_lookup.height_at
        return lookup(x, y)


def candidate_patches(grid: Tensor, known: Tensor, patch: int, stride: int) -> tuple[Tensor, Tensor, int, int]:
    """Cut (N, H, W) grids into footprint patches centred on every ``stride``-th cell.

    Returns patches (N * Cy * Cx, K, K), their known masks, and the candidate grid shape (Cy, Cx).
    Candidate (cy, cx) is centred on grid cell (cy * stride + c, cx * stride + c), c = (K - 1) / 2.
    """
    count, height, width = grid.shape
    values = torch.nn.functional.unfold(grid.unsqueeze(1), kernel_size=patch, stride=stride)  # (N, K*K, L)
    masks = torch.nn.functional.unfold(known.unsqueeze(1).to(grid.dtype), kernel_size=patch, stride=stride) > 0.5
    rows = (height - patch) // stride + 1
    columns = (width - patch) // stride + 1
    values = values.transpose(1, 2).reshape(count * rows * columns, patch, patch)
    masks = masks.transpose(1, 2).reshape(count * rows * columns, patch, patch)
    return values, masks, rows, columns
