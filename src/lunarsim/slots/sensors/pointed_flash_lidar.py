# SPDX-License-Identifier: BSD-3-Clause
"""A flash LiDAR on a pointing mount that keeps the landing area in view (scenario S6, milestone 4).

The body-fixed LiDAR of milestone 2 looks 30 deg ahead of straight down, so it sees the landing
area only for a moment of an approach. A hazard-detection LiDAR is pointed at the landing area
instead. This model points the boresight at a point of the site frame (``pointing.target_m``,
default the nominal target), with a random pointing error per frame, and can step the boresight
over a mosaic (``pointing.mosaic``: rows x columns of frames, ``step_deg`` apart) to cover an
area larger than one field of view. Two messages:

- ``measurement.lidar``: slant range per pixel (NaN without a return), as the body-fixed LiDAR;
- ``measurement.lidar_pointing``: the boresight unit vector in the body frame of that frame (the
  mount's angle readout), so navigation can rebuild every pixel direction.

Pixel directions follow :func:`pixel_directions_body` on both sides. Ranges come from
:func:`march_rays_to_ground`, which finds the **first** ground crossing even for rays that skim
the ground at a low angle (an approach looks at the site from about 16 deg above the horizon,
[D-6846] p. 10), where the Newton steps of ``ray_cast_ground`` may jump past a rock.

Model choices: the mount has no angle limits and no lag; the pointing error is the onboard
knowledge error of the line of sight; the sensor sits at the centre of mass.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.core.message_types import Measurement, MessageKey, SurfaceGeometry, TruthState
from lunarsim.core.rotation_math import rotate_vector_body_to_parent, rotate_vector_parent_to_body
from lunarsim.slots.sensors.sensor_models import SampleSchedule

LIDAR_POINTING_KEY = MessageKey.MEASUREMENT_PREFIX + "lidar_pointing"
BISECTION_STEPS = 12


def flash_pixel_tangents(pixels_per_side: int, field_of_view_deg: float, device=None) -> tuple[Tensor, Tensor]:
    """Return the tangents (a, b) of the pixel centres across (right) and up the field of view, each (P,)."""
    half = math.radians(field_of_view_deg) / 2.0
    centres = -half + (torch.arange(pixels_per_side, dtype=torch.float32, device=device) + 0.5) * (2.0 * half / pixels_per_side)
    tangents = torch.tan(centres)
    rows, columns = torch.meshgrid(tangents, tangents, indexing="ij")
    return columns.reshape(-1), rows.reshape(-1)


def boresight_frame(boresight: Tensor) -> tuple[Tensor, Tensor]:
    """Return unit (right, up) vectors perpendicular to unit ``boresight`` (..., 3), in the same frame.

    ``right = boresight x reference`` with the body +x axis as reference, or +y when the
    boresight is nearly along x, so the frame is defined for every pointing.
    """
    reference_x = torch.zeros_like(boresight)
    reference_x[..., 0] = 1.0
    reference_y = torch.zeros_like(boresight)
    reference_y[..., 1] = 1.0
    near_x = boresight[..., 0].abs().unsqueeze(-1) > 0.95
    reference = torch.where(near_x, reference_y, reference_x)
    right = torch.nn.functional.normalize(torch.cross(boresight, reference, dim=-1), dim=-1)
    up = torch.cross(right, boresight, dim=-1)
    return right, up


def pixel_directions_body(boresight_body: Tensor, tangent_right: Tensor, tangent_up: Tensor) -> Tensor:
    """Return unit pixel directions (N, P, 3) in the body frame for boresights (N, 3) and tangents (P,)."""
    right, up = boresight_frame(boresight_body)
    directions = (
        boresight_body.unsqueeze(1)
        + tangent_right.view(1, -1, 1) * right.unsqueeze(1)
        + tangent_up.view(1, -1, 1) * up.unsqueeze(1)
    )
    return torch.nn.functional.normalize(directions, dim=-1)


def height_bounds(geometry: SurfaceGeometry) -> tuple[float, float]:
    """Return the lowest and highest ground height of the surface (world z), m."""
    if geometry.height_field is None:
        return 0.0, 0.0
    lows = [float(level.heights.min()) for level in geometry.height_field.levels]
    highs = [float(level.heights.max()) for level in geometry.height_field.levels]
    return min(lows), max(highs)


MAXIMUM_GROUND_SLOPE = 2.0  # model choice: steepest ground the adaptive march trusts (63 deg; rock flanks are steeper locally)


def _march_iteration(origin, direction, geometry, distance, previous, found, below_at, end, safe_rate, step: float):
    """One step of the adaptive march for every ray; returns the updated state and the done mask."""
    point = origin + distance.unsqueeze(-1) * direction
    gap = point[..., 2] - geometry.ground_height_at(point[..., 0], point[..., 1])
    under = gap <= 0.0
    below_at = torch.where(under & ~found, distance, below_at)
    previous = torch.where(~found & ~under, distance, previous)
    found = found | under
    done = found | (distance >= end)
    advance = torch.maximum(gap.clamp_min(0.0) / safe_rate, torch.full_like(gap, step))
    distance = torch.where(done, distance, (distance + advance).clamp(max=end))
    return distance, previous, found, below_at, done


def _bisection_iteration(origin, direction, geometry, above, below):
    """One bisection step between a point above and a point below the ground, for every ray."""
    middle = 0.5 * (above + below)
    point = origin + middle.unsqueeze(-1) * direction
    under = point[..., 2] <= geometry.ground_height_at(point[..., 0], point[..., 1])
    return torch.where(under, above, middle), torch.where(under, middle, below)


_STEP_FUNCTIONS = {"march": _march_iteration, "bisection": _bisection_iteration}


def enable_compiled_march() -> None:
    """Compile the march and bisection steps with ``torch.compile`` (GPU runs; each step is then a few fused kernels)."""
    _STEP_FUNCTIONS["march"] = torch.compile(_march_iteration, dynamic=False)
    _STEP_FUNCTIONS["bisection"] = torch.compile(_bisection_iteration, dynamic=False)


def march_rays_to_ground(
    origin: Tensor, direction: Tensor, geometry: SurfaceGeometry, maximum_range: float, step: float,
    bounds: tuple[float, float] | None = None, maximum_steps: int = 2000,
) -> tuple[Tensor, Tensor]:
    """Return (range, hit) of the first ground crossing of rays (..., 3) from world ``origin`` along unit ``direction``.

    The march runs only between the heights where a ray can meet the ground (``bounds``: lowest
    and highest ground). Each step is the larger of ``step`` and the distance the ray can surely
    fly without meeting ground no steeper than ``MAXIMUM_GROUND_SLOPE``: with a height gap ``g``
    above the ground, ``g / (down + slope * horizontal)``. Far from the ground the steps are long;
    near it they shrink to ``step``. The crossing is then refined by bisection. A ray that points
    up, or meets the ground beyond ``maximum_range``, does not hit.
    """
    low, high = bounds if bounds is not None else height_bounds(geometry)
    down = (-direction[..., 2]).clamp_min(1e-4)
    horizontal = direction[..., :2].norm(dim=-1)
    pointing_down = -direction[..., 2] > 0.01
    if geometry.height_field is None:
        distance = ((origin[..., 2] - 0.0) / down).clamp_min(0.0)
        hit = pointing_down & (distance <= maximum_range)
        return distance, hit
    start = ((origin[..., 2] - high) / down).clamp_min(0.0)
    end = ((origin[..., 2] - low) / down).clamp(max=maximum_range)
    safe_rate = down + MAXIMUM_GROUND_SLOPE * horizontal
    distance, previous = start.clone(), start.clone()
    found = torch.zeros_like(start, dtype=torch.bool)
    below_at = torch.full_like(start, math.inf)
    march = _STEP_FUNCTIONS["march"]
    for index in range(maximum_steps):
        distance, previous, found, below_at, done = march(origin, direction, geometry, distance, previous, found, below_at, end, safe_rate, step)
        if index % 8 == 7 and bool(done.all()):
            break
    # Bisection between the last point above and the first point below the ground.
    above, below = previous, torch.where(found, below_at, previous)
    bisection = _STEP_FUNCTIONS["bisection"]
    for _ in range(BISECTION_STEPS):
        above, below = bisection(origin, direction, geometry, above, below)
    distance = 0.5 * (above + below)
    hit = pointing_down & found & (distance <= maximum_range)
    return distance, hit


class PointedFlashLidar:
    """Flash LiDAR on a pointing mount, with a mosaic and per-frame pointing error.

    Parameters (``lidar`` section with a ``pointing`` block): ``rate_hz``, ``field_of_view_deg``,
    ``pixels_per_side``, ``minimum_range_m``, ``maximum_range_m``, ``range_noise_m``,
    ``range_noise_per_m``, ``range_noise_table_m`` (optional list of ``[range_m, sigma_m]``
    that replaces the two before, see :meth:`range_noise_sigma`), ``ray_march_step_m`` (default 0.5), ``compile`` (default true: compile
    the march on the GPU), and ``pointing``: ``target_m``
    (site-frame point, default ``[0, 0, 0]``), ``error_deg`` (one sigma per axis, per frame),
    ``mosaic`` (``rows``, ``columns``, ``step_deg``; default one frame).
    """

    def __init__(self, parameters: dict, count: int, device) -> None:
        """Read the sensor and mount parameters and build the pixel grid."""
        self.schedule = SampleSchedule(float(parameters["rate_hz"]), count, device)
        self.minimum_range = float(parameters["minimum_range_m"])
        self.maximum_range = float(parameters["maximum_range_m"])
        self.range_noise_offset = float(parameters["range_noise_m"])
        self.range_noise_slope = float(parameters["range_noise_per_m"])
        table = parameters.get("range_noise_table_m")
        self.range_noise_table: tuple[Tensor, Tensor] | None = None
        if table:
            pairs = sorted((float(pair[0]), float(pair[1])) for pair in table)
            self.range_noise_table = (
                torch.tensor([pair[0] for pair in pairs], device=device),
                torch.tensor([pair[1] for pair in pairs], device=device),
            )
        self.march_step = float(parameters.get("ray_march_step_m", 0.5))
        if bool(parameters.get("compile", True)) and not str(device).startswith("cpu"):
            enable_compiled_march()  # 7.1 -> 0.7 ms per frame of 16,000 rays (RTX 5060)
        self.tangent_right, self.tangent_up = flash_pixel_tangents(
            int(parameters["pixels_per_side"]), float(parameters["field_of_view_deg"]), device
        )
        self.number_of_beams = int(self.tangent_right.shape[0])
        pointing = dict(parameters.get("pointing") or {})
        self.target = torch.tensor([float(value) for value in pointing.get("target_m", [0.0, 0.0, 0.0])], device=device)
        self.pointing_error = math.radians(float(pointing.get("error_deg", 0.0)))
        mosaic = dict(pointing.get("mosaic") or {})
        rows, columns = int(mosaic.get("rows", 1)), int(mosaic.get("columns", 1))
        step = math.radians(float(mosaic.get("step_deg", 0.0)))
        offsets = [((column - (columns - 1) / 2.0) * step, (row - (rows - 1) / 2.0) * step) for row in range(rows) for column in range(columns)]
        self.mosaic_offsets = torch.tensor(offsets, dtype=torch.float32, device=device)  # (F, 2) angles right, up
        self.frame_index = torch.zeros(count, dtype=torch.long, device=device)
        self.last_value = torch.full((count, self.number_of_beams), float("nan"), device=device)
        self.last_stamp = torch.full((count,), -1.0, device=device)
        self.last_boresight = torch.zeros(count, 3, device=device)
        self.last_boresight[:, 2] = -1.0
        self.bounds: tuple[float, float] | None = None

    def range_noise_sigma(self, distance: Tensor) -> Tensor:
        """Return the one-sigma range noise (m) at slant ``distance`` (m), same shape.

        With ``range_noise_table_m`` (pairs of range and sigma, e.g. measured RMS errors of a
        flight unit) the sigma is interpolated linearly in range and held flat beyond the first
        and last pair; otherwise it is ``range_noise_m + range_noise_per_m * distance``.
        """
        if self.range_noise_table is None:
            return self.range_noise_offset + self.range_noise_slope * distance
        ranges, sigmas = self.range_noise_table
        if ranges.shape[0] == 1:
            return torch.full_like(distance, float(sigmas[0]))
        clamped = distance.clamp(float(ranges[0]), float(ranges[-1]))
        upper = torch.searchsorted(ranges, clamped.contiguous(), right=False).clamp(1, ranges.shape[0] - 1)
        lower = upper - 1
        fraction = (clamped - ranges[lower]) / (ranges[upper] - ranges[lower])
        return sigmas[lower] + fraction * (sigmas[upper] - sigmas[lower])

    def reset(self, env_ids: Tensor) -> None:
        """Start the schedule and the mosaic again and forget the last frame."""
        self.schedule.reset(env_ids)
        self.frame_index[env_ids] = 0
        self.last_value[env_ids] = float("nan")
        self.last_stamp[env_ids] = -1.0

    def boresight_world(self, truth: TruthState, geometry: SurfaceGeometry, generator: torch.Generator) -> Tensor:
        """Return this frame's unit boresight in the world frame: at the target, plus mosaic step and error."""
        count = truth.position.shape[0]
        line_of_sight = torch.nn.functional.normalize(
            (geometry.site_origin + self.target) - (truth.position + geometry.site_origin), dim=1
        )
        right, up = boresight_frame(line_of_sight)
        offset = self.mosaic_offsets[self.frame_index % self.mosaic_offsets.shape[0]]  # (N, 2)
        if self.pointing_error > 0.0:
            offset = offset + self.pointing_error * torch.randn(count, 2, generator=generator, device=offset.device)
        pointed = line_of_sight + torch.tan(offset[:, 0:1]) * right + torch.tan(offset[:, 1:2]) * up
        return torch.nn.functional.normalize(pointed, dim=1)

    def measure(self, truth: TruthState, geometry: SurfaceGeometry, generator: torch.Generator) -> tuple[Measurement, Measurement]:
        """Return (ranges, pointing) of the newest frame; ``valid`` only in frame steps."""
        count, device = truth.position.shape[0], truth.position.device
        due = self.schedule.due(truth.time)
        if bool(due.any()):
            if self.bounds is None:
                self.bounds = height_bounds(geometry)
            boresight_body = rotate_vector_parent_to_body(truth.orientation_quaternion, self.boresight_world(truth, geometry, generator))
            directions_body = pixel_directions_body(boresight_body, self.tangent_right, self.tangent_up)
            beams = self.number_of_beams
            orientation = truth.orientation_quaternion.unsqueeze(1).expand(count, beams, 4)
            directions = rotate_vector_body_to_parent(orientation, directions_body)
            origin = (truth.position + geometry.site_origin).unsqueeze(1).expand(count, beams, 3)
            distance, hit = march_rays_to_ground(origin, directions, geometry, self.maximum_range, self.march_step, self.bounds)
            hit = hit & (distance >= self.minimum_range)
            sigma = self.range_noise_sigma(distance)
            noisy = distance + sigma * torch.randn(count, beams, generator=generator, device=device)
            value = torch.where(hit, noisy, torch.full_like(distance, float("nan")))
            self.last_value = torch.where(due.unsqueeze(1), value, self.last_value)
            self.last_boresight = torch.where(due.unsqueeze(1), boresight_body, self.last_boresight)
            self.last_stamp = torch.where(due, truth.time, self.last_stamp)
            self.frame_index = self.frame_index + due.long()
        valid = due & torch.isfinite(self.last_value).any(dim=1)
        sigma = self.range_noise_sigma(torch.nan_to_num(self.last_value, nan=0.0))
        ranges = Measurement(value=self.last_value.clone(), valid=valid, stamp=self.last_stamp.clone(), covariance=sigma**2)
        pointing = Measurement(
            value=self.last_boresight.clone(), valid=valid.clone(), stamp=self.last_stamp.clone(),
            covariance=torch.full((count, 3), self.pointing_error**2 + 1e-12, device=device),
        )
        return ranges, pointing
