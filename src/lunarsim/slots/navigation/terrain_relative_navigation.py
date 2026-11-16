# SPDX-License-Identifier: BSD-3-Clause
"""``nav: trn_ekf`` - terrain-relative navigation: camera images matched to an onboard map (S5, milestone 6).

Design S5: "How accurately can the lander reach its target when position comes from matching
camera images to an onboard map, and not from truth?" and "The surface slot gives two things: the
true terrain for the camera, and an onboard map with added error for TRN. Thus, the map error is
a parameter and not a separate module." Here the onboard map is built by the navigation binding
from the surface's height lookup at start-up, degraded by ``map_error``: sampled at
``resolution_m`` (design: 5 m) and shifted per episode by a registration error of
``offset_m`` (design: uniform magnitude in [0, 20] m, random direction).

Each camera frame (``measurement.camera``, from ``camera_cheap``):

1. every pixel's ray from the **estimated** pose is marched onto the onboard map; its brightness
   goes into that ground cell of a ``grid_cells`` square grid centred on the image's mean ground
   point (cell size = the image's ground sampling distance, at least the map resolution);
2. the map is shaded with the same sun over that grid widened by ``search_cells`` on each side;
3. the normalised cross-correlation of the two over every shift gives a peak; a peak below
   ``minimum_correlation`` is a lost lock (no fix); otherwise the sub-cell shift is the position
   error of the estimate, and ``estimate + shift`` is a horizontal position fix;
4. the fix updates the ``ekf`` with variance ``(fix_sigma_cells * cell)^2`` per axis, if its
   normalised innovation passes the chi-square gate (99.9 %, 2 degrees of freedom: 13.8),
   otherwise it is rejected.

Health columns (after the filter's own): fix accepted this frame, fixes rejected so far, lost
locks so far, seconds since the last accepted fix. The ``ekf`` part is unchanged (altimeter and
Doppler aiding stay; design S5 lists ``[imu, radar_alt, camera_cheap]``; the Doppler stays for the
same reason as in S6).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey, NavEstimate, SurfaceGeometry
from lunarsim.core.registry import register_binding
from lunarsim.slots.navigation.ekf import ExtendedKalmanFilterNavigation
from lunarsim.slots.sensors.pointed_flash_lidar import march_rays_to_ground
from lunarsim.slots.sensors.shaded_relief_camera import CAMERA_KEY, camera_ray_directions_body, lambert_shading, sun_direction
from lunarsim.terrain.terrain_height_field import RegularHeightGrid, TerrainHeightField
from lunarsim.core.rotation_math import rotate_vector_body_to_parent

CHI_SQUARE_TWO_DEGREES_999 = 13.816


class OnboardTerrainMap:
    """The map TRN matches against: the true terrain resampled, shifted by a per-episode offset."""

    def __init__(self, geometry: SurfaceGeometry, half_size: float, resolution: float, count: int, device) -> None:
        """Sample the surface's height lookup on the map grid (around the site-frame origin of environment 0)."""
        cells = int(round(2.0 * half_size / resolution)) + 1
        coordinates = -half_size + resolution * torch.arange(cells, dtype=torch.float32, device=device)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        origin = geometry.site_origin[0]
        heights = geometry.ground_height_at(x + origin[0], y + origin[1]) - origin[2]
        self.grid = TerrainHeightField([RegularHeightGrid(-half_size, -half_size, resolution, heights)])
        self.resolution = resolution
        self.offset = torch.zeros(count, 2, device=device)

    def draw_offsets(self, env_ids: Tensor, low: float, high: float, generator: torch.Generator) -> None:
        """Draw a registration error of uniform size in [low, high] and random direction for ``env_ids``."""
        count = len(env_ids)
        size = low + (high - low) * torch.rand(count, generator=generator, device=self.offset.device)
        angle = 2.0 * math.pi * torch.rand(count, generator=generator, device=self.offset.device)
        self.offset[env_ids] = torch.stack((size * torch.cos(angle), size * torch.sin(angle)), dim=1)

    def height_at(self, x: Tensor, y: Tensor, env_offset: Tensor) -> Tensor:
        """Return map heights at site-frame (x, y) (env_offset broadcast like x)."""
        return self.grid.height_at(x - env_offset[..., 0], y - env_offset[..., 1])

    def gradient_at(self, x: Tensor, y: Tensor, env_offset: Tensor) -> tuple[Tensor, Tensor]:
        """Return the map slope at site-frame (x, y)."""
        return self.grid.gradient_at(x - env_offset[..., 0], y - env_offset[..., 1])


class _MapGeometry:
    """Adapter: the onboard map with one offset per environment, as a geometry the ray march can use."""

    def __init__(self, onboard_map: OnboardTerrainMap, offset: Tensor) -> None:
        """Keep the map and the (N, 1, 2) offsets (broadcast over the pixels)."""
        self.onboard_map, self.offset = onboard_map, offset
        self.height_field = onboard_map.grid

    def ground_height_at(self, x: Tensor, y: Tensor) -> Tensor:
        """Return map heights at (x, y)."""
        return self.onboard_map.height_at(x, y, self.offset)

    def ground_gradient_at(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Return the map slope at (x, y)."""
        return self.onboard_map.gradient_at(x, y, self.offset)


class _OffsetLookup:
    """Adapter: the map with one offset per environment, as a height lookup for the shading function."""

    def __init__(self, onboard_map: OnboardTerrainMap, offset: Tensor) -> None:
        """Keep the map and the (N, 1, 1, 2) offsets."""
        self.onboard_map, self.offset = onboard_map, offset

    def ground_gradient_at(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Return the map slope at (x, y)."""
        return self.onboard_map.gradient_at(x, y, self.offset)


def correlation_peak(observed: Tensor, filled: Tensor, reference: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return the best shift (row, column, cells, sub-cell) of ``observed`` inside ``reference`` and its normalised correlation.

    ``observed`` (N, G, G) with ``filled`` mask, ``reference`` (N, G + 2S, G + 2S). Shift (0, 0)
    means the observed grid sits at the centre of the reference.
    """
    count, size, _ = observed.shape
    weight = filled.to(observed.dtype)
    filled_count = weight.sum(dim=(1, 2)).clamp_min(1.0)
    observed_mean = (observed * weight).sum(dim=(1, 2)) / filled_count
    template = (observed - observed_mean.view(-1, 1, 1)) * weight
    template = template / template.pow(2).sum(dim=(1, 2)).sqrt().clamp_min(1e-9).view(-1, 1, 1)
    image = reference.unsqueeze(0)  # (1, N, H, W), one channel per environment
    kernel = template.unsqueeze(1)  # (N, 1, G, G)
    mask_kernel = weight.unsqueeze(1)
    product = torch.nn.functional.conv2d(image, kernel, groups=count)[0]  # sum(t * r)
    local_sum = torch.nn.functional.conv2d(image, mask_kernel, groups=count)[0]
    local_square = torch.nn.functional.conv2d(image**2, mask_kernel, groups=count)[0]
    variance = (local_square - local_sum**2 / filled_count.view(-1, 1, 1)).clamp_min(1e-12)
    correlation = product / variance.sqrt()  # t is zero-mean, so sum(t * r) = sum(t * (r - mean r))
    shifts = correlation.shape[-1]
    best = correlation.reshape(count, -1).argmax(dim=1)
    row, column = best // shifts, best % shifts
    peak = correlation.reshape(count, -1).gather(1, best.unsqueeze(1)).squeeze(1)

    def refine(index: Tensor, along_rows: bool) -> Tensor:
        """Return the parabolic sub-cell offset around the peak along one axis."""
        lower, upper = (index - 1).clamp(0, shifts - 1), (index + 1).clamp(0, shifts - 1)
        if along_rows:
            left = correlation[torch.arange(count), lower, column]
            right = correlation[torch.arange(count), upper, column]
        else:
            left = correlation[torch.arange(count), row, lower]
            right = correlation[torch.arange(count), row, upper]
        curvature = left - 2.0 * peak + right
        offset = torch.where(curvature.abs() > 1e-9, 0.5 * (left - right) / curvature, torch.zeros_like(peak))
        return offset.clamp(-0.5, 0.5)

    center = (shifts - 1) / 2.0
    return row.to(observed.dtype) + refine(row, True) - center, column.to(observed.dtype) + refine(column, False) - center, peak


@register_binding(SlotName.NAVIGATION, "trn_ekf")
class TerrainRelativeNavigation(ExtendedKalmanFilterNavigation):
    """``ekf`` plus camera-to-map position fixes (see the module text)."""

    requires = ExtendedKalmanFilterNavigation.requires | frozenset({CAMERA_KEY})
    health_dimension = ExtendedKalmanFilterNavigation.health_dimension + 4

    def __init__(self, parameters, context) -> None:
        """Read the TRN settings and the camera's pixel rays and sun (from the sensor suite parameters)."""
        super().__init__(parameters, context)
        trn = dict(self.parameters.get("trn") or {})
        self.grid_cells = int(trn.get("grid_cells", 48))
        self.search_cells = int(trn.get("search_cells", 12))
        self.minimum_correlation = float(trn.get("minimum_correlation", 0.5))
        self.fix_sigma_cells = float(trn.get("fix_sigma_cells", 0.5))
        map_error = dict(trn.get("map_error") or {})
        self.map_resolution = float(map_error.get("resolution_m", 5.0))
        self.map_half_size = float(map_error.get("half_size_m", 10000.0))
        offset = map_error.get("offset_m", [0.0, 20.0])
        self.offset_low, self.offset_high = float(offset[0]), float(offset[1])
        camera = (context.shared_data.get("sensor_suite_parameters") or {}).get("camera") or {}
        if not camera:
            raise ValueError("nav: trn_ekf needs a sensor suite with a camera section (camera_cheap)")
        self.camera_rays = camera_ray_directions_body(int(camera.get("pixels_per_side", 64)), float(camera.get("field_of_view_deg", 30.0)), self.device)
        self.pixel_angle = math.radians(float(camera.get("field_of_view_deg", 30.0))) / int(camera.get("pixels_per_side", 64))
        sun = dict(camera.get("sun") or {})
        self.sun = sun_direction(float(sun.get("elevation_deg", 10.0)), float(sun.get("azimuth_deg", 90.0)), self.device)
        self.albedo = float(camera.get("albedo", 1.0))
        count = self.number_of_environments
        self.onboard_map: OnboardTerrainMap | None = None
        self.fix_accepted = torch.zeros(count, device=self.device)
        self.fixes_rejected = torch.zeros(count, device=self.device)
        self.locks_lost = torch.zeros(count, device=self.device)
        self.last_fix_time = torch.zeros(count, device=self.device)
        self.generator: torch.Generator | None = None

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Start the filter, draw the map registration errors, clear the TRN counters."""
        super().reset(env_ids, gen)
        if self.onboard_map is None:
            geometry = self.context.bus.read(MessageKey.SURFACE_GEOMETRY, self)
            self.onboard_map = OnboardTerrainMap(geometry, self.map_half_size, self.map_resolution, self.number_of_environments, self.device)
            heights = self.onboard_map.grid.levels[0].heights
            self.map_bounds = (float(heights.min()) - 1.0, float(heights.max()) + 1.0)
        self.generator = gen
        self.onboard_map.draw_offsets(env_ids, self.offset_low, self.offset_high, gen)
        self.fix_accepted[env_ids] = 0.0
        self.fixes_rejected[env_ids] = 0.0
        self.locks_lost[env_ids] = 0.0
        self.last_fix_time[env_ids] = 0.0

    def step(self, bus, dt: float) -> None:
        """Run the filter, then fuse a TRN fix from this step's camera frame, and write the estimate."""
        from lunarsim.slots.sensors.terminal_suite import DOPPLER_VELOCIMETER_KEY, IMU_KEY, RADAR_ALTIMETER_KEY

        geometry: SurfaceGeometry = bus.read(MessageKey.SURFACE_GEOMETRY, self)
        imu = bus.read(IMU_KEY, self)
        self.propagate_with(imu.value, imu.valid, dt)
        altimeter = bus.read(RADAR_ALTIMETER_KEY, self)
        self.fuse_altimeter(geometry, altimeter.value, altimeter.valid, altimeter.stamp, altimeter.covariance)
        doppler = bus.read(DOPPLER_VELOCIMETER_KEY, self)
        self.fuse_doppler(geometry, doppler.value, doppler.valid, doppler.stamp, doppler.covariance)
        self.fix_accepted[:] = 0.0
        camera = bus.read(CAMERA_KEY, self)
        if bool(camera.valid.any()):
            self.fuse_terrain_fix(camera.value, camera.valid)
        self.finish_step(bus, geometry, dt)

    def terrain_fix(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (fix (N, 2), cell size (N,), correlation peak (N,)) from one image and the current estimate."""
        state, onboard_map = self.filter, self.onboard_map
        count, size, search = image.shape[0], self.grid_cells, self.search_cells
        position, orientation = state.position, state.orientation
        offset = onboard_map.offset
        nadir_height = onboard_map.height_at(position[:, 0], position[:, 1], offset)
        altitude = (position[:, 2] - nadir_height).clamp_min(10.0)
        cell = torch.maximum(altitude * self.pixel_angle, torch.full_like(altitude, self.map_resolution))
        rays = rotate_vector_body_to_parent(orientation.unsqueeze(1).expand(count, self.camera_rays.shape[0], 4), self.camera_rays.expand(count, -1, 3))
        # Each pixel's ground point from the estimated pose, marched onto the onboard map (an
        # approach camera looks far ahead at a pitched lander, over hundreds of metres of relief).
        geometry = _MapGeometry(onboard_map, offset.view(-1, 1, 2))
        origin = position.unsqueeze(1).expand_as(rays)
        reach, hit = march_rays_to_ground(origin, rays, geometry, 30000.0, 2.0 * self.map_resolution, self.map_bounds)
        ground_x = position[:, 0:1] + reach * rays[..., 0]
        ground_y = position[:, 1:2] + reach * rays[..., 1]
        image = torch.where(hit, image, torch.full_like(image, float("nan")))
        # The grid is centred on the mean ground point of the image (not under the lander).
        center_x = torch.nanmean(torch.where(hit, ground_x, torch.full_like(ground_x, float("nan"))), dim=1)
        center_y = torch.nanmean(torch.where(hit, ground_y, torch.full_like(ground_y, float("nan"))), dim=1)
        center_x, center_y = torch.nan_to_num(center_x, nan=0.0) + 0.0 * position[:, 0], torch.nan_to_num(center_y, nan=0.0)
        slant = torch.nanmean(torch.where(hit, reach, torch.full_like(reach, float("nan"))), dim=1).nan_to_num(nan=1.0)
        cell = torch.maximum(slant * self.pixel_angle, torch.full_like(slant, self.map_resolution))
        half = size / 2.0
        column = torch.floor((ground_x - center_x.unsqueeze(1)) / cell.unsqueeze(1) + half).long()
        row = torch.floor((ground_y - center_y.unsqueeze(1)) / cell.unsqueeze(1) + half).long()
        usable = torch.isfinite(image) & (column >= 0) & (column < size) & (row >= 0) & (row < size)
        flat = torch.where(usable, row * size + column, torch.zeros_like(row))
        sums = torch.zeros(count, size * size, device=self.device).scatter_add_(1, flat, torch.where(usable, image, torch.zeros_like(image)))
        hits = torch.zeros(count, size * size, device=self.device).scatter_add_(1, flat, usable.float())
        observed = (sums / hits.clamp_min(1.0)).reshape(count, size, size)
        filled = (hits > 0).reshape(count, size, size)
        wide = size + 2 * search
        index = torch.arange(wide, device=self.device, dtype=torch.float32) - (wide - 1) / 2.0
        grid_y, grid_x = torch.meshgrid(index, index, indexing="ij")
        reference_x = center_x[:, None, None] + grid_x * cell.view(-1, 1, 1)
        reference_y = center_y[:, None, None] + grid_y * cell.view(-1, 1, 1)
        lookup = _OffsetLookup(onboard_map, offset.view(-1, 1, 1, 2))
        reference = lambert_shading(lookup, reference_x, reference_y, self.sun, self.albedo)
        shift_row, shift_column, peak = correlation_peak(observed, filled, reference)
        # The observed scene was found shifted by (column, row) cells in the map: the estimate is off by minus that.
        fix = torch.stack((position[:, 0] + shift_column * cell, position[:, 1] + shift_row * cell), dim=1)
        return fix, cell, peak

    def fuse_terrain_fix(self, image: Tensor, valid: Tensor) -> None:
        """Make a fix from the image, gate it, and update the filter's horizontal position."""
        fix, cell, peak = self.terrain_fix(image)
        locked = valid & (peak >= self.minimum_correlation)
        self.locks_lost += (valid & ~locked).float()
        state = self.filter
        residual = fix - state.position[:, :2]
        variance = (self.fix_sigma_cells * cell) ** 2
        jacobian = torch.zeros(fix.shape[0], 2, state.scaled_covariance.shape[1], device=self.device)
        jacobian[:, 0, 0] = 1.0
        jacobian[:, 1, 1] = 1.0
        predicted_variance = torch.stack(
            (state.covariance[:, 0, 0], state.covariance[:, 1, 1]), dim=1
        ) + variance.unsqueeze(1)
        normalised = (residual**2 / predicted_variance).sum(dim=1)
        passed = locked & (normalised <= CHI_SQUARE_TWO_DEGREES_999)
        self.fixes_rejected += (locked & ~passed).float()
        use = passed.unsqueeze(1).expand(-1, 2)
        self.filter.update(torch.nan_to_num(residual), jacobian, variance.unsqueeze(1).expand(-1, 2), use)
        self.fix_accepted = passed.float()
        self.last_fix_time = torch.where(passed, self.time, self.last_fix_time)

    def estimate(self, geometry: SurfaceGeometry, contact_light: Tensor) -> NavEstimate:
        """Return the ekf estimate with the TRN health columns appended."""
        estimate = super().estimate(geometry, contact_light)
        trn = torch.stack((self.fix_accepted, self.fixes_rejected, self.locks_lost, self.time - self.last_fix_time), dim=1)
        return NavEstimate(mean=estimate.mean, covariance=estimate.covariance, health=torch.cat((estimate.health, trn), dim=1))
