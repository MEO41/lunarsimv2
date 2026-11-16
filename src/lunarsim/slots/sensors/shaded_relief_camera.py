# SPDX-License-Identifier: BSD-3-Clause
"""``camera_cheap`` - a downward camera that sees the terrain as shaded relief (scenario S5, milestone 6).

Design S5: "The camera is slow. Train with ``camera_cheap`` (a shaded height map rendered on the
GPU). Run the final tests with ``camera_rtx``." This is the cheap one: every pixel is a ray from
the camera (at the centre of mass, boresight along body -z, model choice) to the true ground
(:func:`~lunarsim.slots.sensors.pointed_flash_lidar.march_rays_to_ground`), and its brightness is
the Lambert shading of the ground there under the sun: ``albedo * max(0, n . s)``, with ``n`` the
ground normal from the height field's gradient and ``s`` the unit vector to the sun (elevation and
azimuth from the scenario), plus white noise. Cast shadows are not modelled (model choice); at
low sun the slopes facing away from the sun are dark, which is what terrain matching uses.

Output ``measurement.camera``: the image (N, pixels^2), row-major from the top-left; pixels
without ground are NaN. Parameters: ``rate_hz``, ``pixels_per_side``, ``field_of_view_deg``,
``maximum_range_m``, ``noise``, ``sun`` (``elevation_deg``, ``azimuth_deg``: from north, clockwise,
toward the sun), ``albedo``, ``ray_march_step_m``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lunarsim.core.message_types import Measurement, MessageKey, SurfaceGeometry, TruthState
from lunarsim.core.rotation_math import rotate_vector_body_to_parent
from lunarsim.slots.sensors.pointed_flash_lidar import flash_pixel_tangents, height_bounds, march_rays_to_ground
from lunarsim.slots.sensors.sensor_models import SampleSchedule

CAMERA_KEY = MessageKey.MEASUREMENT_PREFIX + "camera"


def sun_direction(elevation_deg: float, azimuth_deg: float, device=None) -> Tensor:
    """Return the unit vector toward the sun in east-north-up axes (azimuth from north, clockwise)."""
    elevation, azimuth = math.radians(elevation_deg), math.radians(azimuth_deg)
    return torch.tensor([math.cos(elevation) * math.sin(azimuth), math.cos(elevation) * math.cos(azimuth), math.sin(elevation)], device=device)


def lambert_shading(height_lookup, x: Tensor, y: Tensor, sun: Tensor, albedo: float) -> Tensor:
    """Return ``albedo * max(0, n . sun)`` of the ground at world (x, y) for any shape."""
    slope_x, slope_y = height_lookup.ground_gradient_at(x, y)
    normal = torch.nn.functional.normalize(torch.stack((-slope_x, -slope_y, torch.ones_like(slope_x)), dim=-1), dim=-1)
    return albedo * (normal * sun).sum(dim=-1).clamp_min(0.0)


def camera_ray_directions_body(pixels_per_side: int, field_of_view_deg: float, device=None) -> Tensor:
    """Return unit pixel ray directions (P, 3) in the body frame of a camera looking along body -z.

    Image columns run along body +y and rows along body -x (top of the image toward +x), a
    pinhole camera with square pixels.
    """
    tangent_columns, tangent_rows = flash_pixel_tangents(pixels_per_side, field_of_view_deg, device)
    directions = torch.stack((-tangent_rows, tangent_columns, -torch.ones_like(tangent_rows)), dim=1)
    return torch.nn.functional.normalize(directions, dim=1)


class ShadedReliefCamera:
    """The cheap camera: ray-cast shaded relief of the true terrain (see the module text)."""

    def __init__(self, parameters: dict, count: int, device) -> None:
        """Read the camera and sun settings and build the pixel rays."""
        get = parameters.get
        self.schedule = SampleSchedule(float(get("rate_hz", 2.0)), count, device)
        self.pixels = int(get("pixels_per_side", 64))
        self.field_of_view = float(get("field_of_view_deg", 30.0))
        self.maximum_range = float(get("maximum_range_m", 20000.0))
        self.noise = float(get("noise", 0.01))
        sun = dict(get("sun") or {})
        self.sun = sun_direction(float(sun.get("elevation_deg", 10.0)), float(sun.get("azimuth_deg", 90.0)), device)
        self.albedo = float(get("albedo", 1.0))
        self.march_step = float(get("ray_march_step_m", 2.0))
        self.directions_body = camera_ray_directions_body(self.pixels, self.field_of_view, device)
        self.last_image = torch.full((count, self.pixels * self.pixels), float("nan"), device=device)
        self.last_stamp = torch.full((count,), -1.0, device=device)
        self.bounds: tuple[float, float] | None = None

    def reset(self, env_ids: Tensor) -> None:
        """Forget the last image and restart the schedule."""
        self.schedule.reset(env_ids)
        self.last_image[env_ids] = float("nan")
        self.last_stamp[env_ids] = -1.0

    def measure(self, truth: TruthState, geometry: SurfaceGeometry, generator: torch.Generator) -> Measurement:
        """Return the newest image; ``valid`` only in frame steps with ground in view."""
        count, device = truth.position.shape[0], truth.position.device
        due = self.schedule.due(truth.time.to(torch.float32))
        if bool(due.any()):
            if self.bounds is None:
                self.bounds = height_bounds(geometry)
            beams = self.directions_body.shape[0]
            orientation = truth.orientation_quaternion.to(torch.float32).unsqueeze(1).expand(count, beams, 4)
            directions = rotate_vector_body_to_parent(orientation, self.directions_body.expand(count, beams, 3))
            origin = (truth.position.to(torch.float32) + geometry.site_origin).unsqueeze(1).expand(count, beams, 3)
            distance, hit = march_rays_to_ground(origin, directions, geometry, self.maximum_range, self.march_step, self.bounds)
            ground = origin + distance.unsqueeze(-1) * directions
            ground_x = torch.where(hit, ground[..., 0], torch.zeros_like(distance))
            ground_y = torch.where(hit, ground[..., 1], torch.zeros_like(distance))
            brightness = lambert_shading(geometry, ground_x, ground_y, self.sun, self.albedo)
            brightness = brightness + self.noise * torch.randn(count, beams, generator=generator, device=device)
            image = torch.where(hit, brightness, torch.full_like(brightness, float("nan")))
            self.last_image = torch.where(due.unsqueeze(1), image, self.last_image)
            self.last_stamp = torch.where(due, truth.time.to(torch.float32), self.last_stamp)
        valid = due & torch.isfinite(self.last_image).any(dim=1)
        return Measurement(
            value=self.last_image.clone(), valid=valid, stamp=self.last_stamp.clone(),
            covariance=torch.full_like(self.last_image, self.noise**2),
        )
