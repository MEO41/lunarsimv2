# SPDX-License-Identifier: BSD-3-Clause
"""``nav: hazard_map`` and ``nav: truth_with_hazard_map`` - navigation that also maps the landing area (S6).

Design, scenario S6: "The navigation slot now outputs a hazard map as well as the state estimate.
The agent picks a site from that map and flies to it."

- ``hazard_map``: the ``ekf`` filter, plus an onboard height grid built from the pointed flash
  LiDAR (``measurement.lidar`` ranges and ``measurement.lidar_pointing`` boresight). Every return
  becomes a ground point through the filter's **own** pose estimate after this step's updates,
  so navigation error shifts and blurs the map as it would on a real vehicle. The estimate
  carries the site map (:class:`~lunarsim.core.message_types.SiteMapLayout`).
- ``truth_with_hazard_map``: the design's upper-limit variant ("Set ``nav: truth`` with the true
  hazard map"): the true state, and a map filled from the surface's exact heights at every reset.
  It reads ``TruthState`` like ``nav: truth``, so it is for reference runs only.

Both share the map class :class:`~lunarsim.slots.navigation.onboard_hazard_map.OnboardHazardMap`
and put its geometry into the shared data (``site_map_geometry``) for the agent. Parameters: a
``hazard_map`` block (see the map class) with a ``hazard_criteria`` block, plus the filter's
parameters for ``hazard_map``.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey, NavEstimate, SurfaceGeometry
from lunarsim.core.registry import register_binding
from lunarsim.slots.navigation.ekf import ExtendedKalmanFilterNavigation
from lunarsim.slots.navigation.onboard_hazard_map import OnboardHazardMap, lidar_ground_points
from lunarsim.slots.navigation.truth import TruthNavigation
from lunarsim.slots.sensors.pointed_flash_lidar import LIDAR_POINTING_KEY, flash_pixel_tangents, pixel_directions_body
from lunarsim.slots.sensors.terminal_suite import DOPPLER_VELOCIMETER_KEY, IMU_KEY, LIDAR_KEY, RADAR_ALTIMETER_KEY
from lunarsim.terrain.hazard_assessment import HazardCriteria


def build_hazard_map(parameters: dict, context) -> OnboardHazardMap:
    """Build the onboard map from the ``hazard_map`` block and publish its geometry in the shared data."""
    block = dict(parameters.get("hazard_map") or {})
    criteria = HazardCriteria.from_parameters(block.get("hazard_criteria"), context.shared_data.get("lander_preset"))
    hazard_map = OnboardHazardMap(block, criteria, context.number_of_environments, context.device)
    context.shared_data["site_map_geometry"] = hazard_map.geometry()
    return hazard_map


@register_binding(SlotName.NAVIGATION, "hazard_map")
class HazardMapNavigation(ExtendedKalmanFilterNavigation):
    """``ekf`` plus a LiDAR height grid and the site map of candidate landing sites."""

    requires = ExtendedKalmanFilterNavigation.requires | frozenset({LIDAR_KEY, LIDAR_POINTING_KEY})

    def __init__(self, parameters, context) -> None:
        """Build the filter, the map and the LiDAR pixel grid (from the sensor suite's parameters)."""
        super().__init__(parameters, context)
        self.hazard_map = build_hazard_map(self.parameters, context)
        lidar = (context.shared_data.get("sensor_suite_parameters") or {}).get("lidar") or {}
        if not lidar.get("pointing"):
            raise ValueError("nav: hazard_map needs a sensor suite with a pointed LiDAR (lidar.pointing)")
        self.tangent_right, self.tangent_up = flash_pixel_tangents(
            int(lidar["pixels_per_side"]), float(lidar["field_of_view_deg"]), self.device
        )

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Start the filter and forget the map of the given environments."""
        super().reset(env_ids, gen)
        self.hazard_map.reset(env_ids)

    def step(self, bus, dt: float) -> None:
        """Run the filter, fuse this step's LiDAR frame into the map, judge the sites when due, write the estimate."""
        geometry: SurfaceGeometry = bus.read(MessageKey.SURFACE_GEOMETRY, self)
        imu = bus.read(IMU_KEY, self)
        self.propagate_with(imu.value, imu.valid, dt)
        altimeter = bus.read(RADAR_ALTIMETER_KEY, self)
        self.fuse_altimeter(geometry, altimeter.value, altimeter.valid, altimeter.stamp, altimeter.covariance)
        doppler = bus.read(DOPPLER_VELOCIMETER_KEY, self)
        self.fuse_doppler(geometry, doppler.value, doppler.valid, doppler.stamp, doppler.covariance)
        self.map_lidar_frame(bus)
        self.hazard_map.assess_due(self.time)
        self.finish_step(bus, geometry, dt)

    def map_lidar_frame(self, bus) -> None:
        """Turn this step's LiDAR returns into ground points with the filter's pose and fuse them."""
        ranges = bus.read(LIDAR_KEY, self)
        pointing = bus.read(LIDAR_POINTING_KEY, self)
        if not bool(ranges.valid.any()):
            return
        directions = pixel_directions_body(pointing.value, self.tangent_right, self.tangent_up)
        state = self.filter
        points = lidar_ground_points(state.position, state.orientation, ranges.value, directions)
        sigma = ranges.covariance.clamp_min(0.0).sqrt()
        use = ranges.valid.unsqueeze(1) & torch.isfinite(ranges.value)
        self.hazard_map.add_points(points, sigma, use)

    def estimate(self, geometry: SurfaceGeometry, contact_light: Tensor) -> NavEstimate:
        """Return the filter's estimate with the site map attached."""
        return dataclasses.replace(super().estimate(geometry, contact_light), site_map=self.hazard_map.site_map.clone())


@register_binding(SlotName.NAVIGATION, "truth_with_hazard_map")
class TruthWithHazardMapNavigation(TruthNavigation):
    """The true state and the true hazard map (upper limit of S6; reads TruthState like ``nav: truth``)."""

    requires = frozenset({MessageKey.TRUTH_STATE, MessageKey.SURFACE_GEOMETRY})

    def __init__(self, parameters, context) -> None:
        """Build the map that holds the true heights."""
        super().__init__(parameters, context)
        self.hazard_map = build_hazard_map(self.parameters, context)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Fill the map of the given environments with the true ground around their (new) site."""
        geometry: SurfaceGeometry = self.context.bus.read(MessageKey.SURFACE_GEOMETRY, self)
        self.hazard_map.reset(env_ids)
        self.hazard_map.set_true_heights(geometry, env_ids)

    def step(self, bus, dt: float) -> None:
        """Write the true state with the true site map (judged once per assessment period)."""
        truth = bus.read(MessageKey.TRUTH_STATE, self)
        self.hazard_map.assess_due(truth.time)
        estimate = dataclasses.replace(self.truth_estimate(truth), site_map=self.hazard_map.site_map.clone())
        bus.write(MessageKey.NAVIGATION_ESTIMATE, estimate, self)
