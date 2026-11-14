# SPDX-License-Identifier: BSD-3-Clause
"""``surface: measured_terrain`` - a real landing site from a NASA height map, with craters and rocks.

The ground is two static triangle meshes shared by all environments (see
:mod:`lunarsim.terrain.measured_terrain_builder`): the measured height map around the site, and a
fine patch in the middle with small craters and rocks drawn from published statistics. Every
environment gets its own landing target inside the patch, drawn again at every reset, so the
environments see different ground. Landers of different environments never collide with each
other (the environment filters collisions between environments), so they can share the patch.

The SITE frame of an environment has its origin on the ground at its target. The
``SurfaceGeometry`` message carries those origins and the exact height lookup of the meshes, so
the plant can report the footpad height above the ground below, and the metric can judge the
ground under the lander.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import torch
from torch import Tensor

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey, SurfaceGeometry
from lunarsim.core.registry import register_binding
from lunarsim.slots.surface.flat import GROUND_PRIM_PATH, FlatSurface
from lunarsim.terrain.lunar_height_map_reader import LunarHeightMap
from lunarsim.terrain.measured_terrain_builder import MeasuredTerrain, MeasuredTerrainSettings, build_measured_terrain
from lunarsim.terrain.surface_detail import CraterPopulation, RockPopulation

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def crater_population_from_parameters(parameters: dict | None) -> CraterPopulation | None:
    """Build the crater statistics from the ``craters:`` block of the surface config (None: no craters)."""
    if not parameters or not parameters.get("enabled", True):
        return None
    classes = parameters["degradation_classes"]
    return CraterPopulation(
        cumulative_coefficient_per_m2=float(parameters["cumulative_coefficient_per_m2"]),
        cumulative_exponent=float(parameters["cumulative_exponent"]),
        minimum_diameter_m=float(parameters["minimum_diameter_m"]),
        maximum_diameter_m=float(parameters["maximum_diameter_m"]),
        class_shares=tuple(float(entry["share"]) for entry in classes),
        class_depth_to_diameter=tuple((float(entry["depth_to_diameter"][0]), float(entry["depth_to_diameter"][1])) for entry in classes),
        rim_height_to_depth=float(parameters["rim_height_to_depth"]),
        classes_without_rim=tuple(index for index, entry in enumerate(classes) if not entry.get("has_rim", True)),
    )


def rock_population_from_parameters(parameters: dict | None) -> RockPopulation | None:
    """Build the rock statistics from the ``rocks:`` block of the surface config (None: no rocks)."""
    if not parameters or not parameters.get("enabled", True):
        return None
    return RockPopulation(
        cumulative_coefficient_per_m2=float(parameters["cumulative_coefficient_per_m2"]),
        cumulative_exponent=float(parameters["cumulative_exponent"]),
        minimum_diameter_m=float(parameters["minimum_diameter_m"]),
        maximum_diameter_m=float(parameters["maximum_diameter_m"]),
        height_to_diameter_mean=float(parameters["height_to_diameter_mean"]),
        height_to_diameter_standard_deviation=float(parameters["height_to_diameter_standard_deviation"]),
        height_to_diameter_limits=tuple(float(value) for value in parameters["height_to_diameter_limits"]),
    )


def resolve_repository_path(path: str) -> Path:
    """Return ``path`` as is if absolute, else relative to the repository root."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


@register_binding(SlotName.SURFACE, "measured_terrain")
class MeasuredTerrainSurface(FlatSurface):
    """Real terrain from a height map plus craters and rocks; one landing target per environment.

    Parameters (see ``configs/surface/apollo17_taurus_littrow.yaml`` for values and sources):
    ``height_map`` (``directory``, ``label_file``, ``data_file``), ``site`` (``latitude_deg``,
    ``longitude_deg``, ``east_offset_m``, ``north_offset_m``), ``outer_half_size_m``,
    ``outer_cell_size_m``, ``detail_half_size_m``, ``detail_cell_size_m``,
    ``detail_taper_width_m``, ``target_half_size_m``, ``craters``, ``rocks``,
    ``static_friction``, ``dynamic_friction``, ``restitution``, ``seed_offset``.
    """

    provides = frozenset({MessageKey.SURFACE_WRENCH, MessageKey.SURFACE_GEOMETRY})

    def build_terrain(self, root_seed: int) -> MeasuredTerrain:
        """Read the height map and build the two grids on the CPU (no simulator needed)."""
        get = self.parameters.get
        height_map_parameters = get("height_map", {})
        directory = resolve_repository_path(str(height_map_parameters.get("directory", "assets/terrain/apollo17")))
        site = get("site", {})
        height_map = LunarHeightMap(
            directory / str(height_map_parameters["label_file"]),
            directory / str(height_map_parameters["data_file"]),
            float(site["latitude_deg"]),
            float(site["longitude_deg"]),
        )
        east_offset = float(site.get("east_offset_m", 0.0))
        north_offset = float(site.get("north_offset_m", 0.0))
        center_height = float(height_map.sample_absolute(numpy.array([east_offset]), numpy.array([north_offset]))[0])

        def base_heights(east: numpy.ndarray, north: numpy.ndarray) -> numpy.ndarray:
            """Return measured heights around the (offset) site, with the site centre at z = 0."""
            return height_map.sample_absolute(east + east_offset, north + north_offset) - center_height

        settings = MeasuredTerrainSettings(
            outer_half_size_m=float(get("outer_half_size_m", 600.0)),
            outer_cell_size_m=float(get("outer_cell_size_m", 2.0)),
            detail_half_size_m=float(get("detail_half_size_m", 80.0)),
            detail_cell_size_m=float(get("detail_cell_size_m", 0.25)),
            detail_taper_width_m=float(get("detail_taper_width_m", 5.0)),
            craters=crater_population_from_parameters(get("craters")),
            rocks=rock_population_from_parameters(get("rocks")),
            seed=int(root_seed) + int(get("seed_offset", 0)),
        )
        target_half_size = float(get("target_half_size_m", 50.0))
        if target_half_size + 10.0 > settings.detail_half_size_m - settings.detail_taper_width_m:
            raise ValueError(
                "target_half_size_m must leave at least 10 m (a lander footprint) inside the detail patch"
            )
        return build_measured_terrain(base_heights, settings)

    def setup_scene(self, environment) -> None:
        """Build the terrain and import its two meshes as one static ground for all environments."""
        import isaaclab.sim as sim_utils
        import trimesh
        from isaaclab.terrains.utils import create_prim_from_mesh

        self.terrain = self.build_terrain(environment.scenario.root_seed)
        get = self.parameters.get
        physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(get("static_friction", 0.4)),
            dynamic_friction=float(get("dynamic_friction", 0.4)),
            restitution=float(get("restitution", 0.0)),
            friction_combine_mode="multiply",
            restitution_combine_mode="min",
        )
        visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.40, 0.37), roughness=0.95)
        sim_utils.create_prim(GROUND_PRIM_PATH, "Xform")
        for name, (vertices, triangles) in self.terrain.meshes().items():
            mesh = trimesh.Trimesh(vertices=vertices.astype(numpy.float32), faces=triangles, process=False)
            create_prim_from_mesh(
                f"{GROUND_PRIM_PATH}/{name}", mesh, visual_material=visual_material, physics_material=physics_material
            )
        environment.global_prim_paths.append(GROUND_PRIM_PATH)

    def initialize(self, environment) -> None:
        """Move the height lookup to the GPU and put every target at the patch centre until the first reset."""
        self.height_field = self.terrain.height_field().to(self.device)
        self.target_half_size = float(self.parameters.get("target_half_size_m", 50.0))
        site_origin = torch.zeros(self.number_of_environments, 3, device=self.device)
        site_origin[:, 2] = self.height_field.height_at(site_origin[:, 0], site_origin[:, 1])
        self.geometry = SurfaceGeometry(site_origin=site_origin, height_field=self.height_field)
        self.context.bus.write(MessageKey.SURFACE_GEOMETRY, self.geometry, self)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Draw a new landing target for each given environment, uniform in the target square."""
        count = len(env_ids)
        half = self.target_half_size
        target_x = (2.0 * torch.rand(count, generator=gen, device=self.device) - 1.0) * half
        target_y = (2.0 * torch.rand(count, generator=gen, device=self.device) - 1.0) * half
        target_z = self.height_field.height_at(target_x, target_y)
        self.geometry.site_origin[env_ids] = torch.stack((target_x, target_y, target_z), dim=1)
