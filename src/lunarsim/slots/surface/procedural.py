# SPDX-License-Identifier: BSD-3-Clause
"""``surface: procedural`` - a random height field through Isaac Lab's TerrainImporter."""

from __future__ import annotations

import math

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey, Wrench
from lunarsim.core.registry import register_binding
from lunarsim.slots.surface.flat import GROUND_PRIM_PATH, FlatSurface

MAXIMUM_HEIGHT_FIELD_CELLS = 16_000_000


@register_binding(SlotName.SURFACE, "procedural")
class ProceduralSurface(FlatSurface):
    """One rough height-field patch under all environments. Rigid contact, no plume.

    The site frame of each environment keeps z = 0 at the mean ground level; the ground under
    a lander differs from it by up to ``roughness_m``. Its ``SurfaceGeometry`` has no height
    field, so the height above ground is measured from the mean level (z = 0). This surface is
    smoke-tested only; real terrain is ``measured_terrain``.

    Parameters: ``roughness_m`` (heights are uniform in +-roughness), ``height_step_m``
    (height quantisation), ``horizontal_scale_m`` (cell size), ``feature_size_m`` (spacing of
    the random height samples; heights are interpolated in between), ``static_friction``,
    ``dynamic_friction``, ``restitution``, ``seed_offset``.
    """

    def setup_scene(self, environment) -> None:
        """Generate the height field and import it as the ground of all environments."""
        import isaaclab.sim as sim_utils
        from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg
        from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg

        get = self.parameters.get
        number_of_environments = environment.scene.num_envs
        spacing = environment.scene.cfg.env_spacing
        side = (math.ceil(math.sqrt(number_of_environments)) + 1) * spacing
        horizontal_scale = float(get("horizontal_scale_m", 0.5))
        cells = (side / horizontal_scale) ** 2
        if cells > MAXIMUM_HEIGHT_FIELD_CELLS:
            raise ValueError(
                f"The procedural surface would need {cells:.0f} height cells for {number_of_environments} "
                f"environments at {spacing} m spacing. Use fewer environments, a smaller env spacing, "
                f"or a larger 'horizontal_scale_m' (now {horizontal_scale} m)."
            )
        roughness = float(get("roughness_m", 0.15))
        generator = TerrainGeneratorCfg(
            seed=environment.scenario.root_seed + int(get("seed_offset", 0)),
            size=(side, side),
            num_rows=1,
            num_cols=1,
            horizontal_scale=horizontal_scale,
            vertical_scale=0.005,
            slope_threshold=None,
            sub_terrains={
                "rough": HfRandomUniformTerrainCfg(
                    proportion=1.0,
                    noise_range=(-roughness, roughness),
                    noise_step=float(get("height_step_m", 0.01)),
                    downsampled_scale=float(get("feature_size_m", 4.0)),
                    border_width=0.0,
                )
            },
        )
        importer_configuration = TerrainImporterCfg(
            prim_path=GROUND_PRIM_PATH,
            num_envs=number_of_environments,
            env_spacing=spacing,
            terrain_type="generator",
            terrain_generator=generator,
            use_terrain_origins=False,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=float(get("static_friction", 0.4)),
                dynamic_friction=float(get("dynamic_friction", 0.4)),
                restitution=float(get("restitution", 0.0)),
                friction_combine_mode="multiply",
                restitution_combine_mode="min",
            ),
        )
        self.terrain = TerrainImporter(importer_configuration)
        environment.global_prim_paths.append(GROUND_PRIM_PATH)

    def step(self, bus, dt: float) -> None:
        """Write a zero surface wrench (contact forces come from PhysX)."""
        bus.write(MessageKey.SURFACE_WRENCH, Wrench.zeros(self.number_of_environments, self.device), self)
