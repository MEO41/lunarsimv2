# SPDX-License-Identifier: BSD-3-Clause
"""``surface: flat`` - a flat, rigid ground plane at site height zero; no plume, no sinkage."""

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import MessageKey, SurfaceGeometry, Wrench
from lunarsim.core.registry import register_binding

GROUND_PRIM_PATH = "/World/ground"


@register_binding(SlotName.SURFACE, "flat")
class FlatSurface(BindingBase):
    """Spawns one ground plane shared by all environments.

    Contact forces come from PhysX itself, so the extra surface wrench is zero. The site of each
    environment is its grid origin on the plane (z = 0).
    Parameters: ``static_friction``, ``dynamic_friction``, ``restitution``.
    """

    provides = frozenset({MessageKey.SURFACE_WRENCH, MessageKey.SURFACE_GEOMETRY})

    def initialize(self, environment) -> None:
        """Publish the site origins: the environment grid origins, on the ground plane."""
        site_origin = environment.scene.env_origins.clone()
        site_origin[:, 2] = 0.0
        self.context.bus.write(MessageKey.SURFACE_GEOMETRY, SurfaceGeometry(site_origin=site_origin), self)

    def setup_scene(self, environment) -> None:
        """Spawn the ground plane with the configured contact material."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(self.parameters.get("static_friction", 0.8)),
            dynamic_friction=float(self.parameters.get("dynamic_friction", 0.7)),
            restitution=float(self.parameters.get("restitution", 0.0)),
            friction_combine_mode="multiply",
            restitution_combine_mode="min",
        )
        spawn_ground_plane(prim_path=GROUND_PRIM_PATH, cfg=GroundPlaneCfg(physics_material=material))
        environment.global_prim_paths.append(GROUND_PRIM_PATH)

    def step(self, bus, dt: float) -> None:
        """Write a zero surface wrench."""
        bus.write(MessageKey.SURFACE_WRENCH, Wrench.zeros(self.number_of_environments, self.device), self)
