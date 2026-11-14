# SPDX-License-Identifier: BSD-3-Clause
"""``surface: flat`` - a flat, rigid ground plane at site height zero; no plume, no sinkage."""

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import MessageKey, Wrench
from lunarsim.core.registry import register_binding

GROUND_PRIM_PATH = "/World/ground"


@register_binding(SlotName.SURFACE, "flat")
class FlatSurface(BindingBase):
    """Spawns one ground plane shared by all environments.

    Contact forces come from PhysX itself, so the extra surface wrench is zero.
    Parameters: ``static_friction``, ``dynamic_friction``, ``restitution``.
    """

    provides = frozenset({MessageKey.SURFACE_WRENCH})

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
